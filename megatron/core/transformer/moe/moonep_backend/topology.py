# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NVLink-coverage check for the MoonEP backend.

MoonEP maps expert memory and communication buffers of every rank of an expert-parallel (EP)
group into each peer's address space and synchronizes with device-side barriers. It has no
topology check of its own: a group that spans PCIe-only GPUs or two hosts without a multi-node
NVLink fabric fails with ``exit()``, a hang, or a device trap. This module verifies the topology
up front and fails on every rank with one actionable message instead.

Rules that shape the code:

* No rank raises before the collectives finish. Every local probe runs under ``try/except`` and
  turns failures into problem lines, and the evaluator never raises, so every rank reaches
  every collective and raises the same error.
* Two ranks are on the same kernel instance (host) iff their ``boot_id`` is equal; the hostname
  is only used in messages. Same-host pairs need NVML NVLink P2P status OK. Cross-host pairs
  need one multi-node NVLink clique and fabric memory handles.
* The result is cached per EP rank tuple only: a key that included per-rank state could hit on
  some ranks and miss on others, which would deadlock the next collective.
"""

import logging
import os
import shutil
import socket
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_VERIFIED: set = set()

# nvmlGpuP2PStatus_t.
_P2P_STATUS_NAMES = {
    0: "OK",
    1: "CHIPSET_NOT_SUPPORTED",
    2: "GPU_NOT_SUPPORTED",
    3: "IOH_TOPOLOGY_NOT_SUPPORTED",
    4: "DISABLED_BY_REGKEY",
    5: "NOT_SUPPORTED",
    6: "UNKNOWN",
}
# nvmlGpuFabricState_t / nvmlGpuFabricHealthSummary_t.
_FABRIC_STATE_COMPLETED = 3
_FABRIC_HEALTH_UNHEALTHY = 2
_FABRIC_HEALTH_LIMITED_CAPACITY = 3
# An all-zero ClusterUUID means a single-node NVSwitch system, not a multi-node clique.
_ZERO_CLUSTER_UUID = "00" * 16
_HANDLE_TYPE_ENV = "MOONEP_MEM_HANDLE_TYPE"
_VALID_HANDLE_TYPES = ("auto", "fabric", "fd")
_MIN_COMPUTE_CAPABILITY = (9, 0)

_HOW_TO_FIX = (
    "How to fix: keep --expert-model-parallel-size <= the GPUs of one NVLink domain (8 on an "
    "HGX B200 node); keep --expert-tensor-parallel-size 1 so EP groups are contiguous rank "
    "blocks; launch with torchrun so each node owns a contiguous rank block; expose every GPU of "
    "the node to NVML in each container (e.g. NVIDIA_VISIBLE_DEVICES=all) and share /tmp between "
    "the EP ranks; for multi-node NVLink (GB200 NVL72) enable IMEX so fabric memory handles work "
    "and do not set MOONEP_MEM_HANDLE_TYPE=fd."
)


def _read_boot_id() -> str:
    """Kernel-instance id; shared by all containers of one host (tests monkeypatch this)."""
    with open("/proc/sys/kernel/random/boot_id") as f:
        return f.read().strip()


def _nvml():
    # cuda.bindings ships with torch's CUDA dependencies; NVML sees every GPU of the node
    # regardless of CUDA_VISIBLE_DEVICES, so GPUs are always looked up by PCI bus id.
    from cuda.bindings import nvml

    nvml.init_v2()
    return nvml


def _local_record() -> dict:
    """Describe this rank's GPU and MoonEP capabilities. Never raises."""
    rec = dict(
        rank=dist.get_rank(),
        host=socket.gethostname(),
        boot_id=None,
        pci=None,
        cc=None,
        multicast=None,
        fabric_handles=None,
        handle_env=os.environ.get(_HANDLE_TYPE_ENV, "auto").lower(),
        fabric=None,
        fabric_error=None,
        errors=[],
    )
    try:
        rec["boot_id"] = _read_boot_id()
    except Exception as e:  # pylint: disable=broad-except
        rec["errors"].append(f"cannot read boot_id: {e!r}")
    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        rec["pci"] = f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:{props.pci_device_id:02X}.0"
        rec["cc"] = tuple(torch.cuda.get_device_capability(device))
    except Exception as e:  # pylint: disable=broad-except
        rec["errors"].append(f"cannot query the CUDA device: {e!r}")
    try:
        from moonep._C import nvl_fabric_supported, nvl_multicast_supported

        rec["multicast"] = bool(nvl_multicast_supported())
        rec["fabric_handles"] = bool(nvl_fabric_supported())
    except Exception as e:  # pylint: disable=broad-except
        rec["errors"].append(f"cannot import or query moonep._C: {e!r}")
    if rec["pci"] is None:
        return rec
    try:
        nvml = _nvml()
        handle = nvml.device_get_handle_by_pci_bus_id_v2(rec["pci"])
    except Exception as e:  # pylint: disable=broad-except
        rec["errors"].append(
            f"NVML is unavailable or cannot see this GPU ({e!r}); install cuda-bindings with the "
            "cuda.bindings.nvml module and expose all node GPUs to NVML"
        )
        return rec
    try:
        info = nvml.device_get_gpu_fabric_info_v(handle)
        rec["fabric"] = (
            int(info.state),
            int(info.status),
            bytes(info.cluster_uuid).hex(),
            int(info.clique_id),
            int(info.health_summary),
        )
    except Exception as e:  # pylint: disable=broad-except
        # Only needed for cross-host pairs; the evaluator decides whether it matters.
        rec["fabric_error"] = repr(e)
    return rec


def _probe_same_host_nvlink(records: Sequence[dict], me: int) -> List[dict]:
    """NVML NVLink P2P status from this rank's GPU to every same-host peer. Never raises."""
    mine = records[me]
    peers = [
        j
        for j, rec in enumerate(records)
        if j != me
        and mine["boot_id"] is not None
        and rec["boot_id"] == mine["boot_id"]
        and mine["pci"] is not None
        and rec["pci"] is not None
        and rec["pci"] != mine["pci"]
    ]
    if not peers:
        return []
    try:
        nvml = _nvml()
        h_me = nvml.device_get_handle_by_pci_bus_id_v2(mine["pci"])
    except Exception as e:  # pylint: disable=broad-except
        return [dict(peer=j, error=repr(e)) for j in peers]
    findings = []
    for j in peers:
        try:
            h_peer = nvml.device_get_handle_by_pci_bus_id_v2(records[j]["pci"])
            status = nvml.device_get_p2p_status(
                h_me, h_peer, nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_NVLINK
            )
            findings.append(dict(peer=j, status=int(status)))
        except Exception as e:  # pylint: disable=broad-except
            findings.append(dict(peer=j, error=repr(e)))
    return findings


def _probe_shared_tmpdir(ep_group: dist.ProcessGroup) -> List[bool]:
    """Whether every rank sees group rank 0's temp directory, as MoonEP's fd path needs.

    Collective over ``ep_group``; mirrors ``moonep.buffer._exchange_ipc_fds``. Never raises.
    """
    path = None
    if ep_group.rank() == 0:
        try:
            path = tempfile.mkdtemp(prefix="moonep_probe_")
        except Exception:  # pylint: disable=broad-except
            path = None
    obj = [path]
    dist.broadcast_object_list(obj, src=dist.get_global_rank(ep_group, 0), group=ep_group)
    path = obj[0]
    visible = bool(path) and os.path.isdir(path)
    gathered = [None] * ep_group.size()
    dist.all_gather_object(gathered, visible, group=ep_group)
    if ep_group.rank() == 0 and path:
        shutil.rmtree(path, ignore_errors=True)
    return [bool(v) for v in gathered]


def _describe(rec: dict) -> str:
    return f"{rec.get('rank')} ({rec.get('host')}, GPU {rec.get('pci')})"


def moonep_handle_path(records: Sequence[dict]) -> Tuple[bool, List[str]]:
    """Which memory-handle path MoonEP will pick for these ranks, and problems with it.

    Pure mirror of ``moonep.buffer._use_fabric_for_group``: ``fd`` never uses fabric handles,
    ``fabric`` requires them on every rank, and ``auto`` uses them iff every rank supports them.
    Returns ``(uses_fabric, problems)``, identical on every rank.
    """
    problems = []
    envs = {rec.get("rank"): rec.get("handle_env") for rec in records}
    distinct = sorted({str(v) for v in envs.values()})
    if len(distinct) > 1:
        problems.append(
            f"{_HANDLE_TYPE_ENV} differs across the EP group ({envs}); MoonEP issues different "
            "collectives per value, so it must be identical on every rank"
        )
        mode = "fd"
    else:
        mode = distinct[0] if distinct else "auto"
    if mode not in _VALID_HANDLE_TYPES:
        problems.append(f"{_HANDLE_TYPE_ENV}={mode!r} is invalid (use auto, fabric or fd)")
        mode = "fd"
    if mode == "fd":
        return False, problems
    unsupported = [rec.get("rank") for rec in records if not rec.get("fabric_handles")]
    if mode == "fabric":
        if unsupported:
            problems.append(
                f"{_HANDLE_TYPE_ENV}=fabric, but fabric memory handles are unsupported on ranks "
                f"{unsupported}"
            )
        return True, problems
    return not unsupported, problems


def _fabric_str(rec: dict) -> str:
    fabric = rec.get("fabric")
    if fabric is None:
        return f"unavailable ({rec.get('fabric_error')})"
    state, status, uuid, clique, _ = fabric
    return f"{state}/{status}/{uuid}/{clique}"


def _cross_host_reason(a: dict, b: dict, uses_fabric: bool, all_fabric_handles: bool):
    """Why ranks ``a`` and ``b`` on different hosts cannot run MoonEP, or None if they can."""
    fa, fb = a.get("fabric"), b.get("fabric")
    fabric_desc = (
        f"(fabric state/status/cluster/clique: {_fabric_str(a)} vs {_fabric_str(b)}; an all-zero "
        "ClusterUUID means a single-node NVSwitch system)"
    )
    if fa is None or fb is None:
        return f"different hosts and NVLink fabric info is unavailable {fabric_desc}"
    state_a, status_a, uuid_a, clique_a, health_a = fa
    state_b, status_b, uuid_b, clique_b, health_b = fb
    in_one_clique = (
        state_a == _FABRIC_STATE_COMPLETED
        and state_b == _FABRIC_STATE_COMPLETED
        and status_a == 0
        and status_b == 0
        and uuid_a == uuid_b
        and uuid_a != _ZERO_CLUSTER_UUID
        and clique_a == clique_b
    )
    if not in_one_clique:
        return f"different hosts and not in one multi-node NVLink clique {fabric_desc}"
    if _FABRIC_HEALTH_UNHEALTHY in (health_a, health_b):
        return f"NVLink fabric reports UNHEALTHY (health summaries {health_a} and {health_b})"
    if not uses_fabric:
        return (
            "different hosts, but MoonEP would share memory through POSIX fds, which only work "
            f"on one host ({_HANDLE_TYPE_ENV}=fd or fabric handles unsupported on some rank)"
        )
    if not all_fabric_handles:
        return "different hosts, but fabric memory handles are unsupported on some EP rank"
    return None


def evaluate_moonep_nvlink_records(
    records: Sequence[dict],
    p2p_findings: Sequence[Sequence[dict]],
    uses_fabric: bool,
    tmpdir_visible: Optional[Sequence[bool]],
) -> List[str]:
    """Turn the gathered per-rank records into problem lines. Pure; never raises.

    Args:
        records: ``_local_record()`` of every EP rank, in group-rank order.
        p2p_findings: ``_probe_same_host_nvlink`` of every EP rank, in group-rank order.
        uses_fabric: the memory-handle path from :func:`moonep_handle_path`.
        tmpdir_visible: per rank, whether group rank 0's temp dir is visible (fd path only),
            or None on the fabric path.
    """
    problems: List[str] = []
    try:
        for rec in records:
            for err in rec.get("errors") or ():
                problems.append(f"rank {_describe(rec)}: {err}")
            for field_name, what in (
                ("boot_id", "boot_id"),
                ("pci", "PCI bus id"),
                ("cc", "compute capability"),
                ("multicast", "NVSwitch multicast support"),
            ):
                if rec.get(field_name) is None:
                    problems.append(f"rank {_describe(rec)}: {what} unavailable")
            cc = rec.get("cc")
            if cc is not None and tuple(cc) < _MIN_COMPUTE_CAPABILITY:
                problems.append(
                    f"rank {_describe(rec)}: compute capability {cc[0]}.{cc[1]} < 9.0 "
                    "(MoonEP kernels need sm_90 or newer)"
                )
            if rec.get("multicast") is False:
                problems.append(
                    f"rank {_describe(rec)}: NVSwitch multicast is unsupported (MoonEP requires "
                    "it for its barrier buffers)"
                )

        all_fabric_handles = all(bool(rec.get("fabric_handles")) for rec in records)
        findings_by_pair: Dict[Tuple[int, int], dict] = {}
        for i, findings in enumerate(p2p_findings):
            for finding in findings or ():
                findings_by_pair[(i, finding.get("peer"))] = finding

        # Cross-host failures repeat for every pair of two hosts; report one example per host
        # pair plus a count.
        cross_host: Dict[Tuple[str, str], List] = {}
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a, b = records[i], records[j]
                if a.get("boot_id") is None or b.get("boot_id") is None:
                    continue
                if a["boot_id"] == b["boot_id"]:
                    if a.get("pci") is None or b.get("pci") is None:
                        continue
                    if a["pci"] == b["pci"]:
                        problems.append(
                            f"ranks {_describe(a)} and {_describe(b)} share one GPU (one GPU per "
                            "EP rank is required)"
                        )
                        continue
                    finding = findings_by_pair.get((i, j)) or findings_by_pair.get((j, i))
                    if finding is None or "status" not in finding:
                        detail = finding.get("error") if finding else "no probe result"
                        problems.append(
                            f"cannot verify NVLink between ranks {_describe(a)} and "
                            f"{_describe(b)} ({detail})"
                        )
                    elif finding["status"] != 0:
                        status = finding["status"]
                        name = _P2P_STATUS_NAMES.get(status, "UNKNOWN")
                        hint = (
                            "; NOT_SUPPORTED usually means the GPUs are connected by PCIe only"
                            if status == 5
                            else ""
                        )
                        problems.append(
                            f"ranks {_describe(a)} and {_describe(b)}: NVLink P2P status "
                            f"{name} ({status}){hint}"
                        )
                else:
                    reason = _cross_host_reason(a, b, uses_fabric, all_fabric_handles)
                    if reason is not None:
                        key = tuple(sorted((a["boot_id"], b["boot_id"])))
                        entry = cross_host.setdefault(key, [f"ranks {_describe(a)} and "
                                                            f"{_describe(b)}: {reason}", 0])
                        entry[1] += 1
        for line, count in cross_host.values():
            if count > 1:
                line += f" [and {count - 1} more rank pairs between these two hosts]"
            problems.append(line)

        if not uses_fabric and tmpdir_visible is not None:
            hidden = [
                _describe(records[i]) for i, visible in enumerate(tmpdir_visible) if not visible
            ]
            if hidden:
                problems.append(
                    f"ranks {hidden} cannot see EP group rank 0's temporary directory; MoonEP's "
                    "POSIX-fd memory sharing needs every EP rank on one host with a shared /tmp"
                )
    except Exception as e:  # pylint: disable=broad-except
        problems.append(f"internal error while evaluating the MoonEP topology records: {e!r}")
    return problems


def fabric_health_warnings(records: Sequence[dict]) -> List[str]:
    """Non-fatal notices: LIMITED_CAPACITY fabric health on a group that spans hosts."""
    boot_ids = {rec.get("boot_id") for rec in records}
    if len(boot_ids) <= 1:
        return []
    warnings_ = []
    for rec in records:
        fabric = rec.get("fabric")
        if fabric is not None and fabric[4] == _FABRIC_HEALTH_LIMITED_CAPACITY:
            warnings_.append(
                f"MoonEP: rank {_describe(rec)} reports NVLink fabric health LIMITED_CAPACITY; "
                "cross-host MoonEP traffic may run at reduced bandwidth."
            )
    return warnings_


def _format_moonep_topology_error(failed: Dict[Tuple[int, ...], List[str]]) -> str:
    blocks = []
    for ranks, problems in failed.items():
        lines = [
            "MoonEP requires every expert-parallel (EP) group to be covered by a single "
            f"NVLink/NVSwitch fabric, but EP group ranks {list(ranks)} failed the check:"
        ]
        lines += [f"  - {problem}" for problem in problems]
        blocks.append("\n".join(lines))
    return "\n".join(blocks + [_HOW_TO_FIX])


def assert_moonep_ep_group_nvlink_coverage(
    ep_group: dist.ProcessGroup,
    *,
    consensus_group: Optional[dist.ProcessGroup] = None,
    force: bool = False,
) -> None:
    """Raise on every rank unless ``ep_group`` is one NVLink domain MoonEP can use.

    Collective over ``ep_group`` (and ``consensus_group`` when given). With
    ``consensus_group``, every rank of it raises if any EP group of it failed. The result is
    cached per EP rank tuple; ``force=True`` (on every rank) re-runs the check.
    """
    ranks = tuple(dist.get_process_group_ranks(ep_group))
    if ranks in _VERIFIED and not force:
        return
    me = ep_group.rank()
    records = [None] * ep_group.size()
    dist.all_gather_object(records, _local_record(), group=ep_group)
    findings = [None] * ep_group.size()
    dist.all_gather_object(findings, _probe_same_host_nvlink(records, me), group=ep_group)
    uses_fabric, problems = moonep_handle_path(records)
    tmpdir_visible = None if uses_fabric else _probe_shared_tmpdir(ep_group)
    problems += evaluate_moonep_nvlink_records(records, findings, uses_fabric, tmpdir_visible)

    failed = {ranks: problems} if problems else {}
    if consensus_group is not None:
        gathered = [None] * dist.get_world_size(group=consensus_group)
        dist.all_gather_object(gathered, (ranks, problems), group=consensus_group)
        failed = {tuple(r): p for r, p in gathered if p}
    if failed:
        raise RuntimeError(_format_moonep_topology_error(failed))
    if me == 0:
        for warning in fabric_health_warnings(records):
            logger.warning(warning)
    _VERIFIED.add(ranks)


def _reset_moonep_topology_cache() -> None:
    """Forget verified EP groups. Call on every rank (the next check is collective)."""
    _VERIFIED.clear()
