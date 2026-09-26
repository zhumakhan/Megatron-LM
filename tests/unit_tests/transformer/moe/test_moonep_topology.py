# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NVLink-coverage check of the MoonEP backend: pure evaluator cases and live collective runs."""

import os

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.transformer.moe.moonep_backend import topology
from megatron.core.transformer.moe.moonep_backend.topology import (
    _reset_moonep_topology_cache,
    assert_moonep_ep_group_nvlink_coverage,
    evaluate_moonep_nvlink_records,
    fabric_health_warnings,
    moonep_handle_path,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.moonep_utils import requires_moonep

pytestmark = pytest.mark.internal

ZERO_UUID = "00" * 16
MNNVL_UUID = "ab" * 16
HEADER = "MoonEP requires every expert-parallel (EP) group to be covered"


def _record(rank, boot="boot-a", host="host-a", **overrides):
    rec = dict(
        rank=rank,
        host=host,
        boot_id=boot,
        pci=f"00000000:{0x10 + rank:02X}:00.0",
        cc=(10, 0),
        multicast=True,
        fabric_handles=False,
        handle_env="auto",
        fabric=(3, 0, ZERO_UUID, 0, 1),
        fabric_error=None,
        errors=[],
    )
    rec.update(overrides)
    return rec


def _same_host_findings(records, status=0):
    findings = []
    for i, a in enumerate(records):
        findings.append(
            [
                dict(peer=j, status=status)
                for j, b in enumerate(records)
                if j != i and a["boot_id"] == b["boot_id"] and a["pci"] != b["pci"]
            ]
        )
    return findings


def _evaluate(records, findings=None, tmpdir_visible=None):
    uses_fabric, problems = moonep_handle_path(records)
    if findings is None:
        findings = _same_host_findings(records)
    if tmpdir_visible is None and not uses_fabric:
        tmpdir_visible = [True] * len(records)
    return problems + evaluate_moonep_nvlink_records(
        records, findings, uses_fabric, None if uses_fabric else tmpdir_visible
    )


def _two_hosts(n=8, **overrides):
    records = []
    for r in range(n):
        boot = "boot-a" if r < n // 2 else "boot-b"
        records.append(_record(r, boot=boot, host=f"host-{r * 2 // n}", **overrides))
    return records


def _mnnvl(n=8, **overrides):
    kwargs = dict(fabric_handles=True, fabric=(3, 0, MNNVL_UUID, 7, 1))
    kwargs.update(overrides)
    return _two_hosts(n, **kwargs)


class TestEvaluator:
    def test_single_node_passes(self):
        assert _evaluate([_record(r) for r in range(8)]) == []

    def test_single_node_fabric_error_passes(self):
        records = [_record(r, fabric=None, fabric_error="NVMLError_NotSupported") for r in range(8)]
        assert _evaluate(records) == []

    def test_cross_host_zero_uuid_fails(self):
        problems = _evaluate(_two_hosts())
        assert len(problems) == 1
        assert "not in one multi-node NVLink clique" in problems[0]
        assert "[and 15 more rank pairs between these two hosts]" in problems[0]

    def test_cross_host_fabric_error_fails(self):
        records = _two_hosts(fabric=None, fabric_error="NVMLError_NotSupported")
        assert any("fabric info is unavailable" in p for p in _evaluate(records))

    def test_mnnvl_clique_passes(self):
        assert _evaluate(_mnnvl()) == []

    def test_mnnvl_unhealthy_fails(self):
        records = _mnnvl()
        records[5]["fabric"] = (3, 0, MNNVL_UUID, 7, 2)
        assert any("UNHEALTHY" in p for p in _evaluate(records))

    def test_mnnvl_limited_capacity_warns_only(self):
        records = _mnnvl()
        records[5]["fabric"] = (3, 0, MNNVL_UUID, 7, 3)
        assert _evaluate(records) == []
        assert len(fabric_health_warnings(records)) == 1
        assert fabric_health_warnings([_record(r) for r in range(8)]) == []

    def test_pcie_only_fails(self):
        records = [_record(r) for r in range(4)]
        problems = _evaluate(records, findings=_same_host_findings(records, status=5))
        assert len(problems) == 6
        assert all("NOT_SUPPORTED (5)" in p and "PCIe only" in p for p in problems)

    def test_unreadable_p2p_fails(self):
        records = [_record(r) for r in range(2)]
        findings = [[dict(peer=1, error="NVMLError_Unknown")], [dict(peer=0, error="x")]]
        problems = _evaluate(records, findings=findings)
        assert len(problems) == 1 and "cannot verify NVLink between ranks 0" in problems[0]

    def test_handle_env_mismatch_fails(self):
        records = [_record(r) for r in range(4)]
        records[0]["handle_env"] = "fd"
        uses_fabric, problems = moonep_handle_path(records)
        assert not uses_fabric
        assert len(problems) == 1 and "differs across the EP group" in problems[0]

    def test_invalid_handle_env_fails(self):
        records = [_record(r, handle_env="posix") for r in range(4)]
        assert any("is invalid" in p for p in _evaluate(records))

    def test_forced_fabric_without_support_fails(self):
        records = [_record(r, handle_env="fabric") for r in range(4)]
        assert any("fabric memory handles are unsupported" in p for p in _evaluate(records))

    def test_no_multicast_fails(self):
        records = [_record(r) for r in range(4)]
        records[2]["multicast"] = False
        assert any("multicast is unsupported" in p for p in _evaluate(records))

    def test_duplicate_gpu_fails(self):
        records = [_record(r) for r in range(4)]
        records[3]["pci"] = records[1]["pci"]
        assert any("share one GPU" in p for p in _evaluate(records))

    def test_old_compute_capability_fails(self):
        records = [_record(r) for r in range(4)]
        records[0]["cc"] = (8, 0)
        assert any("compute capability 8.0 < 9.0" in p for p in _evaluate(records))

    def test_fd_path_across_hosts_fails(self):
        problems = _evaluate(_mnnvl(handle_env="fd"))
        assert any("POSIX fds" in p for p in problems)

    def test_tmpdir_not_visible_fails(self):
        records = [_record(r) for r in range(4)]
        problems = _evaluate(records, tmpdir_visible=[True, True, False, True])
        assert len(problems) == 1 and "temporary directory" in problems[0]

    def test_missing_fields_never_raise(self):
        records = [_record(r) for r in range(4)]
        records[1].update(cc=None, pci=None, boot_id=None, multicast=None)
        records[1]["errors"] = ["cannot query the CUDA device: RuntimeError()"]
        problems = _evaluate(records)
        assert any("compute capability unavailable" in p for p in problems)
        assert any("boot_id unavailable" in p for p in problems)
        assert any("cannot query the CUDA device" in p for p in problems)


def _gather_errors(fn):
    """Run ``fn`` on every rank; return every rank's error text (None when it passed)."""
    try:
        fn()
        text = None
    except RuntimeError as e:
        text = str(e)
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, text)
    return gathered


@requires_moonep
class TestLiveCheck:
    def setup_method(self, method):
        _reset_moonep_topology_cache()

    def teardown_method(self, method):
        _reset_moonep_topology_cache()
        Utils.destroy_model_parallel()

    def _init(self, ep):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=ep)
        return ps.get_expert_model_parallel_group()

    @pytest.mark.parametrize("divisor", [1, 2])
    def test_passes_on_this_node(self, divisor):
        ep = Utils.world_size // divisor
        if ep < 2:
            pytest.skip("needs an EP group of at least 2 ranks")
        group = self._init(ep)
        world = torch.distributed.group.WORLD
        assert _gather_errors(
            lambda: assert_moonep_ep_group_nvlink_coverage(group, consensus_group=world)
        ) == [None] * Utils.world_size
        assert tuple(torch.distributed.get_process_group_ranks(group)) in topology._VERIFIED

    def test_handle_env_mismatch_raises_everywhere(self, monkeypatch):
        group = self._init(Utils.world_size)
        if torch.distributed.get_rank() == 0:
            monkeypatch.setenv("MOONEP_MEM_HANDLE_TYPE", "fd")
        errors = _gather_errors(lambda: assert_moonep_ep_group_nvlink_coverage(group))
        assert errors[0] is not None and HEADER in errors[0]
        assert "MOONEP_MEM_HANDLE_TYPE differs" in errors[0]
        assert all(e == errors[0] for e in errors)

    def test_fake_second_host_raises_everywhere(self, monkeypatch):
        group = self._init(Utils.world_size)
        if torch.distributed.get_rank() >= Utils.world_size // 2:
            monkeypatch.setattr(topology, "_read_boot_id", lambda: "another-host")
        errors = _gather_errors(lambda: assert_moonep_ep_group_nvlink_coverage(group))
        assert errors[0] is not None and "not in one multi-node NVLink clique" in errors[0]
        assert all(e == errors[0] for e in errors)

    def test_consensus_propagates_other_group_failure(self, monkeypatch):
        if Utils.world_size < 4:
            pytest.skip("needs two EP groups of at least 2 ranks")
        group = self._init(Utils.world_size // 2)
        if torch.distributed.get_rank() == Utils.world_size - 1:
            monkeypatch.setattr(topology, "_read_boot_id", lambda: "another-host")
        world = torch.distributed.group.WORLD
        errors = _gather_errors(
            lambda: assert_moonep_ep_group_nvlink_coverage(group, consensus_group=world)
        )
        assert all(e is not None for e in errors)
        upper = list(range(Utils.world_size // 2, Utils.world_size))
        assert all(f"EP group ranks {upper} failed the check" in e for e in errors)
        assert all(e == errors[0] for e in errors)

    def test_local_probe_exception_is_contained(self, monkeypatch):
        group = self._init(Utils.world_size)

        def boom():
            raise OSError("no /proc")

        if torch.distributed.get_rank() == 0:
            monkeypatch.setattr(topology, "_read_boot_id", boom)
        errors = _gather_errors(lambda: assert_moonep_ep_group_nvlink_coverage(group))
        assert errors[0] is not None and "cannot read boot_id: OSError('no /proc')" in errors[0]
        assert all(e == errors[0] for e in errors)
        assert os.path.exists("/proc/sys/kernel/random/boot_id")
