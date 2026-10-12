# MOONEP.md: the MoonEP expert-parallel backend in Megatron-LM

**Scope.** This document describes the MoonEP flex-dispatcher backend as it is implemented in the uncommitted working tree of `/pfs/code/Megatron-LM` on top of `dba89efc5`. It covers the MoonEP contract the backend relies on (§1), the Megatron internals it touches (§2), the design decisions (§3), requirement traceability (§4), an implementation map (§5), the configuration envelope (§6), unsupported features (§7), the tests (§8), risks and open questions (§9) and the validation record (Appendix A). It is written for reviewers and maintainers of the integration and for the MoonEP author. The earlier design (fp32 master weights outside autograd, per-layer VMM pools, `MoonEPOptimizer`, `zero_moe_grads()`, a grad-zeroing state machine) is gone; its plan stays in git history (`git show dba89efc5:MOONEP.md`).

**Status (2026-10-06).**
- The redesign is implemented and uncommitted. Expert weights are ordinary bf16 parameters, one process-global VMM pool set serves every MoE layer, and data parallelism is limited to ZeRO-1 and ZeRO-2.
- It needs `moonep.ExpertPools`, which exists only uncommitted in MoonEP-Fork. `pyproject.toml` still pins `30c2f8c`, which lacks it, and `uv.lock` still resolves `d9f044b` (§9 items 3 and 4).
- Native validation on `slinky-1` is in progress. Appendix A.3 lists the runs; their results are `TBD`. This design has not run on B200 or across nodes.

**Baseline.**

| Item | Value |
|---|---|
| Megatron-LM | `/pfs/code/Megatron-LM` @ `dba89efc5` ("fix recompute merge") plus the uncommitted redesign (`git status`). The pre-MoonEP baseline is `1912044d7`; `git diff 1912044d7 -- <file>` shows the whole MoonEP change of a file. `megatron/core/optimizer/__init__.py`, `megatron/core/transformer/module.py` and `test_optimizer_kernels.py` are back to `1912044d7`; `megatron/core/optimizer/moonep_optimizer.py` is deleted. |
| MoonEP | `/pfs/code/MoonEP-Fork` @ `30c2f8c` = upstream `33327eb` ("Public Release 26/09") + `753befe` (runtime varlen, `1 <= s <= S`) + `30c2f8c` (reference model on the new API), plus the uncommitted `moonep.ExpertPools`: `moonep/expert_pools.py`, its export in `moonep/__init__.py`, the README section "Sharing expert pools across layers" and `tests/test_expert_pools.py`. Package `moonep 0.0.1`. |
| Python env | `/pfs/code/megatron-minimax-experiments/.venv`: Python 3.12.3, torch 2.14.1, TransformerEngine 2.19.0, nvidia-cutlass-dsl 4.8.0, cuda-bindings 13.4.3, megatron-core 0.20.0. Not installed: pytest, isort, black. |
| GPU node | `slinky-1`: 8x NVIDIA H200 (sm_90) on NVSwitch with working multicast, driver 580.126.09, no host compiler. `moonep._C` there is a host-only C++ build of `MoonEP/csrc/bindings.cu` made on the login node (§9 item 2). The unit tests ran first under a pytest stand-in, then under pytest 9.1.1 (Appendix A.3 (2)). |
| Historical | A B200 VM with MoonEP `d9f044b` (A.1) and `slinky-0` with an emulated `moonep._C` and multicast (A.2). Both ran the earlier design. |

**Citation conventions.**
- `path:line` cites the working trees above. A Megatron path is the shortest suffix that is unique in the repository: `moonep_backend/manager.py` is `megatron/core/transformer/moe/moonep_backend/manager.py`, `moe/experts.py` holds SequentialMLP and TEGroupedMLP, `training/arguments.py` is `megatron/training/arguments.py`. Test files have unique basenames; they live in `tests/unit_tests/transformer/moe/`, except `test_moonep_kernels.py`, `test_moe_kernels.py` and `kernels/manifest.py` (`tests/unit_tests/determinism/kernels/`). Files the MoonEP change does not touch have the same lines at `1912044d7`.
- `MoonEP/` is `/pfs/code/MoonEP-Fork/`. Apart from `moonep/expert_pools.py`, `moonep/__init__.py` and `README.md`, its files are as committed at `30c2f8c`.
- A bare basename (for example `api.py` after `MoonEP/moonep/api.py`, or `storage.py` after `moonep_backend/storage.py`) repeats the directory of the last full citation of that file above it.
- **[unverified]** marks an inference; the way to check it follows.

---

## 0. TL;DR

- **Where MoonEP plugs in.** It is the fourth backend of the flex token dispatcher: `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend moonep` (`token_dispatcher.py:1863-1878`). `_MoonEPManager` converts the routing metadata, dispatches, stages the expert weights and combines (`moonep_backend/manager.py:89-357`). `MoELayer` builds `MoonEPGroupedMLP` experts instead of the spec's and binds them to the manager (`moe_layer.py:373-388`).
- **Expert weights are ordinary parameters.** Per local expert `i` there are `experts.linear_fc1.weight{i}` `[2*Hi, H]` (gate rows, then up rows) and `experts.linear_fc2.weight{i}` `[H, Hi]`, in `params_dtype` (bf16 under `--bf16`), marked `allreduce=False` (`moonep_backend/experts.py:66-89, 123-128`). These are TEGroupedMLP's names. DDP, the distributed optimizer and Megatron-FSDP v1 keep their main weights, main grads and optimizer state as for any expert linear weight, and the optimizer chain is the standard one.
- **One process-global pool set.** One `moonep.Buffer` and one `MoonEPExpertPools` (wrapping `moonep.ExpertPools`) per process serve every MoE layer; all layers must agree on the Buffer's `(S, H, K, E)` and the pools' `(H, Hi)` (`moonep_backend/buffer.py:22-96`). Per projection and rank, the pools hold a bf16 `[2*epn, out, in]` chunk (staging rows, then prefetch slots) and an fp32 chunk of the same shape (local grads, then reduce slots) (`MoonEP/moonep/expert_pools.py:21-37`).
- **Staging.** Dispatch copies the layer's weights into the staging rows (`torch._foreach_copy_`) and pushes the copies peers need (`prefetch_weight`), unless the pools already hold exactly this (layer, plan) (`moonep_backend/storage.py:130-163`). Combine's backward re-stages the layer before its GEMM backward (`moonep_backend/autograd.py:134-143`). Per micro-batch with N MoE layers this makes 2N-1 pushes, also under block recompute (`test_moonep_layer.py:323`; `test_moonep_recompute.py:455-463`).
- **Gradients.** The GEMM backward writes the layer's dW into the fp32 pools (`moonep_backend/autograd.py:167-177`). Dispatch's backward runs `reduce_grad` and hands the fp32 sums to the parameters TE-style (`moonep_backend/storage.py:174-216`): under a wrapper it adds them into `main_grad` (copies under `overwrite_main_grad`, ZeRO-2), sets `grad_added_to_main_grad=True` and returns placeholder grads; without a wrapper autograd gets the real grads in the parameter dtype. The weights are autograd inputs of `MoonEPDispatch`, so the wrapper's grad hooks fire after the reduce (`moonep_backend/autograd.py:43-47`).
- **Data parallelism: ZeRO-1 and ZeRO-2 only.** Plain DDP, DDP with the distributed optimizer, and Megatron-FSDP v1 `optim` or `optim_grads`. Anything that shards parameters is rejected (Megatron-FSDP `optim_grads_params` and v2, torch FSDP2, GTP weight rematerialization), and so are hybrid sharding and fine-grained parameter gathering (`training/arguments.py:1708-1741`; `moonep_backend/grads.py:42-102`).
- **Topology.** Every EP group must be one NVLink/NVSwitch domain with multicast. `initialize_megatron` checks this collectively before seeding and model construction; if any EP group fails, every rank raises the same error (`training/initialize.py:126-140`; `moonep_backend/topology.py:412-446`).
- **Limits** (§6): PP=1 without VPP, ETP=1, bf16, SwiGLU without bias, dropless top-K routing, fp32 router, no EP-communication overlap, MoE activation recompute only as full block recompute (optionally with CPU offloading), no CUDA graphs, no inference engines, no in-process restart, GPU initialization.
- **Checkpoints.** `torch_dist` keys, shapes and fragments equal SequentialMLP's and TEGroupedMLP's: fc1 goes through the SwiGLU factory, experts sit on a global expert axis, and per-expert `_extra_state` placeholders let SequentialMLP and TEGroupedMLP load MoonEP checkpoints strictly (`moonep_backend/experts.py:216-274`). Checkpoints therefore move both ways and reshard over EP. Megatron-FSDP saves `fsdp_dtensor` with TEGroupedMLP's per-expert names. Checkpoints written by the earlier MoonEP design need `--no-load-optim`.

### Architecture (one EP rank `r`; it owns the global experts `[r*epn, (r+1)*epn)`, `epn = E/R`)

`W_n` is `pools.weights(n)` and `G_n` is `pools.grads(n)` for the projection `n` in gate/up/down; `T` is this rank's token count per call, MoonEP's `s`.

```
FORWARD, MoE layer L
 x [seq,mbs,H] bf16 -> TopKRouter: probs [T,E] fp32, routing_map [T,E] (exactly K per token)
 dispatch_preprocess (@jit_fuser) -> setup_metadata:
     route_w [T,K] fp32 (masked top-k), idx [T,K] int32, tpe [E] int32
 token_dispatch -> _MoonEPManager.dispatch
   guards: inference, capacity, dtype, phase 1 (no gradient-tracking forward of L awaits backward)
   ensure_params_ready(fc1, fc2)                  DDP overlap_param_gather; skipped in a backward
   MoonEPDispatch.apply(x, route_w, idx, tpe, ..., *fc1, *fc2)   weights: autograd inputs only
     -> Buffer.dispatch -> h [nvs_s,H] bf16, w [nvs_s] fp32, cu_seqlens [2*epn], plan
   pools.stage(L, plan, fc1, fc2)                  skipped if the pools hold (L, plan)
     staging(n) <- foreach_copy(fc1[i][:Hi], fc1[i][Hi:], fc2[i])     bf16, local
     Buffer.prefetch_weight(plan): every rank pushes its staging rows over NVLink into
                                   the slots the plan assigns; ends in a cross-rank barrier
 MoonEPGroupedMLP.forward (reads only the pools)
   gate = gmm(h, W_gate^T); up = gmm(h, W_up^T)
   a = ((silu(gate) * up).float() * nan_to_num(w)).bf16(); y = gmm(a, W_down^T)
 token_combine -> MoonEPCombine.apply -> Buffer.combine(plan) -> out [T,H] bf16

BACKWARD, layers in reverse order, same stream
 MoonEPCombine.backward:  Buffer.dispatch(grad_out, plan=plan); pools.begin_grads(L);
                          pools.stage(L, plan, ...)            re-stage unless the pools hold (L, plan)
 GroupedMMLiveGrad.backward (x3): dX = gmm(g, W_n); dW = gmm(g^T, x) bf16 [2*epn] -copy_-> G_n fp32
 MoonEPDispatch.backward: Buffer.combine(plan, grad_h, grad_w) -> dx, d route_w
   pools.reduce(plan): launch_inter_rank_sync; Buffer.reduce_grad: G_n[:epn] += peers' slot dW
   hand-off: fc1[i].main_grad[:Hi] += G_gate[i]; [Hi:] += G_up[i]; fc2[i].main_grad += G_down[i]
             (copy_ under overwrite_main_grad); grad_added_to_main_grad = True; return placeholders
   -> AccumulateGrad of every expert weight -> DDP / Megatron-FSDP grad hook

ITERATION END
 finalize_model_grads: finish_grad_sync (wrappers scale, reduce over expert-DP, check for NaN)
                       -> finalize_moonep_step (recompute bookkeeping, pool occupant reset)
 optimizer.step(): the standard ChainedOptimizer [dense, expert]; the next forward stages the update
```

| Tensor | Shape, dtype | Memory and owner | Written by | Read by |
|---|---|---|---|---|
| `linear_fc1.weight{i}`, `linear_fc2.weight{i}` | `[2*Hi,H]`, `[H,Hi]`, `params_dtype` | ordinary CUDA memory; DDP's parameter buffer or Megatron-FSDP's | optimizer step, checkpoint load | `stage` (copy-in) |
| their `main_grad` | fp32 by default | the wrapper's grad buffer (ZeRO-2: a temporary bucket per backward) | dispatch backward (hand-off) | wrapper reduction, optimizer |
| `W_n = pools.weights(n)` | `[2*epn,out,in]` bf16 | VMM, process-global | rows `[0,epn)`: `stage`; rows `[epn,2*epn)`: peers' pushes | GEMMs; this rank's push (rows `[0,epn)`) |
| `ExpertPools.prefetch_buffer(n)` | `[R,epn,out,in]` bf16 | VMM mapping of every rank's slot rows | every rank's push | nobody directly; the GEMMs read the slots through `W_n` |
| `G_n = pools.grads(n)` | `[2*epn,out,in]` fp32 | VMM, process-global | GEMM backward (`copy_`); `reduce_grad` adds into rows `[0,epn)` and zeroes consumed slots | hand-off (rows `[0,epn)`); peers' `reduce_grad` (slot rows) |
| `ExpertPools.reduce_buffer(n)` | `[R,epn,out,in]` fp32 | VMM mapping of every rank's reduce slots | each rank's GEMM backward (its own chunk) | the owners' `reduce_grad` |

**Memory.** With `P = epn*H*Hi`, the pools cost `36*P` bytes per rank once per process: three projections of `2*epn` rows at 2 bytes (bf16) plus 4 bytes (fp32) per element. Each chunk is rounded up to the VMM granularity (`MoonEP/moonep/expert_pools.py:199-230`). One layer's bf16 expert weights are `6*P` bytes, so the pools cost about six layers' worth of bf16 expert weights, whatever the depth. The earlier design paid the same `36*P` per layer. For example, E=128 at EP=8 (`epn=16`) with H=7168 and Hi=2048 gives `P = 2.35e8` and pools of 8.5 GB per rank. The Buffer adds `NvS*H*2` bytes of symmetric memory (`MoonEP/moonep/api.py:273-289, 362`). The placeholder grads are fresh full-size tensors (`moonep_backend/storage.py:79-87`) that the wrapper hooks drop at once (`distributed_data_parallel.py:607`; `megatron_fsdp/megatron_fsdp.py:732-733`), so the transient peak is one layer's bf16 expert weights.

---

## 1. MoonEP contract primer

### 1.1 `moonep.Buffer`
- Constructor: `Buffer(S, H, K, E, num_ep_ranks, num_sms=None, token_padding=128, group=None, comm_stream_priority=-1, enable_pdl=True, explicitly_destroy=False)` (`MoonEP/moonep/api.py:453-466`). `num_sms=None` becomes 32 (`api.py:252-253`). `num_ep_ranks` must equal the group size, `E % R == 0`, and `epn = E // R` (`api.py:257-269`).
- `NvS = align_up(S*K + (token_padding-1)*2*epn, token_padding)` (`api.py:273-289`). `S` is a capacity: every call needs `1 <= s <= S` (`api.py:841-846`).
- Construction is collective over `group`. It allocates a symmetric bf16 `hidden_buf [NvS_padded, H]` and an int32 meta buffer with a multicast view, zeroes the barrier slots and calls `dist.barrier` (`api.py:362-371`).
- `destroy()` is collective (stream sync, `cuda.synchronize`, barrier) (`api.py:545-587`). With `explicitly_destroy=True`, `__del__` only warns (`api.py:589-597`).

### 1.2 Calls

| Call | Contract |
|---|---|
| `dispatch(x, w, topk, tpe)` (fresh) | `x [s,H]` bf16, `w [s,K]` fp32, `topk [s,K]` int32, `tpe [E]` int32 (`api.py:853-894`). Plans, scatters (zero-filling the segment padding) and returns fresh `h [plan.nvs_s,H]`, `w [plan.nvs_s]`, `cu_seqlens [2*epn]` and `plan` (`api.py:812-831, 927-964`). `cu_seqlens` holds padded segment END offsets, the `torch._grouped_mm` `offs` format; groups `[0,epn)` are this rank's experts, groups `[epn,2*epn)` its prefetch slots (`MoonEP/moonep/planning.py:993-1000`). Without `total_num_tokens`, every EP rank must pass the same `s`; the planner traps otherwise (`MoonEP/README.md:113`). |
| `dispatch(g, plan=plan)` | Plan reuse: no planning, and `cu_seqlens` comes back `None`; asserts `plan.num_tokens == s` (`api.py:908-925`). Megatron uses it in combine's backward and in a recompute's replay. |
| `prefetch_weight(plan, local_*_weight, *_prefetch_buffer)` | Collective push. Each rank writes its own experts `[epn,d1,d2]` into every slot of the all-rank pools `[R,epn,d1,d2]` that `plan.experts_to_copy` assigns to one of them. One fused gate/up/down kernel ends in a cross-rank barrier and has no entry barrier (`api.py:1002-1133`; `MoonEP/moonep/prefetch.py:462-470`). Every EP rank must call it at the same point of its call sequence. |
| `combine(plan, h, w=None)` | Pre-reduces duplicate rows in fp32, accumulates each token's rows in fp32 and returns a fresh `[s,H]` bf16 tensor (`api.py:685-722, 1194-1204, 1230-1235`). It does not apply routing weights. |
| `reduce_grad(plan, local_*_grad, *_reduce_buffer)` | Each owner reads, over NVLink, the slots that hold its experts from every rank's reduce chunk and adds them into its fp32 local grads; after a cross-rank barrier each rank zeroes its consumed slots (`api.py:1293-1351`; `MoonEP/moonep/grad_reduce.py:296-389`). There is no entry barrier, so `launch_inter_rank_sync` must run first (`grad_reduce.py:468-471`). |

### 1.3 Ordering rules
- Barrier sites: planning (`planning.py:661, 1082, 1207`), dispatch exit (`MoonEP/moonep/dispatch.py:694`), combine entry (`MoonEP/moonep/combine.py:244`), inter-rank sync (`MoonEP/moonep/inter_rank_sync.py:73`), the end of every `prefetch_weight` (`prefetch.py:467`) and one per `grad_reduce` launch (`grad_reduce.py:369`). All of them use the same three per-rank barrier slots, so barriers match by sequence only (`api.py:329-333`).
- **Every EP rank must issue the same sequence of barrier-bearing calls.** A mismatch spins for about 100 s and then traps the device (`MoonEP/moonep/_common.py:36-37, 337`).
- Calls on one Buffer must be serialized. Megatron always runs them synchronously on the current stream (`moonep_backend/manager.py:205-206, 350`; `MoonEP/moonep/expert_pools.py:76-79`).
- **First-use host alignment.** MoonEP compiles each kernel on first use, so a slow host could leave its peers spinning into the timeout. `align_ep_group_on_first_use` runs `torch.cuda.synchronize(); dist.barrier(ep_group)` the first time a call site runs with a given key (`moonep_backend/autograd.py:19-30`): fresh dispatch per token count and grad mode (`moonep_backend/manager.py:263-264`), replay per token count (`manager.py:295`), combine (`manager.py:351`), both backwards (`moonep_backend/autograd.py:91, 137`), push (`moonep_backend/storage.py:160`) and reduce (`storage.py:185`).
- **Slot reuse with shared pools.** A push has no entry barrier, so a peer may write this rank's slots as soon as it passes its own previous barrier-bearing call. A push is therefore safe only if every rank issues a barrier-bearing call between its last read of the slots and the push. Megatron satisfies this on one stream: in the forward, the previous layer's combine and the current layer's dispatch sit between the previous GEMMs and the push; in the backward, the later layer's dispatch backward and the current layer's combine-backward dispatch do (`moonep_backend/storage.py:156-159`; `MoonEP/README.md:98`). Peers read this rank's reduce slots only inside `reduce_grad`, after the inter-rank sync (`MoonEP/moonep/expert_pools.py:313-327`), and the next local write to them, the next layer's GEMM backward, follows `reduce_grad`'s barrier in stream order.

### 1.4 Plans, Buffer and pool sharing
- A fresh dispatch allocates new plan tensors, plan-reuse paths build nothing, and outputs are fresh copies with `zero_copy=False` (`api.py:904-906, 656-668, 933-948`). So one Buffer serves every layer and micro-batch, as long as each (layer, micro-batch) keeps its own plan and all layers share `(S, H, K, E, token_padding, num_sms)` (`moonep_backend/buffer.py:3-9`).
- Never use `zero_copy=True` in training: the next call overwrites the view (`api.py:789-796`).
- JIT compile caches are keyed by the Buffer capacity and shapes, not by the per-call `s` (`dispatch.py:711-727`; `grad_reduce.py:397-407`); the prefetch key also includes the pools' rank stride (`prefetch.py:478-489`). Since one pool set serves all layers, each kernel compiles once per process.

### 1.5 `moonep.ExpertPools` (uncommitted MoonEP API)
- `ExpertPools(buffer, group, shapes)` is collective. The group must be the Buffer's, `shapes` maps gate/up/down to a per-expert `(out, in)` whose dims are multiples of 128, and every rank must agree on `(epn, VMM granularity, shapes)` (`MoonEP/moonep/expert_pools.py:84-197`).
- Layout: per projection and dtype, each rank allocates one VMM chunk of `[2*epn, out, in]` elements, rounded up to the VMM granularity, and all `R` chunks are mapped back to back (`expert_pools.py:21-37, 199-230`). Accessors: `weights(n)` and `grads(n)` `[2*epn,out,in]`; `staging(n)` and `local_grads(n)`, their first `epn` rows; `prefetch_buffer(n)` and `reduce_buffer(n)`, the `[R,epn,out,in]` views of every rank's slot rows (`expert_pools.py:250-278`).
- Construction zeroes this rank's chunks, because VMM memory is not zero-filled and `reduce_grad` clears only consumed slots, then synchronizes and barriers (`expert_pools.py:151-155, 223`).
- `push(plan)` is `prefetch_weight` from the staging rows (`expert_pools.py:282-302`). `reduce(plan)` is `launch_inter_rank_sync` plus `reduce_grad` into `local_grads` (`expert_pools.py:304-327`). `destroy()` synchronizes, barriers and drops the mappings; it is idempotent (`expert_pools.py:329-347`).
- All views of one projection and dtype share one autograd version counter, so they must stay out of `save_for_backward` (`expert_pools.py:41-42`).
- The usage contract is `MoonEP/README.md:70-98`. Megatron departs from it in one point: the dW GEMM overwrites `grads(n)` instead of adding into it, so neither `local_grads(n)` nor the slots need zeroing between layers (ADR-7).

### 1.6 Hardware, limits and failure modes
- Multicast is mandatory: `Buffer()` asserts it (`MoonEP/moonep/buffer.py:299`), and a failing driver call inside `moonep._C` calls `exit()` (`MoonEP/csrc/nvl_shared_buffer.cuh:17-27`).
- Memory handles: MoonEP uses fabric handles if every rank supports them. Otherwise it passes POSIX fds over AF_UNIX sockets in a tempdir that group rank 0 creates, which needs one host and a shared `/tmp` (`MoonEP/moonep/buffer.py:31-57, 130-144`). `MOONEP_MEM_HANDLE_TYPE` must be identical on all ranks, because `fd` skips a collective that the other modes issue (`buffer.py:36-47`).
- MoonEP has no NVLink or P2P check. A bad topology shows up as `exit()`, a hang or a device trap, which is why Megatron checks first (ADR-11).
- Limits: K <= 32 and R <= 128 (`planning.py:1519-1528`); H a multiple of 128 for combine (`combine.py:63, 612-614`); both inner dims multiples of 128 for prefetch and reduce (`prefetch.py:565-568`; `grad_reduce.py:501-502`; `expert_pools.py:175-180`).
- The reference model provides the dense-reference check used in A.3 (`check_against_dense_reference`, `MoonEP/my_moon_ep.py:431-495`). It all-reduces the router grad over EP once per micro-batch (`my_moon_ep.py:401, 411`). Megatron must not port that: DDP already averages the router grad (§2.4).

---

## 2. Megatron internals primer

### 2.1 MoE layer
- `BaseMoELayer` takes `pg_collection.ep`, sets `num_local_experts = E // EP` and owns the global experts `ep_rank*epn + i` (`moe_layer.py:193-210`), the same ownership as MoonEP's.
- `MoELayer.__init__` builds the router (`moe_layer.py:276-283`), the dispatcher (`moe_layer.py:345-370`), the experts (`moe_layer.py:372-394`) and the shared experts (`moe_layer.py:396-410`), in that order. The dispatcher, and with it `_MoonEPManager`, exists before the experts.
- The layer number arrives after construction: `TransformerLayer` builds the MLP and then calls `set_layer_number` (`transformer_layer.py:465-473`; 1-based, `transformer_block.py:417`). `BaseMoELayer.set_layer_number` forwards it to the router and to MoonEP experts (`moe_layer.py:222-227`).
- Forward stages: shared experts, `route`, `preprocess`, `dispatch`, `routed_experts_compute`, `combine`, `postprocess` (`moe_layer.py:728-769`). `routed_experts_compute` calls `dispatch_postprocess`, then the experts with `(dispatched_input, tokens_per_expert, permuted_probs)`, which must return `(out, None)` (`moe_layer.py:604-639`; interface at `moe_layer.py:70-85`).
- TP > 1 in training requires sequence parallelism (`moe_layer.py:709-713`).

### 2.2 Flex dispatcher
- `_DispatchManager` is the backend ABC: `setup_metadata`, `dispatch`, `combine`, `get_permuted_hidden_states_by_experts`, `get_restored_hidden_states_by_experts` (`token_dispatcher.py:963-998`).
- `MoEFlexTokenDispatcher` selects the backend (`token_dispatcher.py:1835-1883`). The MoonEP branch asserts ETP = 1 and EP > 1 and builds `_MoonEPManager` (`token_dispatcher.py:1863-1878`).
- `dispatch_preprocess` is `@jit_fuser`, i.e. `torch.compile` (`megatron/core/jit.py:16-33`; `token_dispatcher.py:1937-1962`), so `setup_metadata` must not call MoonEP. It reshapes the routing map and probs to `[T, tp*ep, epn]` (`token_dispatcher.py:1909-1935`).
- `token_dispatch` calls `manager.dispatch` and returns `manager.dispatched_probs` (`token_dispatcher.py:1964-1997`). `dispatch_postprocess` passes `get_number_of_tokens_per_expert()` to the experts (`token_dispatcher.py:1999-2016`). `token_combine` calls `manager.combine` (`token_dispatcher.py:2027-2049`).
- With `moe_shared_expert_overlap`, the flex dispatcher runs the shared experts' fc1 and fc2 on a side stream around `dispatch` and `combine`, for every backend (`token_dispatcher.py:1988-1995, 2064-2067`). `_validate_moonep` forbids it (§6).

### 2.3 Expert weights and checkpoint layout
- `SequentialMLP` builds one `MLP` per local expert (`moe/experts.py:1642-1684`). Per expert, fc1 is `[2*Hi, H]` with the gate rows first and fc2 is `[H, Hi]` (`mlp.py:340-347`).
- Routing weights apply after the activation and before fc2, in fp32 (`mlp.py:368-371`; TEGroupedMLP at `moe/experts.py:1125-1127`).
- `torch_dist`: `SequentialMLP.sharded_state_dict` maps local expert `i` to global expert `ep_rank*epn + i` under `{prefix}experts.linear_fc1.weight` and `{prefix}experts.linear_fc2.weight`, where `{prefix}` is the experts module's own prefix, so the keys read `...mlp.experts.experts.linear_fc1.weight`. fc1 goes through the SwiGLU factory, which stores gate and up rows as two fragments, and `replica_id[2]` is the expert-DP rank (`moe/experts.py:1759-1800`; `mlp.py:460-530`). TEGroupedMLP's keys are interchangeable with these (`moe/experts.py:1160-1202`). `TransformerBlock` adds the layer axis for homogeneous layers (`transformer_block.py:866-890`).
- `fsdp_dtensor`: `preprocess_fsdp_dtensor_state_dict` splits SwiGLU fc1 into `_w`/`_v` entries and renames local to global expert ids (`training/checkpointing.py:1834-1859`). It recognizes experts only by per-expert names, `mlp.experts.linear_fc{1,2}.weight{i}` (TEGroupedMLP) or `mlp.experts.local_experts.{i}.linear_fc{1,2}.weight` (SequentialMLP) (`fsdp_dtensor_checkpoint.py:105-127, 258-279`). A key without an expert index is kept unchanged on every EP rank (`fsdp_dtensor_checkpoint.py:143-150`).

### 2.4 Router
- `TopKRouter.weight` is a `params_dtype` parameter, and the logits are computed in `moe_router_dtype` (`moe/router.py:74-126`).
- The routing map gets exactly K experts per token (`moe_utils.py:943-959`) unless capacity dropping is on (`moe/router.py:966-975`). `padding_mask` changes neither `probs` nor `routing_map` (`moe/router.py:926-1039`).
- The router weight has no `allreduce` attribute, so DDP averages its grad over the dense data-parallel group like any dense weight (`distributed/param_and_grad_buffer.py:966-980`).

### 2.5 Config, CLI and init order
- A `Literal` config field becomes argparse choices (`argument_utils.py:151-155`; factory at `training/arguments.py:2584`). `core_transformer_config_from_args` copies args by field name (`argument_utils.py:318-321`), and `--swiglu` sets `F.silu` plus GLU (`argument_utils.py:334-337`).
- The config is built before `init_process_group` (`pretrain_gpt.py:575-579` versus `training/initialize.py:393`), again per micro-batch in `get_batch` (`pretrain_gpt.py:103`), and in unit tests without process groups. So `_validate_moonep` is pure: no GPU, no collectives, no `moonep` import (`transformer_config.py:1617-1623`).
- `finish_mpu_init` is the first point where args are final, the device is set and the EP groups exist (`training/initialize.py:116-124`). The MoonEP topology check runs right there (`training/initialize.py:126-140`), before seeding (`training/initialize.py:154`).
- Model wrapping: `.cuda()`, then `Float16Module`, then DDP or Megatron-FSDP (`models/dist_utils.py:65-73, 186-216`).

### 2.6 Process groups and RNG
- The expert grid is `RankGenerator(tp=ETP, ep=EP, dp=EDP, pp=PP, cp=1)`; with ETP = 1 each EP group is a contiguous rank block (`parallel_state.py:869-895, 1353-1365`). EP peers always share a PP stage (`parallel_state.py:903-906`).
- The expert RNG seed is `seed + 1024 + 100*ep_rank + etp_rank`, the same across expert-DP (`tensor_parallel/random.py:509`).
- `ProcessGroupCollection` carries `ep`, `expt_tp`, `expt_dp` and the other groups (`process_groups_config.py:104-160`). `MoonEPGroupedMLP` uses `pg_collection.ep` and `pg_collection.expt_dp` (`moonep_backend/experts.py:115-117`); nothing in the backend reads global groups from `parallel_state`.

### 2.7 Float16Module
- `Float16Module` casts every floating-point parameter and buffer that is not marked `keep_in_fp32` through `module._apply` (`transformer/module.py:489-506, 539-542`). MoonEP's expert weights carry no marker, so they are cast like any linear weight (`test_moonep_layer.py::test_float16_module_casts_weights_like_any_linear`). Nothing MoonEP-specific remains here.

### 2.8 DDP and the distributed optimizer
- DDP registers every parameter with `requires_grad` and sets `grad_added_to_main_grad = False` on it (`distributed_data_parallel.py:181-195`). Buffers are keyed by `(param dtype, grad dtype, is_expert = not allreduce, ...)`, so `allreduce=False` puts MoonEP's weights into the expert-parallel buffers, which reduce over expert-DP (`distributed/param_and_grad_buffer.py:966-980`).
- The grad hook runs on each parameter's AccumulateGrad node (`distributed_data_parallel.py:465-502`). It adds `.grad` into `main_grad` unless `grad_added_to_main_grad` is set (or `zero_out_wgrad`), then sets `param.grad = None`. With `overlap_grad_reduce` it first asserts `param.grad is not None` and then marks the bucket ready (`distributed_data_parallel.py:581-612`). `zero_grad_buffer()` resets `grad_added_to_main_grad` once per iteration (`distributed_data_parallel.py:718-732`).
- With the distributed optimizer, `param.data` is re-pointed into DDP's buffer (`distributed/param_and_grad_buffer.py:1367-1371`) and `main_grad` is a view of the grad buffer (`distributed/param_and_grad_buffer.py:1429-1431`). So code must read parameters through module attributes, never through cached tensors.
- With `overlap_param_gather`, a module's forward pre-hook waits for the all-gather of its own parameters only (`recurse=False`, `distributed_data_parallel.py:545-567`). A module that never runs forward never waits. `ensure_params_ready(params)` publishes such parameters through the per-bucket callback that DDP stores on them (`core/utils.py:3307-3324`; `distributed_data_parallel.py:29-60, 430-447`). DDP is the only writer of that callback in `megatron/`; Megatron-FSDP stores none.
- DDP scales the expert grads, reduces them over expert-DP and checks every bucket for NaN, Inf and unexpectedly large values (`distributed_data_parallel.py:249-296`; `distributed/param_and_grad_buffer.py:389-425`).

### 2.9 Megatron-FSDP v1
- Strategies: `optim` (ZeRO-1) reduce-scatters on the final micro-batch, `optim_grads` (ZeRO-2) on every micro-batch, and `optim_grads_params` (ZeRO-3) also shards parameters (`mcore_fsdp_adapter.py:229-239`). ZeRO-2 turns `overlap_grad_reduce` on (`megatron_fsdp/megatron_fsdp.py:346-352`).
- FSDP units default to `TransformerLayer`, `MoETransformerLayer` and `MambaLayer` (`mcore_fsdp_adapter.py:236-239`). A unit's pre-forward and pre-backward hooks (`megatron_fsdp.py:847, 860, 973, 985`) all-gather its parameters and set `grad_added_to_main_grad = False`, `__fsdp_param__ = True` and `overwrite_main_grad`, which is true when the parameter's grads are sharded (`megatron_fsdp.py:508-563, 389-404`).
- The grad hook: on a sharded grad buffer it takes a temporary main-grad bucket from `get_main_grad()` and copies `.grad` into it, unless `grad_added_to_main_grad` says the bucket already holds the wgrad; it then deletes `.grad` and resets the flag (`megatron_fsdp.py:691-736`).
- Between `finish_grad_sync` and the next forward, module attributes hold the optimizer's DTensor parameters. The wrapper's `forward` swaps the raw parameters back in (`megatron_fsdp.py:1461-1463, 1468-1507, 1576-1584`).
- Expert parameters are recognized by name: `".experts." in name` (`megatron_fsdp/param_and_grad_buffer.py:1846`).
- Training-script rules: `--ckpt-format fsdp_dtensor` and `CUDA_DEVICE_MAX_CONNECTIONS != 1` (`training/arguments.py:1185-1189`). The default strategy is `optim_grads_params` (`training/arguments.py:3446`), and main grads default to the model dtype (`--megatron-fsdp-main-grads-dtype auto`, `training/arguments.py:3941-3944`). Megatron-FSDP v2 requires `optim_grads_params` (`mcore_fsdp_adapter.py:793-796`).
- Megatron-FSDP v1 loads GLU models with `strict=False` (`mcore_fsdp_adapter.py:315-335`).

### 2.10 `finalize_model_grads`
- It runs once per iteration after the last backward, and not for forward-only schedules (`pipeline_parallel/schedules.py:864-872`).
- Order: every chunk's `finish_grad_sync` (the wrappers' reductions), then `finalize_moonep_step(model)` when MoonEP is on (`finalize_model_grads.py:684-692`), then the remaining all-reduces and router updates, and last the per-token scaling (`finalize_model_grads.py:746-765`).

### 2.11 Optimizer
- `get_megatron_optimizer` is unchanged (identical to `1912044d7`). `_get_param_groups` skips parameters without `requires_grad` and keys experts by `allreduce` (`core/optimizer/__init__.py:339, 357`), and the result is `ChainedOptimizer([dense, expert])` (`core/optimizer/__init__.py:1312`). MoonEP's weights land in the same sub-optimizer and parameter group as SequentialMLP's (`test_moonep_training.py::test_param_group_overrides_match_sequential`).

### 2.12 `train_step`, schedules and recompute
- `train_step` runs `zero_grad_buffer()`, `optimizer.zero_grad()`, `forward_backward_func` and `optimizer.step()` (`training/training.py:3330-3335, 3410, 3489`). Nothing MoonEP-specific remains in it.
- With PP = 1, forward and backward alternate per micro-batch, and every micro-batch but the last runs under `no_sync` (`pipeline_parallel/schedules.py:815-862`). Forward-only schedules do not disable autograd; their callers do (`evaluate` at `training/training.py:5693`; `--freeze-all-layers` at `training/training.py:2559`).
- Block recompute checkpoints each of the first `recompute_num_layers` layers separately, with `tensor_parallel.checkpoint` in bf16 (`core/recompute.py:330-332, 361-376`). CPU offloading on this path requires block recompute (`core/recompute.py:164-176`).
- `CheckpointFunction` sets the global `is_checkpointing()` flag during the first (no-grad) forward and during the recompute inside the backward, without `try/finally` (`tensor_parallel/random.py:571-585, 609, 631, 645, 672`).

### 2.13 Distributed checkpointing
- Optimizer state is matched to model `ShardedTensor`s by `id(ten.data)`, so each sharded entry must hold the parameter object itself (`dist_checkpointing/optimizer.py:45-80`).
- If a strict `load_state_dict` fails, `load_checkpoint` retries with `strict=False` and only prints `load_return` (`training/checkpointing.py:3065-3071`).
- The default format is `torch_dist` (`config/training_config.py:486`).

### 2.14 Shutdown and inference
- `destroy_model_parallel()` has one production caller, the in-process-restart fault path (`training/training.py:715-719`), so it must not run collectives.
- The end of `pretrain` and the exit branch of `train()` are reached by every rank (`training/training.py:2277, 5571`).
- Inference engines set a process-wide `InferenceMode` flag (`core/inference/utils.py:49-63`).

---

## 3. Design decisions

**ADR-1: Backend placement.**
- *Decision*: add `'moonep'` to `moe_flex_dispatcher_backend`, with a `_MoonEPManager` (`transformer_config.py:989`; `token_dispatcher.py:1863-1878`).
- *Alternatives*: a new `moe_token_dispatcher_type`, or a new MoE layer class. Both bypass the flex extension point, and the second duplicates the stage split that other code consumes.
- *Why*: it matches deepep, hybridep and ncclep, the CLI choice appears automatically (§2.5), and `MoELayer`'s stages are reused unchanged.

**ADR-2: `MoELayer` builds the experts.** The spec helpers never see the backend, so `MoELayer.__init__` builds `MoonEPGroupedMLP` when `uses_moonep(config)` is true. The dispatcher already exists by then, so `bind_experts` connects the two and hands the experts the shared pools (`moe_layer.py:373-388`; `moonep_backend/manager.py:179-182`).

**ADR-3: Expert weights are ordinary per-expert parameters named like TEGroupedMLP's.**
- *Decision*: two parameter containers, `linear_fc1` and `linear_fc2`, hold `weight{i}` per local expert: `[2*Hi, H]` with gate rows first, and `[H, Hi]`, in `params_dtype`, `allreduce=False` (`moonep_backend/experts.py:66-89, 123-126`). The containers never run forward. The weights are initialized like SequentialMLP's, per expert fc1 then fc2 under the expert-parallel RNG, so they match bit for bit (`moonep_backend/experts.py:134-141`; `test_moonep_layer.py::test_initial_weights_match_sequential`).
- *Rejected: fp32 masters outside autograd (the earlier design).* It needed its own optimizer, a grad-zeroing protocol, alias repair, and bans on DDP overlap and on every FSDP.
- *Rejected: one 3D `[epn, out, in]` parameter per projection.* The `fsdp_dtensor` path recognizes experts and SwiGLU fc1 only by per-expert names (`fsdp_dtensor_checkpoint.py:105-127, 258-279`). A stacked weight would keep one key on every EP rank (`fsdp_dtensor_checkpoint.py:143-150`), so EP ranks would collide, and its SwiGLU halves would not be split. TE-style tooling expects per-expert names too [unverified beyond `fsdp_dtensor`: check `parameter_names.py` and `upcycling_utils.py` for the TEGroupedMLP patterns].
- *Rejected: separate gate, up and down parameters.* No SwiGLU rule matches them (`fsdp_dtensor_checkpoint.py:258-279`), and the `torch_dist` layout would differ from SequentialMLP's fc1.
- *Why*: DDP, the distributed optimizer and Megatron-FSDP handle the weights like any expert linear weight, and checkpoints stay SequentialMLP's and TEGroupedMLP's. The cost is one bf16 copy into the pools per stage.

**ADR-4: One process-global pool set.**
- *Decision*: one `MoonEPExpertPools` per EP group and `(H, Hi)`, created by the first MoE layer's manager and shared by all layers (`moonep_backend/buffer.py:73-96`; `moonep_backend/manager.py:163-166`).
- *Alternative*: per-layer pools, as in the earlier design and in `my_moon_ep.py`: `36*P` bytes per layer and rank instead of once.
- *Why*: memory independent of depth; MoonEP describes the prefetch pool as process-global (`MoonEP/README.md:56`).
- *Consequences*: each backward except the last forwarded layer's must re-stage (ADR-5); two layers' expert computations cannot overlap (`MoonEP/README.md:98`); all MoE layers must share `H` and `Hi` (`moonep_backend/buffer.py:83-90`).

**ADR-5: Stage in the dispatcher; re-stage in combine's backward; skip by occupant.**
- *Decision*: `stage()` runs right after `Buffer.dispatch` in `_MoonEPManager.dispatch` (`moonep_backend/manager.py:324-328`) and in `MoonEPCombine.backward` before the GEMM backward (`moonep_backend/autograd.py:138-142`). It skips the copy and the push when `pools.occupant` is the same `(layer state, plan)` objects (`moonep_backend/storage.py:143-148`).
- *Why*: the dispatcher already holds the plan and the weights. The occupant is host state that changes identically on every EP rank, so all ranks push or skip together and the collective stays matched. Identity comparison with strong references cannot alias a later plan. `finalize_moonep_step` clears the occupant every iteration (`moonep_backend/grads.py:121`).
- *Result*: N pushes in the forward and N-1 in the backward per micro-batch; under block recompute a checkpointed layer's push moves from its combine backward to its recompute, so the count stays 2N-1 (`test_moonep_recompute.py:455-463`).

**ADR-6: Gradient hand-off through `MoonEPDispatch`, TE-style.**
- *Decision*: the expert weights are autograd inputs of `MoonEPDispatch` and are never read in its forward (`moonep_backend/autograd.py:43-47, 81-84`). Its backward combines, runs `reduce_grad`, then writes the fp32 sums into `main_grad`: added under DDP and ZeRO-1, copied under `overwrite_main_grad` (ZeRO-2). It sets `grad_added_to_main_grad = True` and returns placeholders (`moonep_backend/storage.py:174-216`). Without a wrapper it returns the grads in the parameter dtype (`storage.py:191-194`).
- *Alternatives*: (a) return fp32 grads to autograd, which casts a leaf's grad to the leaf's dtype and loses the fp32 sum [unverified here; observed in a CPU probe while designing; check by returning an fp32 grad for a bf16 leaf and reading `.grad.dtype`]; (b) write `main_grad` from a hook or in `finalize_model_grads`, which is too late, because DDP's overlapped reduction starts when a weight's AccumulateGrad hook fires (`distributed_data_parallel.py:581-612`) and ZeRO-2 hands out a fresh bucket per backward (`megatron_fsdp/megatron_fsdp.py:691-736`).
- *Why*: both wrappers already honor this convention for Megatron's fused-wgrad linear layers and TE (`tensor_parallel/layers.py:803-808`; `distributed_data_parallel.py:603-607`; `megatron_fsdp.py:705-736`). The placeholders are fresh tensors, zeros under `zero_out_wgrad`, so the `param.grad is not None` assertion of `overlap_grad_reduce` holds (`moonep_backend/storage.py:79-87`; `distributed_data_parallel.py:595-602`).

**ADR-7: dW overwrites the shared fp32 pool.**
- *Decision*: the GEMM backward copies the bf16 dW of all `2*epn` groups into `G_n` instead of adding (`moonep_backend/autograd.py:173-176`).
- *Why*: the pools hold one layer's grads at a time, and accumulation over micro-batches happens in `main_grad`. Overwriting removes all zeroing between layers, unlike MoonEP's suggested add-then-zero (`MoonEP/README.md:97`).
- *Requirements*: `torch._grouped_mm` must write exact zeros for empty groups (A.1; `test_moonep_kernels.py:82-97`). `grads_owner` makes a second layer's backward raise if it starts before the first layer's reduce (`moonep_backend/storage.py:167-172, 187`).

**ADR-8: ZeRO-1 and ZeRO-2 only.**
- *Decision*: accept plain DDP, DDP with the distributed optimizer (ZeRO-1), and Megatron-FSDP v1 `optim` (ZeRO-1) or `optim_grads` (ZeRO-2). Reject everything that shards parameters, plus hybrid sharding, fine-grained parameter gathering, Megatron-FSDP `no_shard` (plain DDP covers it) and MoE layers outside every FSDP unit (§6).
- *Why ZeRO-3 is out*: MoonEP stages each layer's whole local experts from the persistent parameters, at dispatch and again in combine's backward. With sharded parameters, each stage would first need an all-gather of weights whose modules never run forward, in both passes, ordered against MoonEP's barrier kernels, and a release afterwards. None of the ZeRO-3 implementations offers that hook: Megatron-FSDP v2 requires `optim_grads_params` (`mcore_fsdp_adapter.py:793-796`), torch FSDP2 already rejects EP > 1 (`training/arguments.py:1049-1050`), and GTP weight rematerialization shards what it rematerializes (`transformer_config.py:1656-1661`). The requested scope was ZeRO-1 and ZeRO-2.
- *Why units are required under Megatron-FSDP*: the weight containers never run forward, so only an enclosing unit's hooks gather them and set `overwrite_main_grad` and `grad_added_to_main_grad` (§2.9; `moonep_backend/grads.py:45-51`). `ensure_params_ready` cannot help there: Megatron DDP is the only writer of the param-ready callback in `megatron/` (`distributed_data_parallel.py:430-447`).

**ADR-9: Routing weights before fc2.** The experts scale the activation in fp32 before fc2, like Megatron's MLP (`moonep_backend/experts.py:208-212`; `mlp.py:368-371`). MoonEP's reference scales after fc2; without a bias the two differ only in rounding. `nan_to_num` is needed because padding and tail slots carry stale weights.

**ADR-10: Recompute replays the forward's plan.**
- *Decision*: a checkpointed layer's first (no-grad) forward records `(plan, cu_seqlens, token ids, probs, T)`. The recompute in the backward dispatches with that plan, if its forward is the layer's only checkpointed forward still waiting; otherwise it plans afresh (`moonep_backend/manager.py:233-251, 279-323`).
- *Why*: no second planning pass, and the recompute's token layout equals the forward's. The recompute takes the recomputed probabilities at the recorded experts, so the router still gets its gradient. Tokens whose top-K flipped between the no-grad forward and the recompute keep the forward's weight; they are counted on the device and reported once per iteration (`moonep_backend/grads.py:18-39`; `moonep_backend/experts.py:161-175`). A recompute is detected only inside a backward, because Megatron sets and clears the checkpoint flag without `try/finally` (`moonep_backend/manager.py:73-75, 243`).
- *Shared pools*: the replay still stages, skipping the push only if the pools still hold its forward's copies (`moonep_backend/manager.py:324-327`).

**ADR-11: Topology check at initialization.** MoonEP has no topology check, and its failures are `exit()`, hangs or traps. `initialize_megatron` runs a collective check with WORLD consensus right after the process groups exist: NVML NVLink P2P status for same-host pairs, fabric clique and health for cross-host pairs, multicast, compute capability, and the memory-handle path MoonEP will pick (`training/initialize.py:126-140`; `moonep_backend/topology.py:1-20, 412-446`). The result is cached per EP rank tuple; `_MoonEPManager.__init__` re-runs it EP-locally for library users (`moonep_backend/manager.py:147-148`).

---

## 4. Requirements traceability

| Requirement | Mechanism | Code | Tests |
|---|---|---|---|
| Expert weights are plain linear weights that any FSDP handles like other linear layers | per-expert parameters named like TEGroupedMLP's; standard optimizer chain | `moonep_backend/experts.py:66-152` | `test_moonep_layer.py::test_layers_share_one_pool_set`; `test_moonep_training.py::test_distributed_optimizer_matches_alltoall`; `test_moonep_fsdp.py::test_matches_alltoall` |
| One buffer across all layers | process-global Buffer and `MoonEPExpertPools` | `moonep_backend/buffer.py:22-96`; `moonep_backend/storage.py:90-116` | `test_moonep_layer.py::test_layers_share_one_pool_set`, `::test_chain_matches_layers_run_alone`, `::test_interleaved_layers_match_sequential_order` |
| Persistent bf16 compute weights, copied into and out of the buffer during dispatch and combine | `stage` in dispatch and in combine's backward; `reduce_and_copy_out` in dispatch's backward | `moonep_backend/storage.py:130-216`; `moonep_backend/autograd.py:87-143` | `test_moonep_layer.py::test_skewed_routing_uses_prefetch`, `::test_weight_update_reaches_the_next_forward`, `::test_main_grad_like_ddp_and_fsdp` |
| ZeRO-1 and ZeRO-2 only | `validate_args` block and the wrapper check | `training/arguments.py:1708-1741`; `moonep_backend/grads.py:42-102` | `test_moonep_config.py::test_validate_args_*`; `test_moonep_fsdp.py::TestMoonEPWrapperCheck` |
| NVLink coverage asserted at initialization (kept) | collective check with WORLD consensus | `moonep_backend/topology.py:412-446`; `training/initialize.py:126-140` | `test_moonep_topology.py` |

---

## 5. Implementation map

### 5.1 `megatron/core/transformer/moe/moonep_backend/`
No module imports `moonep` at module level, so Megatron imports without MoonEP (`moonep_backend/__init__.py:10-11`).

| Module | Responsibility | Public API (path:line) |
|---|---|---|
| `__init__.py` | backend detection | `uses_moonep(config)` (`moonep_backend/__init__.py:19-24`); `iter_moonep_experts(module)` (`moonep_backend/__init__.py:27-31`) |
| `buffer.py` | process-global Buffer and pools, teardown | `get_moonep_buffer(group, *, max_tokens_per_rank, hidden_size, topk, num_experts, num_sms, token_padding)`: collective on first use, raises on a key mismatch, `explicitly_destroy=True` (`moonep_backend/buffer.py:22-70`). `get_moonep_expert_pools(buffer, group, hidden_size, ffn_hidden_size)`: keyed by `(EP ranks, H, Hi)` (`buffer.py:73-96`). `destroy_moonep_buffer()`: collective, pools then Buffer (`buffer.py:99-107`). `release_moonep_buffer()`: drops references, never blocks (`buffer.py:110-117`). `clear_moonep_pool_occupant()` (`buffer.py:120-123`). `moonep_buffer_is_live()` (`buffer.py:126-128`). |
| `storage.py` | the shared pools: staging, gradient reduction and hand-off | `MoonEPLayerState`: `layer_number`, `fwd_outstanding`, `pending_checkpoints`, `recompute_tokens`, `recompute_mismatches` (`moonep_backend/storage.py:50-68`). `MoonEPExpertPools(buffer, ep_group, hidden_size, ffn_hidden_size)` wraps `moonep.ExpertPools` with shapes gate `(Hi,H)`, up `(Hi,H)`, down `(H,Hi)` (`storage.py:90-116`). Methods: `weights(n)`, `grads(n)` (`storage.py:120-126`); `stage(state, plan, fc1, fc2)` returns whether it pushed (`storage.py:130-163`); `begin_grads(state)` (`storage.py:167-172`); `reduce_and_copy_out(plan, fc1, fc2)` (`storage.py:174-216`); `destroy()` (`storage.py:218-221`). Helpers: `_main_grad` uses Megatron-FSDP's lazy `get_main_grad()` (`storage.py:71-76`); `_dummy_grad` (`storage.py:79-87`). |
| `experts.py` | the expert module | `GroupedExpertWeights(num_weights, shape, dtype)` with `weights()` (`moonep_backend/experts.py:66-89`). `MoonEPGroupedMLP(num_local_experts, config, pg_collection, name)` (`experts.py:92-132`): `set_layer_number` (`experts.py:145-148`), `expert_weights()` reads the attributes on every call (`experts.py:150-152`), `weights_require_grad()` all or none (`experts.py:154-159`), `report_recompute_mismatches_` (`experts.py:161-175`), `backward_dw` no-op (`experts.py:177-178`), `forward(hidden_nvsh, cu_seqlens, route_weights_nvs)` over the pools (`experts.py:180-214`), `sharded_state_dict` (`experts.py:216-268`), `_load_from_state_dict` drops the `_extra_state` placeholders (`experts.py:270-274`). |
| `autograd.py` | autograd functions | `align_ep_group_on_first_use(group, key)` (`moonep_backend/autograd.py:19-30`). `MoonEPDispatch.apply(x, route_weights, topk, tpe, buffer, pools, state, holder, plan, wgrad, *params)` (`autograd.py:33-107`). `MoonEPCombine.apply(expert_output, buffer, plan, group, group_ranks, restage)` (`autograd.py:110-143`). `GroupedMMLiveGrad.apply(x, weight, offs, fp32_target)`: `fp32_target=None` skips the dW GEMM for frozen experts, and the pool is not saved for backward (`autograd.py:146-177`). |
| `manager.py` | the flex backend | messages (`moonep_backend/manager.py:30-70`); `_CheckpointedDispatch` (`manager.py:78-86`); `_MoonEPManager(group, num_local_experts, router_topk, num_experts, config)` (`manager.py:89-177`): `bind_experts` (`manager.py:179-182`), `setup_metadata` with a masked top-k (`manager.py:184-197`), `dispatch` (`manager.py:199-330`), `get_number_of_tokens_per_expert`, which returns the `[2*epn]` cumulative END offsets rather than counts (`manager.py:336-338`), `combine` (`manager.py:344-357`). |
| `grads.py` | once-per-iteration bookkeeping and the wrapper check | `assert_moonep_wrapper_supported(model)` (`moonep_backend/grads.py:42-102`); `finalize_moonep_step(model)`: runs the wrapper check once per chunk list (cached as `chunk._moonep_wrapper_checked`), reports recompute mismatches, resets `pending_checkpoints` and clears the occupant (`grads.py:105-121`). |
| `topology.py` | NVLink-coverage check, unchanged by the redesign | `assert_moonep_ep_group_nvlink_coverage(ep_group, *, consensus_group=None, force=False)` (`moonep_backend/topology.py:412-446`); `evaluate_moonep_nvlink_records` (`topology.py:276-381`); `moonep_handle_path` (`topology.py:200-231`); `fabric_health_warnings` (`topology.py:384-397`); `_reset_moonep_topology_cache` (`topology.py:449-451`). |

`_MoonEPManager.dispatch` in order (`moonep_backend/manager.py:199-330`): guards (`manager.py:212-219`); read the weights through the module (`manager.py:220-223`); decide `wgrad` and `grad_tracking` before `apply` (`manager.py:224-228`); phase-1 guard (`manager.py:230-232`); recompute detection (`manager.py:233-251`); fresh path with `ensure_params_ready` outside a backward, dispatch and recording (`manager.py:255-289`); replay path (`manager.py:290-323`); `stage` and the re-stage tuple for combine (`manager.py:324-330`).

### 5.2 Edited Megatron files

| File | Change |
|---|---|
| `transformer_config.py` | `'moonep'` in the backend `Literal` (`transformer_config.py:989`); fields `moe_moonep_max_tokens_per_rank` and `moe_moonep_token_padding` (`transformer_config.py:998-1006`); `_validate_moonep` (`transformer_config.py:1617-1841`); the `moe_enable_deepep` conflict (`transformer_config.py:2301-2306`); the call at the end of `__post_init__` (`transformer_config.py:4063-4065`) |
| `token_dispatcher.py` | the `moonep` branch (`token_dispatcher.py:1863-1878`) and the error string (`token_dispatcher.py:1879-1883`) |
| `moe_layer.py` | builds and binds `MoonEPGroupedMLP` (`moe_layer.py:373-388`); forwards the layer number (`moe_layer.py:226-227`) |
| `finalize_model_grads.py` | `finalize_moonep_step(model)` after `finish_grad_sync` (`finalize_model_grads.py:689-692`) |
| `parallel_state.py` | `destroy_model_parallel` releases the Buffer, pools and topology cache without collectives (`parallel_state.py:2517-2529`) |
| `training/arguments.py` | the MoonEP block of `validate_args` (`training/arguments.py:1695-1750`) |
| `training/initialize.py` | the collective topology check (`training/initialize.py:126-140`) |
| `training/training.py` | `_maybe_destroy_moonep_buffer` (`training/training.py:2287-2296`), called at the end of `pretrain` (`training/training.py:2277`) and in `train()`'s exit branch (`training/training.py:5571`) |
| `pyproject.toml` | `no_pypi_wheels` (`pyproject.toml:193`); no build isolation (`pyproject.toml:206`); static metadata `nvidia-cutlass-dsl>=4.6.2,<4.9` (`pyproject.toml:252-259`); git source at `30c2f8c` (`pyproject.toml:274`) |
| `uv.lock` | not regenerated: still `d9f044b` and the old metadata (`uv.lock:36-38, 2571, 2675-2677`) |
| `moe/README.md` | the MoonEP table row and section (`moe/README.md:312, 315-336`) |
| tests | §8 |

---

## 6. Config and CLI surface

**Fields.** New or changed fields become flags automatically (§2.5).

| Field / flag | Type (default) | Meaning |
|---|---|---|
| `moe_flex_dispatcher_backend` / `--moe-flex-dispatcher-backend moonep` | `Literal` (`deepep`) (`transformer_config.py:989`) | selects MoonEP |
| `moe_moonep_max_tokens_per_rank` / `--moe-moonep-max-tokens-per-rank` | `Optional[int]` (None) (`transformer_config.py:998`) | Buffer capacity S. `validate_args` derives `seq_length * max(micro_batch_size, eval_micro_batch_size) / CP`, divided by TP under sequence parallelism (`training/arguments.py:1698-1707`). Library users must set it (`moonep_backend/manager.py:137-140`). |
| `moe_moonep_token_padding` / `--moe-moonep-token-padding` | `int` (128) (`transformer_config.py:1004`) | `Buffer(token_padding=...)` |
| `moe_flex_dispatcher_num_sms` (existing) | `Optional[int]` (None) (`transformer_config.py:1078`) | `Buffer(num_sms=...)`; None means MoonEP's 32 (`MoonEP/moonep/api.py:252-253`) |

**Static rules in `TransformerConfig._validate_moonep`** (`transformer_config.py:1617-1841`). Each failure raises `ValueError("moe_flex_dispatcher_backend='moonep' requires <rule> (<reason>), got ...")` (`transformer_config.py:1627-1629`). Before it, `__post_init__` rejects `moe_enable_deepep` with MoonEP, because that flag would silently rewrite the backend to `deepep` (`transformer_config.py:2301-2306`). `freeze_base_model_for_mtp` is allowed (`test_moonep_config.py::test_allowed_overrides`).

| Rule | Lines | Reason in the message |
|---|---|---|
| `moe_token_dispatcher_type == 'flex'` | 1631-1634 | none |
| `num_moe_experts` set | 1635-1638 | "MoonEP is an expert-parallel MoE backend" |
| `1 < expert_model_parallel_size <= 128` | 1639-1644 | "MoonEP moves tokens between EP ranks and its planner supports at most 128 ranks" |
| `num_moe_experts % EP == 0` | 1645-1649 | none |
| `expert_tensor_parallel_size == 1` | 1650-1655 | "MoonEP moves whole experts; pass --expert-tensor-parallel-size 1 explicitly when tensor_model_parallel_size > 1" |
| `gtp_weight_remat_size == expert_gtp_weight_remat_size == 1` | 1656-1661 | "GTP shards the weights it rematerializes; MoonEP stages whole experts from unsharded weights" |
| `pipeline_model_parallel_size == 1`; `virtual_pipeline_model_parallel_size is None` | 1662-1673 | "phase 1 supports one outstanding gradient-tracking forward per MoE layer" |
| `bf16 and not fp16` | 1674-1678 | "MoonEP kernels are bf16" |
| `gated_linear_unit` and `activation_func is F.silu` | 1679-1684 | "gated_linear_unit=True and activation_func=F.silu, i.e. --swiglu" |
| `glu_linear_offset == 0`, no activation clamps, `moe_mlp_glu_interleave_size is None` | 1685-1698 | "glu_linear_offset == 0 and no activation clamps or GLU interleaving" |
| `not add_bias_linear` | 1699-1702 | "MoonEP experts have no bias; pass --disable-bias-linear" |
| dropless: no capacity factor, capacity padding, router padding for quantization or CUDA-graph expert padding | 1703-1717 | "...MoonEP planning traps on any other token count" |
| `fp8 is None`, `fp4 is None`, `not fp8_param` | 1718-1722 | rule text: "bf16 expert compute (no fp8/fp4)" |
| `moe_router_dtype == 'fp32'` | 1723-1727 | "fp32 gating logits and routing weights, as in the MoonEP reference model; pass --moe-router-dtype fp32" |
| no `moe_latent_size`, `moe_apply_probs_on_input` or hash routing layers | 1728-1737 | "they change the expert widths or the point where routing weights apply" |
| `not moe_shared_expert_overlap` | 1738-1742 | "overlapping shared experts on a side stream with MoonEP's cross-rank barrier kernels is untested" |
| no `overlap_moe_expert_parallel_comm`, `delay_wgrad_compute` or `overlap_dispatch_backward_with_experts_wgrad` | 1743-1753 | "phase 1 keeps one prefetch/reduce slot set per layer and computes dW inline" (outdated wording, §9 item 14) |
| `cuda_graph_impl == 'none'` | 1754-1758 | "MoonEP kernels are shaped by the per-call token count" |
| full recompute only with `recompute_method='block'` | 1762-1766 | "the only MoE activation recompute MoonEP supports" |
| no selective `moe` or `moe_act` recompute | 1767-1775 | "--moe-layer-recompute is selective 'moe'; use full recompute with recompute_method='block'" |
| `sequence_packing_scheduler is None` | 1777-1782 | "its packs differ in size between EP ranks and can exceed moe_moonep_max_tokens_per_rank". Never reached today: the generic packing check rejects every non-alltoall MoE dispatcher first with an AssertionError (`transformer_config.py:4060-4064`); `test_moonep_config.py::test_sequence_packing_scheduler_rejected` covers both |
| no `hybrid_context_parallel` or `variable_seq_lengths` | 1783-1788 | rule text: "the same token count on every EP rank" |
| `mlp_chunks_for_training == 1`, `not mtp_use_repeated_layer` | 1786-1792 | rule text: "one call per MoE layer per micro-batch" |
| no `fine_grained_activation_offloading` or `moe_paged_stash` | 1793-1799 | "untested with MoonEP" |
| `cpu_offloading` only with full block recompute | 1800-1806 | none |
| no meta-device or CPU initialization; `transformer_impl != 'inference_optimized'` | 1807-1817 | "MoonEP initializes its expert weights on the GPU under the expert-parallel RNG and builds its VMM pools when the first MoE layer is constructed" |
| `hidden_size % 128 == 0`, `moe_ffn_hidden_size % 128 == 0` | 1818-1824 | "MoonEP prefetch and grad-reduce tiles" |
| `1 <= moe_router_topk <= 32` | 1825-1828 | "MoonEP planner limit" |
| `moe_moonep_max_tokens_per_rank` None or > 0; `moe_moonep_token_padding >= 1` | 1829-1837 | none |
| `importlib.util.find_spec("moonep")` | 1838-1841 | "the moonep package (https://github.com/zhumakhan/MoonEP) to be installed" |

**Training-only rules in `validate_args`** (`training/arguments.py:1695-1750`, AssertionError).

| Rule | Lines | Message |
|---|---|---|
| S derived if unset | 1698-1707 | "MoonEP needs --seq-length to size its token buffer, or pass --moe-moonep-max-tokens-per-rank." |
| no torch FSDP2 | 1711-1715 | "MoonEP does not support torch FSDP2: it shards every parameter over the dense data-parallel group. ..." The generic rule against FSDP2 with EP fires first (`training/arguments.py:1049-1050`). |
| Megatron-FSDP v1 only | 1717-1720 | "MoonEP supports Megatron-FSDP v1 only (--megatron-fsdp-version 1); v2 requires parameter sharding." |
| dense and expert strategies in {`optim`, `optim_grads`} | 1721-1729 | "MoonEP with Megatron-FSDP supports the 'optim' (ZeRO-1) and 'optim_grads' (ZeRO-2) sharding strategies only, for dense and expert parameters (the default is optim_grads_params), got [...]" |
| `num_distributed_optimizer_instances == 1` | 1730-1733 | "MoonEP with Megatron-FSDP does not support hybrid sharding (--num-distributed-optimizer-instances > 1)." |
| `outer_dp_sharding_strategy == 'no_shard'` | 1734-1737 | "MoonEP with Megatron-FSDP does not support hybrid sharding (--outer-dp-sharding-strategy)." |
| no fine-grained parameter gathering | 1738-1741 | "MoonEP with Megatron-FSDP needs whole-layer parameter gathering (no --megatron-fsdp-enable-fine-grained-param-gather)." |
| DDP: `--ckpt-format torch_dist` | 1742-1745 | "MoonEP with Megatron DDP supports only --ckpt-format torch_dist." |
| no `--inprocess-restart` | 1746-1749 | "MoonEP (phase 1) does not support --inprocess-restart: a MoonEP device trap poisons the CUDA context, and Buffer teardown is collective." |

Generic Megatron rules also apply: Megatron-FSDP requires `--ckpt-format fsdp_dtensor` and `CUDA_DEVICE_MAX_CONNECTIONS != 1` (`training/arguments.py:1185-1189`), so pass `--data-parallel-sharding-strategy optim|optim_grads` (the default is `optim_grads_params`, `training/arguments.py:3446`) and `--megatron-fsdp-main-grads-dtype fp32` to keep the fp32 sums. With TP or CP > 1 on Hopper, DDP requires `CUDA_DEVICE_MAX_CONNECTIONS=1` (`training/arguments.py:1482-1508`; §9 item 1).

**Wrapper check for library users** (`moonep_backend/grads.py:42-102`). `finalize_moonep_step` runs it once per list of model chunks, after the first backward and before the first optimizer step (`grads.py:113-116`). It raises `ValueError` for:
- `FullyShardedDataParallelV2` and `TorchFullyShardedDataParallel` chunks: "... shards parameters." (`grads.py:59-64`);
- Megatron-FSDP dense or expert strategies outside {`optim`, `optim_grads`} (`grads.py:68-78`);
- hybrid sharding: `num_distributed_optimizer_instances > 1` or `outer_dp_sharding_strategy != 'no_shard'` (`grads.py:79-83`);
- `megatron_fsdp_enable_fine_grained_param_gather` (`grads.py:84-89`);
- a MoonEP experts module outside every FSDP unit of `chunk.fsdp_unit_modules` (`grads.py:90-102`).

**Runtime guards.** The manager's guards fire before the first barrier-bearing call, and their conditions are identical on every EP rank, so all ranks raise together (`moonep_backend/manager.py:212-213`).

| Guard | Where | Message |
|---|---|---|
| an inference engine is active | `moonep_backend/manager.py:214-215` | `INFERENCE_MSG` (`manager.py:30-34`) |
| more than S tokens | `manager.py:216-217` | `capacity_message` (`manager.py:37-44`) |
| non-bf16 hidden states | `manager.py:218-219` | `dtype_message` (`manager.py:47-49`) |
| a second gradient-tracking forward of a layer before its backward (no-grad forwards in between are fine) | `manager.py:230-232` | `guard_message` (`manager.py:52-62`) |
| a recompute dispatches another token count than its forward | `manager.py:248-251` | `recompute_tokens_message` (`manager.py:65-70`) |
| a layer's backward starts while another layer's dW wait in the pools | `moonep_backend/storage.py:167-172` | `interleaved_backward_message` (`storage.py:40-47`) |
| a layer's expert weights are partly frozen | `moonep_backend/experts.py:154-159` | `mixed_requires_grad_message` (`experts.py:58-63`) |
| a layer asks for another Buffer or pool shape | `moonep_backend/buffer.py:46-53, 83-90` | "must share one moonep.Buffer" / "one expert pool set" |
| `singleton_local_shards` checkpoint metadata | `moonep_backend/experts.py:231-234` | `NotImplementedError` |

Construction asserts: `moonep` importable, EP > 1, `E % EP == 0`, S set, and `E/R` local experts (`moonep_backend/manager.py:128-140, 158-162`); ETP = 1 and EP > 1 (`token_dispatcher.py:1866-1869`); a `ProcessGroupCollection` and `epn*EP == E` (`moonep_backend/experts.py:112, 119-122`).

**Recommended flags**: `moe/README.md:319-329`.

---

## 7. Unsupported features and roadmap

- **PP and VPP.** `_validate_moonep` rejects them (§6) because 1F1B runs several gradient-tracking forwards of a layer before their backwards, while the phase-1 guard allows one (`fwd_outstanding` is a bool, `moonep_backend/storage.py:56`) and the recompute keeps one record per layer (`moonep_backend/manager.py:243-251`). The shared pools already re-stage in every backward from the plan saved on the combine node (`moonep_backend/autograd.py:131, 139-142`), and EP peers share a PP stage and run the same schedule (`parallel_state.py:903-906`), so the barrier order stays symmetric. Lifting the rule needs a per-layer counter instead of the bool, a record per checkpoint, and 1F1B and interleaved tests [unverified until tested].
- **EP-communication overlap (combined 1F1B) and delayed wgrad.** Combined 1F1B runs one micro-batch's dispatch or combine on a communication stream while another micro-batch's expert GEMMs run, but two layers' expert computations cannot share one pool set (`MoonEP/README.md:98`), and all calls on one Buffer must be serialized. Options: a second pool set alternated by micro-batch parity, or serializing the overlap. Delayed wgrad breaks the rule that a layer's dW sit in the pools only between its GEMM backward and its reduce (ADR-7).
- **ZeRO-3** (Megatron-FSDP `optim_grads_params` and v2, torch FSDP2, GTP weight rematerialization): staging would need an explicit gather of the weight containers before each stage, in forward and backward, and a release afterwards (ADR-8).
- **Other Megatron-FSDP modes**: `no_shard` (use DDP), hybrid sharding and fine-grained parameter gathering are rejected; the last would need the containers gathered explicitly.
- **Other recompute modes.** Uniform full recompute and selective `moe` recompute use the same `CheckpointFunction`, so the replay would apply unchanged, but they are untested. Selective `moe` plus `shared_experts` nests the shared experts' checkpoint inside the `moe` one, and `CheckpointFunction.forward` clears the global flag unconditionally (`tensor_parallel/random.py:631`). That hides the outer checkpoint from the dispatcher, which would then plan and push twice: correct, but more work [unverified].
- **CUDA graphs**: the token count is a kernel argument and `plan.nvs_s` sizes the outputs, so graphs need bucketed `s` and a warm-up (`MoonEP/README.md:116`).
- **Inference engines, THD packing, variable sequence lengths**: these need a different `s` per rank, which MoonEP supports through `total_num_tokens`, a host-known sum over the EP group (`MoonEP/README.md:114`); the backend does not pass it.
- **fp8 and MXFP4 experts**: MoonEP's prefetch already pushes `int8` and `uint8` weights, and block scales with `uint8` weights in the same launch (`MoonEP/moonep/prefetch.py:38-42, 606-609`).
- **fp32 dW GEMM**: blocked because `torch._grouped_mm` requires the output dtype to match bf16 inputs (A.1).
- **Multi-node NVLink (MNNVL)**: the fabric-handle path is untested (`MoonEP/varlen_moon_ep.md:373`).
- **ETP > 1**: out of scope.
- **Warm-up**: a construction-time dispatch, push, combine and reduce would compile every kernel before training, so no first compile could stall a rank mid-step (§9 item 5).

---

## 8. Test plan

**Running.** Unit tests target one node of 8 GPUs in one NVLink domain (some cells skip with fewer) and run under `torch.distributed.run`, with `PYTHONPATH` covering MoonEP-Fork and Megatron-LM:
```
python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/transformer/moe/test_moonep_layer.py
```
- Do not pass `-x`. Tests collect their checks in `RankChecks` and raise on every rank after the last collective (`moonep_utils.py:167-198`), because a rank that raised alone would leave its peers in a MoonEP barrier or the next NCCL call.
- Unset `CUDA_DEVICE_MAX_CONNECTIONS` for the Megatron-FSDP ZeRO-2 cells, which skip otherwise (`test_moonep_fsdp.py::needs_concurrent_streams`).
- Markers: `pytest.mark.internal`, plus `requires_moonep`, which skips without `moonep`, CUDA, sm_90 or two GPUs (`moonep_utils.py:21-36`). The CI MoE bucket globs `tests/unit_tests/transformer/moe/**/*.py` with a 1800 s limit (`h100/unit-tests.yaml:269-275`); the CI image has no `moonep`, so these tests skip there.
- Shared helpers (`moonep_utils.py`): configs (`moonep_utils.py:38-82`), `build_moe_layer` (`moonep_utils.py:85-90`), weight copies (`moonep_utils.py:93-113`), grad readers that use `main_grad` once a wrapper took the grads (`moonep_utils.py:116-139`), `moonep_calls`, which counts fresh, replayed and combine-backward dispatches, pushes and reduces (`moonep_utils.py:142-158`), and collective teardown (`moonep_utils.py:201-214`).
- Each file keeps one Buffer shape where it can, because a new `(S, H, K, E, EP)` recompiles MoonEP's kernels, and it orders cells by EP size. Sharding cells use EP=4 on 8 GPUs, since at EP=8 expert-DP is 1 and ZeRO-1/2 shards nothing of the experts.

| File | Cases | EP | Coverage |
|---|---|---|---|
| `test_moonep_config.py` | 91 | CPU | every `_validate_moonep` rule (`test_forbidden_overrides`); allowed block recompute, block recompute with CPU offloading and `freeze_base_model_for_mtp`; the `moe_enable_deepep` conflict; missing package; `validate_args` accepting DistOpt with both overlaps and Megatron-FSDP `optim`/`optim_grads`, and rejecting `optim_grads_params` (also as the CLI default), v2, hybrid sharding, fine-grained gathering, torch FSDP2 and non-`torch_dist` DDP formats |
| `test_moonep_topology.py` | 24 | 8, 4 | 18 pure evaluator cases (pass, cross-host, fabric health, P2P status, handle path, multicast, duplicate GPU, compute capability, tmpdir, missing fields); live pass at EP=world and world/2; live negatives raising the identical error on every rank; WORLD consensus; contained local-probe exceptions |
| `test_moonep_layer.py` | 24 | 8 | Buffer cache, destroy and release; one pool set shared by all layers, its shapes and all-rank mapping, parameter names, dtype and `allreduce`; `Float16Module` casting; parity with alltoall plus SequentialMLP (output, input, router and expert grads; 1 push, 1 reduce); below capacity (`s < S`); skewed routing, with every used slot equal to its owner's bf16 weights; 3 accumulated micro-batches; a 3-layer chain bitwise equal to each layer alone (balanced and skewed; 5 pushes); interleaved `F_A F_B B_B B_A` equal to sequential order; no-grad forwards between a forward and its backward; an in-place weight update reaching the next forward; `main_grad` hand-off, accumulating and overwriting; frozen router and input; frozen experts (no grads, no `reduce_grad`, partly frozen raises); the phase-1 guard; capacity and inference guards; initialization equal to SequentialMLP's |
| `test_moonep_recompute.py` | 17 | 8 | replay against recompute without replay, bitwise (balanced and skewed, 3 micro-batches); an eval forward dropping the record; schedules with several waiting checkpoints planning afresh; `finalize_moonep_step` forgetting a forward without backward; a stuck checkpoint flag not replaying outside a backward; mismatch reporting (1 token quiet, all tokens warn); `GPTModel` with 4 MoE layers under block recompute, also with CPU offloading inside and past the checkpointed window, bitwise against the reference with `2*(2N-1)` pushes |
| `test_moonep_training.py` | 11 | 8, 4 | DDP `main_grad` parity with SequentialMLP after `finalize_model_grads` (expert-DP 1 and 2, per-token loss, `average_in_collective`); DDP's NaN check covering expert grads; parameter groups and overrides (MuP) equal to SequentialMLP's; 5-step loss parity under plain DDP with the alltoall optimizer chain; the distributed optimizer's chain, buffers and element ownership, plus 3 steps at expert-DP 1 and 2; `overlap_grad_reduce` and `overlap_param_gather` at expert-DP 2 with one bucket per layer's experts, expert replicas equal after every forward |
| `test_moonep_checkpoint.py` | 7 | 8, 4 | sharded keys, shapes, offsets and replica ids equal to SequentialMLP's, each entry holding the parameter; bitwise round trips MoonEP to MoonEP, SequentialMLP to MoonEP and MoonEP to SequentialMLP, with a bitwise re-save; EP 8 to 4 resharding; a GPT with the distributed optimizer at EP=4 saved in `dp_reshardable` and `fully_reshardable`, loaded into a fresh model, and stepping bitwise like the original |
| `test_moonep_fsdp.py` | 23 | 4 | Megatron-FSDP `optim`/`optim_grads` against alltoall under the same wrapper (balanced and skewed, per-expert grad norms); against MoonEP under DDP with the distributed optimizer (tight tolerance); a ZeRO-2 stress cell with 4 MoE layers, 4 micro-batches, block recompute and skewed routing; `fsdp_dtensor` naming by global expert id; `fsdp_dtensor` round trip from `optim` to `optim_grads`; the wrapper check on CPU (14 cases: other strategies including `no_shard`, hybrid sharding, fine-grained gathering, MoE layers outside FSDP units, parameter-sharding wrappers, accepted ZeRO-1/2 configurations) |
| `test_moonep_kernels.py` | 7 | 1, 8 | `GroupedMMLiveGrad` against a per-group fp32 reference with empty groups giving exact-zero dW, rows past the last segment never read, and the target's old contents overwritten; no dW GEMM without a target; bitwise replays under stream contention; a MoonEP `MoELayer` replaying bitwise, balanced and skewed, with and without a block-recompute checkpoint |
| `test_moe_kernels.py[flex-moonep-ep2]` | 1 | 2 | the generic MoE-layer replay cell with standard grad zeroing (`test_moe_kernels.py:548-561`) |

Counts are the cases pytest collected on 2026-10-06 at `a17669776` (Appendix A.3 (2)).

**Determinism registration.** The `moe_moonep` entry registers `autograd.py`, `experts.py`, `manager.py` and `storage.py` with `test_moonep_kernels.py` (`kernels/manifest.py:324-344`). `token_dispatcher.py` and `moe_layer.py` are sources of other entries, which `test_moe_kernels.py[flex-moonep-ep2]` covers.

**Functional tests.** `gpt3_mcore_te_moonep_ep8_1node` and `gpt3_mcore_te_moonep_ep8_resume_torch_dist_1node`: 12 layers, EP=8, distributed optimizer, `torch_dist`, `--deterministic-mode`, `CUDA_DEVICE_MAX_CONNECTIONS=1`. The recipe keeps them at `scope: [mr-github-broken]` until the CI image ships `moonep` (`h100/moe.yaml:282-288`). Golden values come from the first green CI run. There is no Megatron-FSDP functional twin yet.

**Gaps.** No PP, TP > 1 with sequence parallelism, or multi-node coverage. No test overlaps NCCL with MoonEP under `CUDA_DEVICE_MAX_CONNECTIONS=1`. No unit test compares against TEGroupedMLP (the end-to-end runs in A.3 do). No test loads a checkpoint of the earlier design. No `fsdp_dtensor` interchange test between MoonEP and TEGroupedMLP. No performance measurement.

---

## 9. Risks and open questions

1. **NCCL concurrent with MoonEP barrier kernels.** DDP `overlap_grad_reduce` and `overlap_param_gather`, and Megatron-FSDP ZeRO-2, which forces overlapped reduce-scatters (`megatron_fsdp/megatron_fsdp.py:346-352`), run NCCL kernels while MoonEP's cooperative barrier kernels spin on other streams. MoonEP's barrier-bearing kernels use `num_sms` CTAs (32 by default, `MoonEP/moonep/api.py:252-253`). With `CUDA_DEVICE_MAX_CONNECTIONS=1`, all streams share one hardware queue, which can hold one rank's collective behind a barrier that waits for a peer held behind its own collective [unverified; this is the reasoning of `test_moonep_fsdp.py::needs_concurrent_streams`, whose ZeRO-2 cells skip under that setting]. Megatron-FSDP already refuses it (`training/arguments.py:1185-1186`), but DDP *requires* it with TP or CP > 1 on Hopper (`training/arguments.py:1482-1508`). So MoonEP with TP > 1, DDP overlap and H200 is an untested and possibly deadlocking combination [unverified: run `test_overlapped_grad_reduce_and_param_gather` with `CUDA_DEVICE_MAX_CONNECTIONS=1`]. Question for the MoonEP author: is there a deadlock risk with concurrent NCCL, and should `num_sms` stay well below the SM count?
2. **Host-only `moonep._C` on `slinky-1`.** The native runs use `bindings.cu` compiled as C++ with g++ (§A.3), not MoonEP's `CUDAExtension` build with nvcc (`MoonEP/setup.py:50-66`). That is equivalent only because `csrc/` contains no device code, which holds at `30c2f8c` [verified by grep for `__global__` and `__device__`; re-check whenever MoonEP's `csrc/` changes]. The official build was not exercised, because the node has no compiler and the venv's pip CUDA toolkit pieces are mismatched (nvcc and ptxas 13.0.88 versus nvvm and crt 13.4.92).
3. **The MoonEP pin.** `pyproject.toml:274` pins `30c2f8c`, which has no `moonep.ExpertPools`, so `from moonep import ExpertPools` (`moonep_backend/storage.py:103`) fails on the pinned package. Once MoonEP-Fork's `ExpertPools` is committed and pushed, bump the pin to that commit. The README's "as pinned in `pyproject.toml`" is wrong until then (`moe/README.md:317`).
4. **`uv.lock` is stale.** It still resolves `d9f044b` with the old `nvidia-cutlass-dsl>=4.4.2,<4.6` metadata (`uv.lock:36-38, 2571, 2675-2677`). Regenerate it with `uv lock` inside the CI container after the pin bump; never edit it by hand. MoonEP's own `setup.py` pins `nvidia-cutlass-dsl==4.6.2` (`MoonEP/setup.py:73`), while the kernels ran on 4.8.0.
5. **Fixed device barrier timeout.** MoonEP traps after about 100 s (`MoonEP/moonep/_common.py:36-37`). Any host stall of one EP rank longer than that while its peers sit in a MoonEP kernel traps the device: a slow data loader, a slow checkpoint write, or JIT-compile skew. The first-use alignment barriers cover compile skew per call site (§1.3). Question for the author: a configurable timeout, or ahead-of-time compilation?
6. **A forward without its backward blocks the next one.** A gradient-tracking forward that never gets a backward, for example `forward_backward_func(forward_only=True)` with autograd enabled and trainable experts, leaves `fwd_outstanding` set, and the layer's next gradient-tracking forward raises the phase-1 guard (`moonep_backend/manager.py:230-232, 252-253`). `finalize_moonep_step` does not reset it (`moonep_backend/grads.py:118-120`), and finalize does not run for forward-only schedules anyway (`pipeline_parallel/schedules.py:864`). Megatron's own forward-only paths disable autograd (§2.12).
7. **Staging cost.** Every stage copies one layer's bf16 expert weights into the pools and pushes them: 2N-1 copies and pushes per micro-batch, against N pushes and no copies in the earlier design with per-layer pools. Neither step time nor bandwidth has been measured [unverified: compare step time against alltoall at the A.3 sizes, then at production `(E, H, Hi)`].
8. **Checkpoints of the earlier design.** They store fp32 masters under the same `torch_dist` keys, fragments and `_extra_state` placeholders (`git show dba89efc5:megatron/core/transformer/moe/moonep_backend/experts.py`), so the model weights should load into bf16 parameters through DCP's dtype conversion. Their `MoonEPOptimizer` state cannot map onto the expert `DistributedOptimizer`, hence `--no-load-optim` (`moe/README.md:335`) [unverified: no test loads such a checkpoint; save with `dba89efc5` and load with the working tree].
9. **Private MoonEP API.** The manager reads `Buffer._require_ctx()` to check `E/R` (`moonep_backend/manager.py:158`). Everything else goes through public calls and `moonep.ExpertPools`.
10. **`fsdp_dtensor` caveats.** The `fsdp_dtensor` optimizer-checkpoint tests require TE's FusedAdam, because other Adams skip the empty shards of uneven sharding and ranks then disagree on optimizer keys (`test_moonep_fsdp.py::needs_te_fused_adam`). Megatron-FSDP loads GLU models with `strict=False` (`mcore_fsdp_adapter.py:315-335`), so a key mismatch could load nothing silently; the round-trip test asserts that the weights changed.
11. **Megatron side, unrelated to MoonEP.** Under Megatron-FSDP, the fused-wgrad path of Megatron's own linear layers calls `_wgrad_gemm(weight.main_grad, ...)` with its default `accumulate=False` and ignores `overwrite_main_grad` (`tensor_parallel/layers.py:638, 803-808`). Under `optim` the main-grad buffer accumulates micro-batches (`megatron_fsdp/megatron_fsdp.py:720-729`), so that path would drop all but the last micro-batch's wgrad [read from the code, not run]. MoonEP's hand-off honors the flag (`moonep_backend/storage.py:195-211`). The end-to-end runs use `--no-gradient-accumulation-fusion`.
12. **Repository hygiene.** `24665641a` committed a compiled dataset helper without its ABI suffix, `megatron/core/datasets/helpers_cpp` (ELF, 311,544 bytes), which `.gitignore`'s `*.so` does not catch. The native runs load it because the nodes have no `make` (§A.3). Drop it from the PR.
13. **Topology check versus broken multicast binding.** The check reads `nvl_multicast_supported()` (`moonep_backend/topology.py:108-114`), which reports `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` (`MoonEP/csrc/nvl_shared_buffer.cuh:383-392`), not whether `cuMulticastBindMem` works. A node like `slinky-0`, where binding fails (A.2), may pass the check and then die in `Buffer()` through `exit()` [unverified: A.2 stubbed the probe; read the attribute on such a node and check that every rank exits rather than hangs].
14. **Outdated wording.** The EP-overlap rule's reason still says "phase 1 keeps one prefetch/reduce slot set per layer" (`transformer_config.py:1747-1748`); the slot set is now shared by all layers.
15. **Open questions for the MoonEP author.** Are dispatch, combine and `reduce_grad` deterministic across runs and routing patterns (the grad-reduce prescan keeps a fixed slot order, `MoonEP/moonep/grad_reduce.py:215-220`)? Is a grouped GEMM with fp32 dW accumulation planned (dW is rounded to bf16 per micro-batch, `moonep_backend/autograd.py:174`)? Which fabric predicate is intended for MNNVL?

---

## Appendix A: Validation record

### A.1 B200 VM, MoonEP `d9f044b` (historical)
- One node of 8x B200 (sm_100, NV18 between all pairs), before any port. `my_moon_ep.py` ran at 86.2 ms per step (E=128, K=16, H=4096, Hi=8192, S=4096); `moe_moon_ep.py` matched its dense reference, and repeated steps were bit-identical.
- Facts the current design still relies on: on torch 2.14, `torch._grouped_mm` writes exact zeros for empty groups in the 2D-by-2D dW (20 runs under NaN-poisoned allocations), rejects `out_dtype=float32` with bf16 inputs, accepts a transposed `[G,out,in]` weight view at equal speed, and leaves rows past `offs[-1]` unwritten. NVML reported NVLink P2P OK for all GPU pairs; VMM and multicast granularity were 2 MiB; DCP converted dtypes on load in both directions.

### A.2 `slinky-0`, MoonEP `30c2f8c`, emulated (2026-10-05, historical)
- `slinky-0` (8x H200, driver 580.126.09) cannot bind NVSwitch multicast (`cuMulticastBindMem` fails with `CUDA_ERROR_ILLEGAL_STATE`) and has no compiler. A test-only emulation replaced `moonep._C` with cuda-python VMM over POSIX fds and the multicast PLAN broadcast with unicast stores, and raised the barrier timeout to 5 minutes.
- Under it, MoonEP's dense-reference checks passed, the earlier design's unit tests passed, a `pretrain_gpt.py` run resumed bit-identically from a `torch_dist` checkpoint, and block recompute with and without CPU offloading trained bit-identically. None of this ran the current design.

### A.3 `slinky-1`, native, MoonEP `30c2f8c` + `ExpertPools` (2026-10-06, in progress)

**Environment.** 8x H200 (sm_90) with working NVSwitch multicast, driver 580.126.09, the venv of the header, `PYTHONPATH=/pfs/code/MoonEP-Fork:/pfs/code/Megatron-LM`, `PYTHONDONTWRITEBYTECODE=1`. Nothing is emulated: the real multicast path and the real VMM and IPC code of `moonep._C` run.
- `moonep._C` is built on the login node as a host-only C++ extension: `csrc/bindings.cu` copied to a `.cpp` and compiled by g++ through `torch.utils.cpp_extension.CppExtension`, linked against `libcuda`, `libcudart` and `libc10_cuda`. MoonEP's kernels are CuTe DSL kernels that nvidia-cutlass-dsl compiles at first use (§9 item 2).
- A pytest stand-in (a minimal `pytest` package plus a runner, in the coordinator's scratchpad) runs the unit tests under `torch.distributed.run --nproc-per-node 8`. It supports the marks and fixtures the MoonEP tests use: `parametrize`, `skipif`, `skip`, `raises`, `param`, `caplog`, `capfd`, `monkeypatch`, `tmp_path` and `tmp_path_dist_ckpt`.
- `pretrain_gpt.py` runs through a wrapper that skips the runtime build of the dataset helpers and loads the prebuilt `megatron/core/datasets/helpers_cpp` (the nodes have no `make`), with `TORCHDYNAMO_DISABLE=1` (no C compiler for Triton launchers).

**(1) MoonEP's own checks.** `check_against_dense_reference` (`MoonEP/my_moon_ep.py:431-495`), as the coordinator's check script runs it: `MoonEPMoe(E=8*R, K=8, H=1024, Hi=1024, S=512)` with the router rows of half the experts zeroed, so the prefetch and grad-reduce paths run. Plus MoonEP-Fork's `ExpertPools` test.

| Check | Result | Max relative error (out, dx, router grad, expert grads) |
|---|---|---|
| `s = S` (512) | TBD | TBD |
| `s = 131` | TBD | TBD |
| `s = 1` | TBD | TBD |
| per-rank `s` with `total_num_tokens` | TBD | TBD |
| `MoonEP/tests/test_expert_pools.py` (6 tests: layout and aliasing, push, reduce, two layers sharing pools, idempotent destroy, argument checks) | TBD | n/a |

**(2) Megatron unit tests** (cases as counted in §8). Run on 2026-10-06 at `a17669776` with MoonEP-Fork `11713d2`, under real pytest 9.1.1 rather than the stand-in: one `torch.distributed.run --nproc-per-node 8 -m pytest -q -p no:cacheprovider <file>` per file, `NCCL_ALGO=Ring`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `CUDA_DEVICE_MAX_CONNECTIONS` unset. `test_moonep_fsdp.py` ran in a separate job without the two determinism variables. All 8 ranks reported the same result for every file. Wall time includes process start and MoonEP's first-use kernel compilation.

| File | Cases | Passed / skipped / failed | Wall time |
|---|---|---|---|
| `test_moonep_config.py` | 91 | 91 / 0 / 0 | 1:34 |
| `test_moonep_topology.py` | 24 | 24 / 0 / 0 | 1:18 |
| `test_moonep_layer.py` | 24 | 24 / 0 / 0 | 5:15 |
| `test_moonep_recompute.py` | 17 | 17 / 0 / 0 | 4:05 |
| `test_moonep_training.py` | 11 | 11 / 0 / 0 | 5:02 |
| `test_moonep_checkpoint.py` | 7 | 7 / 0 / 0 | 4:06 |
| `test_moonep_fsdp.py` | 23 | 23 / 0 / 0 | 4:49 (pytest) |
| `test_moonep_kernels.py` | 7 | 7 / 0 / 0 | 2:32 |
| `test_moe_kernels.py[flex-moonep-ep2]` (`-k moonep`, 55 deselected) | 1 | 1 / 0 / 0 | 2:08 |

**(3) End-to-end ZeRO matrix.** `pretrain_gpt.py --mock-data` for 8 iterations: 4 layers, all MoE; hidden 512, 8 heads, sequence 1024, micro-batch 4 and global batch 64 (two micro-batches per step on 8 ranks); 16 experts, top-4, `moe_ffn_hidden_size` 1024, fp32 router, aux loss 1e-2; `--transformer-impl transformer_engine`, `--deterministic-mode`, `--no-gradient-accumulation-fusion`, distributed optimizer. The baseline is alltoall with TEGroupedMLP (`--moe-grouped-gemm`) under the same wrapper. DistOpt rows save `torch_dist`; Megatron-FSDP rows save `fsdp_dtensor` with `--megatron-fsdp-main-grads-dtype fp32`. CDMC is `CUDA_DEVICE_MAX_CONNECTIONS`. Loss and grad norm are at iteration 8; memory is the max allocated memory in the log.

| Configuration | EP | CDMC | lm loss, alltoall | lm loss, MoonEP | grad norm, alltoall | grad norm, MoonEP | max alloc, alltoall | max alloc, MoonEP |
|---|---|---|---|---|---|---|---|---|
| DistOpt | 4 | 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| DistOpt + `--overlap-grad-reduce --overlap-param-gather` | 4 | unset | TBD | TBD | TBD | TBD | TBD | TBD |
| DistOpt | 8 | 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| Megatron-FSDP `optim` | 4 | unset | TBD | TBD | TBD | TBD | TBD | TBD |
| Megatron-FSDP `optim_grads` | 4 | unset | TBD | TBD | TBD | TBD | TBD | TBD |
| DistOpt + block recompute (2 layers) + CPU offloading (2 layers) | 4 | 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| Megatron-FSDP `optim_grads` + block recompute (2 layers) | 4 | unset | TBD | TBD | TBD | TBD | TBD | TBD |

**(4) Checkpoint resume.** The same model at EP=4 runs 12 iterations, saving at 6 and 12; a second run resumes from iteration 6. Iterations 7-12 must log bit-identical `lm loss`.

| Wrapper | Format | Iterations 7-12 bit-identical |
|---|---|---|
| DistOpt | `torch_dist` | TBD |
| Megatron-FSDP `optim_grads` | `fsdp_dtensor` | TBD |
