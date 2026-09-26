# MOONEP.md: integrating MoonEP into Megatron-LM (phase 1)

**Scope.** This is a plan for hand-implementing a MoonEP expert-parallel backend inside Megatron-LM. It covers fp32-master / bf16-compute training, an NVLink-coverage assertion at initialization, grad-zeroing semantics that never delete MoonEP `.grad` aliases, and a model-level `zero_moe_grads()` that warns on every missed zeroing. Every Megatron and MoonEP claim cites the source it was checked against.

**Baseline**

| Item | Value |
|---|---|
| Megatron-LM | `/home/ubuntu/Megatron-LM` @ `1912044d7` (only `pyproject.toml` modified, plus this file) |
| MoonEP | `/home/ubuntu/MoonEP` @ `d9f044b` (`moonep 0.0.1`, editable install) |
| Python env | `/home/ubuntu/MoonEP/.venv`: Python 3.12.3, torch 2.14.0+cu132 (CUDA 13.2), megatron-core 0.20.0, cuda-bindings 13.4.3 (`cuda.bindings.nvml` importable) |
| Not installed | `transformer_engine`, `apex`, `pynvml`, `pytest`, `pytest-mock`, `click`, `requests`, `pybind11`, `isort`; `uv` is not on PATH |
| Hardware | 1 node × 8 NVIDIA B200 (sm_100, 148 SMs), NV18 between all pairs. NVML: NVLink P2P status OK for all 56 ordered pairs; every GPU reports fabric state COMPLETED, status 0, all-zero ClusterUUID, clique 0, health summary HEALTHY. No `/dev/nvidia-caps-imex-channels`. |

**Citation conventions**
- `path:line` refers to the commits above.
- MoonEP paths start with `MoonEP/`.
- torch paths are inside `/home/ubuntu/MoonEP/.venv/lib/python3.12/site-packages/`.
- "**Verified on this VM**" means I ran it (Appendix A has the commands).
- "**[unverified]**" means an inference. The way to check it is always given.

---

## 0. TL;DR

- **Where MoonEP plugs in.** It becomes a fourth backend of Megatron's flex dispatcher: `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend moonep`.
  - A new `_MoonEPManager` does routing-format conversion, `Buffer.dispatch`, `Buffer.prefetch_weight` and `Buffer.combine`.
  - The per-micro-batch `Buffer.reduce_grad` runs inside each layer's dispatch-backward.
  - `MoELayer` builds `MoonEPGroupedMLP` experts instead of the spec's experts, and forwards `set_layer_number` to them (the layer number only arrives after construction, §2.1).
- **Mixed precision follows MoonEP's reference model** (`MoonEP/my_moon_ep.py:150-239`).
  - fp32 master `nn.Parameter`s are created with `requires_grad=False`. Their `.grad` and `.main_grad` permanently alias fp32 VMM gradient rows, which are zeroed at construction.
  - The bf16 compute copies live in VMM pools shared over NVLink.
  - `requires_grad=False` is the truth, since autograd never produces these gradients. It keeps the masters out of DDP buffers, DDP hooks, the distributed-optimizer layout and the dense/expert Megatron optimizers with no DDP edits.
  - A new `MoonEPOptimizer(FP32Optimizer)` is appended last to the `ChainedOptimizer`. It contributes its grad norm to the chain's joint norm and single clip coefficient (`megatron/core/optimizer/optimizer.py:1869-1885, 2013-2030`), then runs the Adam(W) step and `sync_compute_weights()`.
- **NVLink assertion.**
  - Static invariants go in `TransformerConfig.__post_init__` (config construction).
  - The collective check runs in `initialize_megatron` right after the EP groups exist, before seeding and model build, with consensus over the whole world. It checks:
    - NVML NVLink P2P for every pair of ranks on the same kernel instance (same `boot_id`);
    - fabric state, cluster, clique and health for cross-host pairs only;
    - NVSwitch multicast and compute capability on every rank;
    - which memory-handle path MoonEP will pick, and whether that path can work.
  - No rank raises before the collectives finish: every local probe failure becomes a problem line. Either every rank passes or every rank raises the same actionable error. The probes pass on this node (verified).
  - A cached, EP-local re-check runs in `_MoonEPManager.__init__` for library users who skip `initialize_megatron`.
  - **Interpretation of requirement 3**: the collective half cannot run inside `__post_init__` (§2.5), so "config-initialization level" means config construction plus the first point of initialization where the EP group exists. §4 states this explicitly.
- **zero_grad.**
  - `MoonEPOptimizer.zero_grad()` never deletes or zeroes MoonEP grads. It only records that the framework reached its zeroing point.
  - `model.zero_moe_grads()`, a new `MegatronModule` method that also works on DDP and `Float16Module` wrappers, zeroes them in place. `train_step` calls it right after `optimizer.zero_grad()`.
  - A per-layer state machine (starts CLEAN) warns on every gradient-producing forward that would add to stale grads. Gradient accumulation and `torch.no_grad()` eval never trigger it. The notice goes through `logger.warning`, falls back to stderr when the logging level hides warnings, and raises in strict mode.
  - Extra guards: grads that reach the optimizer without MoonEP finalization raise; finalized grads discarded before any optimizer step warn; broken `.grad` aliases warn and are re-bound.
- **Phase-1 limits** (asserted, §6): PP=1, no VPP, no EP comm overlap, no MoE recompute, no CUDA graphs, ETP=1, SwiGLU, no bias, dropless routing, bf16, fp32 router dtype, `torch_dist` checkpoints, Adam only, DDP `overlap_grad_reduce`/`overlap_param_gather` off, no inference engines, no in-process restart, no freeze flags.

### Architecture (one EP rank `r`; it owns experts `[lo,hi) = [r·E/R, (r+1)·E/R)`; `B = E/R` prefetch slots)

```
FORWARD
 x [s,b,H] bf16 ─► TopKRouter (bf16 weight param, fp32 logits via --moe-router-dtype fp32)
                   └► probs[T,E] fp32, routing_map[T,E] bool      (T = s*b tokens on this rank)
 MoEFlexTokenDispatcher.dispatch_preprocess  (@jit_fuser = torch.compile)
   └► _MoonEPManager.setup_metadata: w[T,K] fp32 (masked top-k of probs), idx[T,K] int32, tpe[E] int32
 token_dispatch ─► _MoonEPManager.dispatch: guards (inference, capacity, dtype, second forward)
               ─► MoonEPDispatch.apply ─► moonep.Buffer.dispatch      (ONE process-global Buffer)
               ─► Buffer.prefetch_weight(plan): this layer's pool rows [E,E+B) ◄── peers' rows [0,E) (NVLink read)
   └► h_nvs[nvs_s,H] bf16, w_nvs[nvs_s] fp32, cu_seqlens[E+B] int32 (cumulative END offsets), plan
 MoonEPGroupedMLP.forward   (bf16 VMM pools W_x [E+B, out, in], Megatron layout)
   grad-state check (warns if STALE)
   gate = gmm(h, W_gate^T) ; up = gmm(h, W_up^T) ; a = silu(gate)*up
   a = (a.float() * nan_to_num(w_nvs)).bf16()      # probs before fc2, as Megatron does
   y = gmm(a, W_down^T)
 token_combine ─► MoonEPCombine.apply ─► Buffer.combine(plan)
   (fp32 accumulation of each token's rows; duplicate rows are first pre-summed in fp32 and stored back as bf16)
   └► out [T,H] bf16 ─view─► [s,b,H] (+ shared experts) ─► residual

BACKWARD (same CUDA stream, per layer, per micro-batch)
 MoonEPCombine.backward : Buffer.dispatch(grad_out, plan)          -> grad_y [nvs_s,H]
 GroupedMMLiveGrad.backward (x3):
   dX = gmm(g, W) over all E+B groups
   dW = gmm(g^T, x) over the 2*epn live groups (bf16) then
        += fp32 VMM G_x[lo:hi]    (== master.grad == master.main_grad)
        += fp32 VMM G_x[E:E+B]    (== this rank's reduce chunk RB_x[r])
 MoonEPDispatch.backward: Buffer.combine(plan, grad_h, grad_w) -> grad_x, grad_w[T,K] (to router)
                          launch_inter_rank_sync ; Buffer.reduce_grad(plan)
                          (owner pulls peers' RB slots of its experts, += G_x[lo:hi]; each rank then clears its consumed slots)
 finalize_model_grads (once per step): DDP dense sync ;
                          MoonEP: NaN/Inf/large check ; G_x[lo:hi] *= 1/|dp_cp| ; SUM over expert-DP if >1 ;
                          per-token loss: *= 1/num_tokens

OPTIMIZER STEP  (ChainedOptimizer: one joint grad norm and one clip coefficient for every sub-optimizer)
 [0] DistributedOptimizer : all DDP params (attention, router, norms, embeddings, shared experts)
 [1] MoonEPOptimizer      : fp32 masters ─► Adam(W) step
     ─► sync_compute_weights(): W_x[lo:hi] (bf16) <- masters (fp32); peers read them at their next prefetch
```

| Tensor | dtype / memory | Owner | Written by | Read by | Reduced by |
|---|---|---|---|---|---|
| `W_x` compute pools `[E+B,out,in]` | bf16, VMM composite. Rows `[lo,hi)` and `[E,E+B)` are local memory; all other rows are remote mappings | `MoonEPExpertStorage` | `sync_compute_weights` (rows `[lo,hi)`), `prefetch_weight` (rows `[E,E+B)`) | grouped GEMMs, peers' prefetch | – |
| masters `weight_{gate,up,down}` `[epn,out,in]` | fp32, ordinary CUDA tensors | `MoonEPGroupedMLP` (`nn.Parameter`, `requires_grad=False`) | `MoonEPOptimizer`, `load_state_dict` | sync, checkpoint | – |
| `G_x[lo:hi]` | fp32, VMM | storage; aliased as masters' `.grad`/`.main_grad` | local dW, `reduce_grad`, zeroing | optimizer, grad norm | `reduce_grad` (EP), finalize (1/\|dp_cp\|, expert-DP SUM) |
| `G_x[E:E+B)` = `RB_x[r]` | fp32, VMM | storage | dW of the copied experts, zeroing | peers' `reduce_grad` | cleared by `reduce_grad` |
| router weight | bf16 param + fp32 main param in DistOpt | `TopKRouter` | DistOpt | router GEMM | DDP (mean over dp_cp) |

---

## 1. MoonEP contract primer (what the library needs from its caller)

### 1.1 `moonep.Buffer`
- **Constructor**: `Buffer(S, H, K, E, num_ep_ranks, num_sms=None, token_padding=128, B=None, group=None, comm_stream_priority=-1, enable_pdl=True, explicitly_destroy=False)` (`MoonEP/moonep/api.py:472-486`).
  - `num_sms=None` becomes 32 (`api.py:272-273`).
  - `num_ep_ranks` must equal the group size (`api.py:277-281`), and `E % R == 0` (`api.py:289`).
  - `B` defaults to `E//R` (`api.py:292-293`).
  - `NvS = S·K + (token_padding-1)·2·(E/R)` (`api.py:296-307`).
- **Construction is collective over `group`.**
  - It allocates a symmetric bf16 `hidden_buf [NvS_padded,H]` and an int32 `meta_buf` with an NVSwitch multicast view (`api.py:380-383`).
  - It zeroes the barrier slots and calls `dist.barrier` (`api.py:385-389`).
  - `S` is a capacity; every call needs `1 ≤ s ≤ S` (`api.py:860-862`).
- **`destroy()` is collective**: comm-stream sync, `cuda.synchronize`, `dist.barrier(group)` (`api.py:568-610`). With `explicitly_destroy=True`, `__del__` only emits a `ResourceWarning` (`api.py:612-620`); the tensors are then freed without the barrier when the object dies.

### 1.2 Calls

| Call | Contract |
|---|---|
| `dispatch(x, w, topk, tpe)` (fresh) | **Inputs**: `x` `[s,H]` bf16 (dtype asserted at `MoonEP/moonep/dispatch.py:897`); `w` `[s,K]` fp32; `topk` `[s,K]` int32; `tpe` `[E]` int32, contiguous (`api.py:870-911`). A non-contiguous or non-16-byte-aligned `x`, `w` or `topk` is cloned first (`api.py:868-869, 882-885, 902-903`), so pass contiguous tensors to avoid a copy. **Steps**: inter-rank sync, planning, scatter kernel, dedup epilogue, copy-out (`api.py:662-706`). **Returns** fresh `h_nvs [nvs_s,H]`, `w_nvs [nvs_s]`, `cu_seqlens [E+B]` as padded **end** offsets (the `_grouped_mm` `offs` format), and `plan` (`api.py:830-848, 950-981`). |
| `dispatch(g, plan=plan)` (plan reuse = backward of combine) | Skips planning and the dedup builder (`api.py:684-691, 924-942`). Asserts `plan.num_tokens == s`. |
| `prefetch_weight(plan, full_*_weight)` | **One-sided NVLink read**: copies the peer-owned rows `W[experts_to_copy[rank,b]]` (in `[0,E)`) into this rank's slot `W[E+b]` for the three `[E+B,d1,d2]` tensors (`api.py:177-201, 1019-1066`). No barrier; slots marked `-1` are not written. It is correct only because a rank writes its own rows solely in `sync_compute_weights`, and the dispatch that produced `plan` starts with an inter-rank sync (`api.py:662-663`) that orders peers' earlier writes before this read (`my_moon_ep.py:370-376`). `d1,d2 % 128 == 0` (`MoonEP/moonep/prefetch.py:401-402`). |
| `combine(plan, h, w=None)` | Inter-rank sync, copy-in, fp32 pre-reduction of duplicate rows written back to the shard as bf16 (`MoonEP/moonep/combine_prologue.py:456-478`), then the combine kernel, which accumulates in fp32 and writes bf16 (`api.py:723-745`). `h` must be contiguous bf16 (`api.py:1167-1177`). Returns a fresh `[s,H]` bf16 tensor (`api.py:1204-1209`). **Does not apply routing weights.** The caller scales rows before combine (`MoonEP/my_moon_ep.py:442-446`). |
| `reduce_grad(plan, full_*_grad, *_reduce_buffer)` | Adds every rank's slot grads of this rank's experts into its own rows. After a barrier, each rank clears its own consumed slots (`api.py:1278-1285`; `MoonEP/moonep/grad_reduce.py:1-27, 368-389`). **No entry barrier**: "the caller must ensure all ranks have finished writing" (`grad_reduce.py:468-469`), so call `launch_inter_rank_sync(ctx)` first (`my_moon_ep.py:471-474`). |

### 1.3 Ordering rules
- Every barrier-bearing kernel uses the same three per-rank slots (`api.py:347-351`), and barriers are matched only by sequence.
- Barrier sites: planning ×3 (`MoonEP/moonep/planning.py:685,1118,1237`), dispatch exit (`dispatch.py:694`), combine entry (`MoonEP/moonep/combine.py:244`), inter-rank sync (`MoonEP/moonep/inter_rank_sync.py:73`), and one per `grad_reduce` launch (`grad_reduce.py:369`, three launches per `reduce_grad`, `api.py:218-238`).
- **Every EP rank must issue the same sequence of dispatch/combine/reduce calls.** A mismatch spins for about 100 s at 2 GHz and then traps the device (`MoonEP/moonep/_common.py:36-37, 337`).
- All calls on one Buffer must be serialized. Phase 1 always uses the synchronous mode, which runs on the current stream.

### 1.4 Plans and Buffer sharing
- Fresh dispatch allocates new plan tensors. Plan-reuse paths build no dedup map (`api.py:679-691`), and outputs are fresh copies when `zero_copy=False` (`api.py:950-965`).
- So **one Buffer can serve every layer and micro-batch**, as long as each (layer, micro-batch) keeps its own plan until its last use and all layers share `(S,H,K,E,B,token_padding,num_sms)`.
- Never use `zero_copy=True` in training: the view is overwritten by the next call (`api.py:807-814`).
- JIT compile caches are keyed by the Buffer capacity and shapes, not by the per-call `s` (for example `dispatch.py:711-720`, `grad_reduce.py:397-406`). Kernels compile on first use.

### 1.5 Training memory layout (from the reference, `MoonEP/my_moon_ep.py`)
- **Composite bf16 weight pool per projection per layer**: every rank's own-expert chunk followed by this rank's prefetch chunk, giving `[E+B, ...]` (`my_moon_ep.py:296-311`).
- **Composite fp32 grad `[E+B,...]`**: rows `[E,E+B)` are this rank's reduce chunk. They are the same memory as `rb[rank]` of the all-rank reduce view `[R,B,...]` (`my_moon_ep.py:313-338`).
- **Zeroed at construction**: the own rows and the own reduce chunk are zeroed in `__init__` (`my_moon_ep.py:207-210`), and `__init__` ends with `torch.cuda.synchronize(); dist.barrier(group)` (`my_moon_ep.py:238-239`). VMM memory from `cuMemCreate` is not documented to be zero-filled, and `reduce_grad` only clears slots it consumes, so this zeroing is part of the contract.
- `B == E/R` is required (`my_moon_ep.py:171`; `MoonEP/README.md:58`). Reasons:
  - equal chunk shapes for `nvl_dist_map`;
  - every remote expert gets a slot;
  - `offs_live` validity (the groups outside `[lo,hi)` and `[E,E+B)` are empty).
- A rank writes only its own rows and its slots (`my_moon_ep.py:207-210, 364-366, 380`). All other rows are live remote mappings.
- Chunk bytes must be a multiple of `get_vmm_granularity()` (`my_moon_ep.py:252-256`). This is **2 MiB on this VM** (verified).
- dW goes straight into the fp32 rows through `_GroupedMMLiveGrad`, with dW computed only over the live groups (`my_moon_ep.py:98-147, 433-440`).
- Masters' `.grad` aliases `mg[lo:hi]` (`my_moon_ep.py:223-228`). "Never `opt.zero_grad(set_to_none=True)`" (`my_moon_ep.py:153-157`).
- `sync_compute_weights()` must run after every step (`my_moon_ep.py:369-380`). `reduce_expert_grads()` runs once per micro-batch (`my_moon_ep.py:452-485`).

### 1.6 Hardware and failure modes
- **Multicast is mandatory** (`api.py:381`; assert at `MoonEP/moonep/buffer.py:303`). `get_multicast_granularity` goes through `CUCHECK`, which calls `exit()` (`MoonEP/csrc/nvl_shared_buffer.cuh:17-27, 394-405`).
- **Memory sharing** uses fabric handles only if **every** rank supports them (`buffer.py:43-61`). The comment at `buffer.py:29-31` wrongly says "when the group spans more than one node". Otherwise POSIX fds are passed over AF_UNIX sockets in a tempdir made by group rank 0 (`buffer.py:134-148`), which needs the same host and a shared `/tmp`. `MOONEP_MEM_HANDLE_TYPE=fd` returns before the all-gather that the other modes issue (`buffer.py:40-41` versus `buffer.py:49-51`), so the variable must be identical on all ranks.
- **MoonEP has no NVLink or P2P check.** A bad topology shows up as `exit()`, a hang in `dist.barrier`, or an `AssertionError`. That is why §5 M2 checks topology first.
- **Limits**:
  - K ≤ 32 and R ≤ 128 (`planning.py:1542-1551`);
  - H % 128 (`combine.py:612-614`);
  - both inner dims % 128 for prefetch and `grad_reduce` (`grad_reduce.py:497-498`).

### 1.7 Known defects in the reference (do not copy)
- `my_moon_ep.py:237` sets `self.last_plan`, but lines 403 and 463 use `self._last_plan`.
- `my_moon_ep.py:519` uses `moe.arr`, which does not exist.
- `my_moon_ep.py:544` returns `wg.grad` as `dwd`.
- `my_moon_ep.py:469-470` all-reduces the accumulated router grad over EP every micro-batch. This over-counts under gradient accumulation, and Megatron's router is DDP-managed anyway.
- `my_moon_ep.py:179` calls `torch.manual_seed(0)` inside `__init__`, which is a global RNG side effect.
- The comment at `my_moon_ep.py:424-426` says rows past `cu_seqlens[-1]` "come out as exact zeros". **That is false on B200** (verified): they are uninitialized, and NaN under allocator poisoning. It is harmless only because nothing reads those rows.

### 1.8 Verified on this VM (Appendix A)
- **MoonEP runs on sm_100 here.**
  - `my_moon_ep.py` (E=128, K=16, H=4096, Hi=8192, S=4096): 86.2 ms per train step; "bf16 compute copies == cast fp32 masters: True".
  - `moe_moon_ep.py` against its dense reference (the router rows of experts 0–63 are zeroed, so all tokens go to experts owned by ranks 4–7 and the prefetch and reduce paths run):
    - forward max abs diff 1.56e-2 (ref max ≈2.8);
    - dx ≤ 9.4e-2 (ref max ≈5);
    - dW ≤ 0.17 (ref max ≈40) on ranks 4–7;
    - ranks 0–3 got no tokens for their own experts, so their dW is exactly 0 in both;
    - router grad diff 0;
    - repeated identical steps are **bit-identical**.
- **`torch._grouped_mm` on B200, torch 2.14**:
  - (a) empty groups in the 2d×2d dW are exact zeros, checked 20× under NaN-poisoned allocations, in both layouts;
  - (b) `out_dtype=torch.float32` with bf16 inputs raises "Grouped gemm output dtype must match `mat_a` dtype";
  - (c) a transposed `mat_b` view (weights stored `[G,out,in]`) works in the forward, and dX/dW are correct, at the same speed as `[G,in,out]`;
  - (d) output rows past `offs[-1]` are not written;
  - (e) fp32 inputs are accepted.
- **Device properties**:
  - VMM granularity = multicast granularity = 2 MiB;
  - `nvl_multicast_supported()=True`, `nvl_fabric_supported()=False`, so MoonEP picks the fd path;
  - NVML NVLink P2P status is `P2P_STATUS_OK` (0) for all 56 ordered GPU pairs; the status enum is `OK=0, CHIPSET_NOT_SUPPORTED=1, GPU_NOT_SUPPORTED=2, IOH_TOPOLOGY_NOT_SUPPORTED=3, DISABLED_BY_REGKEY=4, NOT_SUPPORTED=5, UNKNOWN=6`;
  - every GPU's fabric info (`GpuFabricInfo_v3`): state 3 (COMPLETED), status 0, clique 0, **all-zero cluster UUID**, `health_summary` 1 (HEALTHY; enum `NOT_SUPPORTED=0, HEALTHY=1, UNHEALTHY=2, LIMITED_CAPACITY=3`);
  - NVML ignores `CUDA_VISIBLE_DEVICES` (it listed all 8 GPUs with only 2 visible), so map GPUs by PCI bus id;
  - the torch PCI fields formatted as `%08X:%02X:%02X.0` give NVML's bus id (for example `00000000:EA:00.0` for physical GPU 3 under `CUDA_VISIBLE_DEVICES=3`).

---

## 2. Megatron internals primer (touched subsystems)

### 2.1 MoE layer
- `BaseMoELayer` takes `pg_collection.ep`, sets `num_local_experts = E//ep_size`, and owns global experts `ep_rank·epn + i` (`megatron/core/transformer/moe/moe_layer.py:181-214`). This matches MoonEP's `lo = rank·epn`.
- `MoELayer.__init__` builds, in order:
  - the router (`moe_layer.py:273-281`);
  - the dispatcher, through an if-chain that raises on unknown types (`moe_layer.py:342-367`);
  - the experts from the spec (`moe_layer.py:369-375`);
  - the shared experts (`moe_layer.py:377-389`).
- **The dispatcher exists before the experts**, which is why §5 M5 needs a bind step.
- **The layer number arrives after construction.** `TransformerLayer` builds the MLP without `layer_number` and then calls `self.mlp.set_layer_number(self.layer_number)` (`megatron/core/transformer/transformer_layer.py:465-473`). `BaseMoELayer.set_layer_number` forwards it only to the router (`moe_layer.py:221-224`). Inside `MoELayer.__init__`, `layer_number` is therefore `None` on the normal path. `TransformerLayer.layer_number` is 1-based (`transformer_block.py:417`) and includes the pipeline offset for decoder layers (`transformer_layer.py:364-368`).
- The forward stages are shared experts → `route` → `preprocess` → `dispatch` → `routed_experts_compute` → `combine` → `postprocess` (`moe_layer.py:709-750`).
- `routed_experts_compute` calls `dispatch_postprocess`, then `experts(dispatched_input, tokens_per_expert, permuted_probs)`, which must return `(out, None)`, then `combine_preprocess` (`moe_layer.py:584-620`).
- The experts interface is defined at `moe_layer.py:69-98`.
- TP>1 in training requires SP (`moe_layer.py:690-694`).

### 2.2 Flex dispatcher and managers
- The base dispatcher takes `pg_collection.ep`, `expt_tp` and `tp_ep` (`token_dispatcher.py:84-92`). It is a plain class, not an `nn.Module` (`token_dispatcher.py:63`).
- `_DispatchManager` is an ABC with `setup_metadata`, `dispatch`, `combine`, `get_permuted_hidden_states_by_experts` and `get_restored_hidden_states_by_experts` (`token_dispatcher.py:963-998`). The DeepEP template is at `token_dispatcher.py:1225-1468`, with top-k on the dense probs at line 1300.
- `MoEFlexTokenDispatcher` picks the backend at `token_dispatcher.py:1835-1867`, and any other value raises (`token_dispatcher.py:1864-1867`).
- `dispatch_preprocess` is `@jit_fuser`, which means **torch.compile** (`megatron/core/jit.py:16-33`; `token_dispatcher.py:1921-1946`). It reshapes routing map and probs to `[T, tp·ep, epn]` (tp = expert-TP, which is 1 for MoonEP; `token_dispatcher.py:1893-1919`) and calls `setup_metadata`.
- `token_dispatch` calls `manager.dispatch(h, True, True)` with the async flags passed positionally, and returns `manager.dispatched_probs` (`token_dispatcher.py:1948-1981`).
- `dispatch_postprocess` returns the manager's permuted tensors and `get_number_of_tokens_per_expert()` (`token_dispatcher.py:1983-2000`).
- `token_combine` calls `manager.combine` (`token_dispatcher.py:2011-2033`). `combine_postprocess` calls `view(hidden_shape)` (`token_dispatcher.py:2035-2052`).
- With `moe_shared_expert_overlap`, the flex dispatcher itself runs the shared-expert fc1/fc2 on `SharedExpertMLP.stream` around `manager.dispatch`/`combine`, for any backend (`token_dispatcher.py:1972-1979, 2031-2032, 2048-2051`).

### 2.3 Experts and checkpoint layout
- `SequentialMLP` (the only grouped option here, since TE is not installed) builds one `MLP` per expert (`megatron/core/transformer/moe/experts.py:1640-1684`).
- Per expert, fc1 is `[2·Hi, H]` with **gate rows first** and fc2 is `[H, Hi]`, both `[out,in]` (`megatron/core/transformer/mlp.py:203-257, 340-347`). The GPU init path is `layers.py:1170-1191` plus `layers.py:225-237`, under the expert RNG tracker; fc1 is initialized before fc2 in each expert.
- Probs are applied **after the activation and before fc2**, computed in fp32 and cast back to bf16 (`mlp.py:368-371`; TEGroupedMLP at `experts.py:1125-1127`).
- **Checkpoint key.**
  - `SequentialMLP.sharded_state_dict` gives `'{prefix}experts.linear_fc1.weight'` with offsets `(layer, NL)`, `(expert, E)`, and the SwiGLU halves as fragments `(axis,0,2)` and `(axis,1,2)`. fc2 is similar. `replica_id[2]` is the expert-DP rank (`experts.py:1759-1800`; `mlp.py:391-413, 460-530`).
  - TEGroupedMLP uses the same keys ("interchangable with SequentialMLP's", `experts.py:1160-1202`).
  - The layer axis `(0,L,NL)` is added by `TransformerBlock` for homogeneous layers only (`megatron/core/transformer/transformer_block.py:866-890`).

### 2.4 Router
- `TopKRouter.weight` is created fp32 on the CPU, initialized with `init_method`, then cast to `params_dtype` (bf16) (`megatron/core/transformer/moe/router.py:74-99`).
- The logits are computed in `moe_router_dtype` (`router.py:104-126`).
- `routing()` builds `routing_map` by scattering **exactly K** indices per token (`megatron/core/transformer/moe/moe_utils.py:941-958`). Only capacity dropping changes that (`router.py:966-975`).
- `padding_mask` never changes the forward values of `probs` or `routing_map`. It only affects z-loss, the aux losses, the expert-bias token counts and the router diagnostics (`router.py:926-1039`); quantile routing rejects it (`router.py:947-950`).
- Aux losses attach to `probs` and backprop only if `probs` receives a gradient (`router.py:1016-1036`).

### 2.5 Config, CLI and init order
- `TransformerConfig.__post_init__` starts at `transformer_config.py:1603` and ends at `transformer_config.py:3814`.
  - The flex validations are at `transformer_config.py:2061-2091`. The deprecated `moe_enable_deepep` silently rewrites the backend to `'deepep'` unless it is `'hybridep'` (`transformer_config.py:2061-2070`).
  - The `recompute_modules` default is `['core_attn']` even when recompute is off (`transformer_config.py:2311-2312`).
  - `moe_layer_recompute` mutates the recompute settings (`transformer_config.py:2392-2403`).
  - The EP-overlap checks are at `transformer_config.py:3469-3562`.
- ETP defaults to TP (`megatron/core/model_parallel_config.py:520-521`).
- **CLI from config.** A `Literal` field becomes argparse choices automatically (`megatron/training/argument_utils.py:151-154`; factory at `arguments.py:2528`). `core_transformer_config_from_args` copies args by field name (`argument_utils.py:319-321`), and `--swiglu` sets `F.silu` plus GLU (`argument_utils.py:334-337`).
- **Why the collective NVLink check cannot live in `__post_init__`.** The config is built:
  - before `init_process_group` (`pretrain_gpt.py:579` versus `initialize.py:377`);
  - again on **every micro-batch** (`pretrain_gpt.py:103`);
  - inside `initialize_model_parallel` via `get_embedding_ranks` when PP>1 (`pretrain_gpt.py:540-543`);
  - in unit tests without any process group.
- The earliest point where args are final, the device is set and the EP groups exist is `finish_mpu_init`, right after `_initialize_distributed(...)` (`megatron/training/initialize.py:116-124`). Inside it:
  - `set_device` is at `initialize.py:307`;
  - `initialize_model_parallel` is at `initialize.py:395-419`;
  - seeding happens later, at `initialize.py:136-149`.
- **Model wrap order** (builder path) is `.cuda()` → `Float16Module` → DDP (`megatron/training/models/dist_utils.py:180-217`). The legacy path does the same (`training.py:2653-2663`). Freeze flags are applied before wrapping (`training.py:2621-2624`); `--freeze-base-model-for-mtp` calls `param.requires_grad_(True)` on every `mtp.layers.` parameter (`training.py:2493-2502`).

### 2.6 Process groups
- The expert grid is `RankGenerator(tp=ETP, ep=EP, dp=EDP, pp=PP, cp=1)` (`megatron/core/parallel_state.py:869-895`), and the EP groups are built at `parallel_state.py:1353-1365`. With ETP=1 each EP group is a contiguous rank block. EP peers always share a PP stage (`parallel_state.py:903-906`).
- The expert RNG seed is `seed+1024+100·ep_rank+etp_rank`, the same across expert-DP (`megatron/core/tensor_parallel/random.py:509`).
- `ProcessGroupCollection` has `ep`, `expt_tp`, `tp_ep`, `tp_ep_pp` and `expt_dp` (`megatron/core/process_groups_config.py:104-160`).

### 2.7 Float16Module and `nn.Module._apply`
- `Float16Module` runs `convert_module_to_dtype_except_fp32_marked`, which is `module._apply(t.to(dtype) unless t.keep_in_fp32)` (`megatron/core/transformer/module.py:476-548`). It is **not** `.bfloat16()`.
- torch `_apply` recurses into children first (`torch/nn/modules/module.py:931-933`), then swaps `param.data`. For `param.grad` it separately calls `fn(param.grad)` and assigns `grad.data = ...` (`torch/nn/modules/module.py:959-1033`).
- So a master marked `keep_in_fp32` still gets its **`.grad` converted to a new bf16 tensor, breaking the alias**, unless the grad tensor is marked as well. Buffers are replaced through `self._buffers[k] = fn(buf)` (`torch/nn/modules/module.py:1034-1036`).

### 2.8 DDP
- DDP registers every `named_parameter` with `requires_grad` (`megatron/core/distributed/distributed_data_parallel.py:185-195`) and hooks only those (`distributed_data_parallel.py:466-502`). It keeps its groups as attributes (`ddp_config` at `:138`, `dp_cp_group` at `:147`, `expt_dp_group` at `:149`).
- Buffers are keyed by `(dtype, grad dtype, is_expert = not allreduce, ...)` (`megatron/core/distributed/param_and_grad_buffer.py:966-980`).
- With the distributed optimizer, `param.data` is re-pointed into DDP's buffer (`param_and_grad_buffer.py:1367-1371`), and `param.main_grad` is always a buffer view (`param_and_grad_buffer.py:1429-1431`).
- **The hook** adds `.grad` into `main_grad` and then sets `param.grad = None` (`distributed_data_parallel.py:588-612`).
- **Scaling.** Dense and expert grads are both pre-scaled by 1/|dp_cp| and SUM-reduced. With `average_in_collective`, experts get `edp/dp_cp` and AVG. With per-token loss the factor is 1.0, and per-token loss with `average_in_collective` is rejected (`distributed_data_parallel.py:249-296, 307-313`; `param_and_grad_buffer.py:712-720`).
- **Grad checks.** Before the collective, `check_grads` validates each bucket's grad norm for NaN and Inf (fatal) and for unexpectedly large values (non-fatal, `context="grads"`) through the rerun state machine (`param_and_grad_buffer.py:389-425, 703-707`). In training, `check_for_nan_in_grad` comes from `--check-for-nan-in-loss-and-grad`, which defaults on (`training.py:2850-2851`).
- **If MoonEP masters were registered** they would never get an autograd grad. With `overlap_grad_reduce`, a bucket group that mixes them with autograd-updated params fails an assertion in `reset()` at the second iteration (`param_and_grad_buffer.py:336-340`); a group of only such params never asserts. Without the assertion, the optimizer silently steps on an all-zero `main_grad`.
- `zero_grad_buffer` zeros DDP buffers only (`distributed_data_parallel.py:718-732`).

### 2.9 `finalize_model_grads`
- It runs once per step, after the last backward (`megatron/core/pipeline_parallel/schedules.py:864-872`). `train()` keeps a `finalize_model_grads_func` that the caller already installed (`training.py:4813-4814`).
- Order:
  - DDP `finish_grad_sync` (`finalize_model_grads.py:685-686`);
  - TP all-reduce of SP parameters (`finalize_model_grads.py:707`);
  - embeddings and router expert-bias updates;
  - per-token scaling of the **DDP buffers only** (`finalize_model_grads.py:743-759`).
- `dp_cp_group` is picked at `finalize_model_grads.py:649` (`pg_collection` case) or `finalize_model_grads.py:657` (fallback).
- The rerun state machine asserts that `validate_result` is called inside forward-backward when reruns are enabled (`megatron/core/rerun_state_machine.py:542-544`); finalize satisfies that.

### 2.10 Optimizer
- `get_megatron_optimizer` is at `megatron/core/optimizer/__init__.py:1030-1312`. Along the way:
  - `is_mfsdp_v2` is computed at `optimizer/__init__.py:1081`;
  - emerging optimizers return early (`optimizer/__init__.py:1084-1093`);
  - process groups are set up (`optimizer/__init__.py:1114-1126`);
  - dense optimizers are built (`optimizer/__init__.py:1226-1265`);
  - the expert optimizer is built (`optimizer/__init__.py:1267-1304`);
  - it returns `ChainedOptimizer(optimizers)` (`optimizer/__init__.py:1312`).
- `_get_param_groups` skips `not requires_grad` (`optimizer/__init__.py:339`), applies `config_overrides` by `ParamKey` match (`optimizer/__init__.py:344-349`; MuP overrides scale LR and eps of every matrix-like parameter, `optimizer/__init__.py:219-300`), keys experts by `allreduce` (`optimizer/__init__.py:357`), and aligns the group keys across ranks with `all_gather_object` (`optimizer/__init__.py:371-377`).
- `skip_megatron_wrapping=True` returns the tuple `(optimizer, init_state_fn)` with the torch optimizer unwrapped (`optimizer/__init__.py:645-646`). With the torch fallback, Adam is `torch.optim.AdamW` when `decoupled_weight_decay` (default) is set (`optimizer/__init__.py:569`). For SGD, `init_state_fn` is `None` (`optimizer/__init__.py:631-638`).
- **`FP32Optimizer`** (`megatron/core/optimizer/optimizer.py:1313-1459`):
  - steps the parameters in place;
  - `prepare_grads` rebinds `param.grad = param.main_grad` when `main_grad` exists (`optimizer.py:1356-1359`);
  - `zero_grad` defaults to `set_to_none=True` (`optimizer.py:1333-1338` → `optimizer.py:59-77`);
  - `sharded_state_dict(is_loading=True)` calls `self.init_state_fn(...)` unconditionally (`optimizer.py:1436-1443`).
- **`ChainedOptimizer`**:
  - asserts every sub-optimizer shares the same config (`optimizer.py:1521`);
  - forwards `set_to_none` positionally (`optimizer.py:1565-1567`);
  - takes the loss scale from the first sub-optimizer (`optimizer.py:1569-1573`);
  - computes a global grad norm, as sqrt of summed squares when grad-stat groups differ (`optimizer.py:1869-1885`);
  - clips every sub-optimizer with that norm (`optimizer.py:2013-2030`).
- Megatron's own fp32 master of a bf16 parameter is `param.detach().clone().float()` (`optimizer.py:1061`).

### 2.11 `train_step` and schedules
- In `train_step`, inside the rerun loop (`megatron/training/training.py:3314`):
  - `zero_grad_buffer()` (`training.py:3316-3317`), then `optimizer.zero_grad()` (`training.py:3320`);
  - then `forward_backward_func` (`training.py:3395-3408`);
  - then `optimizer.step()` (`training.py:3474`).
- The rerun decision is all-reduced over the world (`megatron/core/rerun_state_machine.py:254-270, 329`), so all ranks rerun together.
- **Schedules.**
  - PP=1 alternates forward and backward strictly per micro-batch (`schedules.py:815-862`).
  - The loss is divided by tokens and by micro-batch count (`schedules.py:336-341`).
  - EP overlap switches to combined 1F1B, where the forward of micro-batch i+1 is interleaved with the backward of micro-batch i on two streams (`schedules.py:779-795`; `combined_1f1b.py:62-67`; `model_chunk_schedule_plan.py:575-586`).
  - 1F1B warm-up runs several forwards before the first backward (`schedules.py:2292-2294`), and VPP does too (`schedules.py:965-971`).
- `evaluate()` runs `model.eval()` under `torch.no_grad()` (`training.py:5618, 5677`).
- `--freeze-all-layers` runs the forward under `torch.no_grad()` and marks the step forward-only (`training.py:2535-2546`).
- `setup_logging` can raise the root logger level with `--logging-level` or `MEGATRON_LOGGING_LEVEL` (`initialize.py:657-678`), which hides `logger.warning`.

### 2.12 Distributed checkpointing
- Optimizer state is matched to model `ShardedTensor`s by `id(ten.data)` (`megatron/core/dist_checkpointing/optimizer.py:45-80`); an unmatched parameter raises (`dist_checkpointing/optimizer.py:145`).
- `ShardedTensor.from_rank_offsets` semantics are defined at `megatron/core/dist_checkpointing/mapping.py:154-209`.
- The model is loaded with `module.load_state_dict` (`megatron/training/checkpointing.py:3049-3091`). If a strict load raises, it retries with `strict=False` and only prints `load_return` (`checkpointing.py:3065-3071`), so missing keys can pass silently. torch copies parameters in place (`param.copy_(input_param)` in `_load_from_state_dict`, `torch/nn/modules/module.py:2350, 2500`) and runs load post-hooks per submodule (`torch/nn/modules/module.py:2589-2621`).
- Without an optimizer load, bf16 runs call `optimizer.reload_model_params()` (`checkpointing.py:3218-3226`).
- The default format is `torch_dist` (`megatron/training/config/training_config.py:486`). The planner validates shapes only (`megatron/core/dist_checkpointing/strategies/torch.py:514-529`).
- `ChainedOptimizer.sharded_state_dict` prefixes sub-optimizer keys with `chained_{idx}.` unless the DistOpt format is fully reshardable (`optimizer.py:1650-1680`; metadata at `checkpointing.py:576-586`), so chain positions must stay stable.
- DCP converts dtype on load in both directions (verified).

### 2.13 Shutdown paths and inference entry points
- `parallel_state.destroy_model_parallel()` (`parallel_state.py:2505-2516`) has one production caller: `destroy_global_state` (`training.py:714-719`), which only in-process restart invokes, from a 10 s `ThreadedFinalize` on the fault path (`megatron/training/inprocess_restart.py:27-29, 70`). A normal `pretrain` run never calls it.
- The normal end of `pretrain` deregisters the GTP symmetric-memory pools before exit (`training.py:2268-2274`); the exit-condition branch of `train()` does the same after a world barrier (`training.py:5544-5555`). Both are the template for a collective MoonEP teardown.
- `Utils.initialize_model_parallel` in the test harness calls `ps.destroy_model_parallel()` before initializing (`tests/unit_tests/test_utilities.py:212-213`), and `Utils.destroy_model_parallel` synchronizes and barriers first (`test_utilities.py:184-199`).
- Megatron inference engines set a process-wide `InferenceMode` flag that modules should read (`megatron/core/inference/utils.py:49-63`); `MoELayer` already branches on it (`moe_layer.py:699`).

---

## 3. Design decisions (ADRs)

**ADR-A: Backend placement.**
- *Decision*: add `'moonep'` to `moe_flex_dispatcher_backend` with a `_MoonEPManager`.
- *Alternatives*:
  - a new `moe_token_dispatcher_type` (a cleaner class, but it bypasses the flex extension point);
  - a new MoE layer class (duplicates the stage split that other paths consume).
- *Why*: it matches deepep/hybridep/ncclep (`token_dispatcher.py:1835-1867`), the CLI value appears automatically (`argument_utils.py:151-154`), and the `MoELayer` stage split is reused unchanged.

**ADR-B: Experts selection.** `MoELayer.__init__` builds `MoonEPGroupedMLP` when the backend is moonep, because the spec helpers never see the config (`megatron/core/models/gpt/moe_module_specs.py:57-90`). It then binds the experts' storage into the manager, and `BaseMoELayer.set_layer_number` forwards the layer number to the experts.

**ADR-C: Megatron-native weight layout.**
- Pools, grads and masters are stored `[out,in]` per projection: gate and up `[·,Hi,H]`, down `[·,H,Hi]`. The GEMMs consume transposed views.
- *Why*:
  - checkpoints become key- and shape-identical to `SequentialMLP`/TEGroupedMLP (§2.3), with fc1 = gate ‖ up expressed as two `ShardedTensor`s, like the SwiGLU factory (`mlp.py:460-530`);
  - no converter or factory is needed, and the optimizer-state id mapping keeps working (§2.12);
  - the transposed-operand GEMMs work on sm_100 at equal speed (verified §1.8);
  - MoonEP's prefetch and reduce kernels do not care what the two inner dims mean (`prefetch.py:395-402`; `grad_reduce.py:487-498`).

**ADR-D: Masters are `requires_grad=False` parameters.**
- *Alternatives*:
  - registered with `requires_grad=True` plus explicit skips in DDP (`distributed_data_parallel.py:185-195, 466-467`), in the DistOpt layout (`training.py:2430`; `dist_utils.py:402-404`) and in param groups (`optimizer/__init__.py:357-418`), which is about six coupled edits;
  - unregistered "hidden" masters, which break `load_state_dict` (`checkpointing.py:3065`).
- *Why*:
  - autograd truly never computes these grads, so the flag is honest;
  - DDP, the layout and `_get_param_groups` all skip such parameters (`distributed_data_parallel.py:186`; `training.py:2430`; `dist_utils.py:402-404`; `optimizer/__init__.py:339`);
  - checkpoint loading copies in place (`torch/nn/modules/module.py:2500`) and runs post-hooks per submodule (`torch/nn/modules/module.py:2589-2621`);
  - torch AdamW steps a `requires_grad=False` parameter that has `.grad` set, and the alias survives (verified).
- *Consequence*: `requires_grad` no longer says whether the experts train. Phase 1 forbids the freeze flags that toggle it (§6), and `check_invariants()` asserts it stays `False`.

**ADR-E: Where probs are applied.**
- *Decision*: before fc2, as `(act.float()·w).bf16()`, like Megatron (`mlp.py:368-371`). MoonEP's reference applies them after fc2.
- The two are mathematically equal with no fc2 bias; only the rounding point differs. Matching the baseline tightens the parity tests.
- `nan_to_num(w_nvs)` is still required, because padding and tail slots hold stale weights (`my_moon_ep.py:442-446`).

### Hypotheses H1–H16

| # | Outcome | Resolution (implemented in) |
|---|---|---|
| H1 | **Confirmed, refined** | Megatron casts through `convert_module_to_dtype_except_fp32_marked` (§2.7), not `.bfloat16()`, and `_apply` converts `.grad` separately, so marking the parameter alone does not help. `MoonEPGroupedMLP._apply` ignores dtype conversions and refuses device moves (verified on CPU: masters stay fp32, the alias survives, sibling modules are converted, `.to('meta')` raises, `load_state_dict` still copies in place). Masters and grad views are also marked `keep_in_fp32`; pools are plain attributes, never buffers; invariants are checked after wrapping (M4, M7). |
| H2 | **Confirmed; failure mode corrected** | Registered masters would not hang: see §2.8. DistOpt would also re-point `.data`, and DDP would allocate a duplicate `main_grad`. Exclusion comes from `requires_grad=False` (ADR-D), with zero DDP edits. |
| H3 | **Confirmed, modified** | Masters never reach DistOpt or `Float16Optimizer` because `_get_param_groups` skips them. `MoonEPOptimizer(FP32Optimizer)` is appended **last**, so the loss scale still comes from the first sub-optimizer and the `chained_{idx}` checkpoint prefixes stay stable. Its grad-stats group is `expt_tp_pp_with_egtp_remat_group`; `ChainedOptimizer` combines the per-group norms (non-shared path, `optimizer.py:1879-1884`) and applies one clip coefficient (`optimizer/clip_grads.py:183`) (M7). |
| H4 | **Confirmed** | Only `MoonEPOptimizer` holds the masters, and its `zero_grad` never touches grads. Remaining hazard: torch `nn.Module.zero_grad` walks `self.parameters()` (`torch/nn/modules/module.py:2957-2984`) and is used by the test harness (`tests/unit_tests/determinism/utils.py:113`); Megatron's training path never calls it. Mitigation: `repair_grad_aliases_` in the forward and in `prepare_grads` warns and re-binds (M4, M7). |
| H5 | **Confirmed** | The router has no `allreduce` attribute, so it lives in the dense DDP buffer and is averaged over dp_cp (`router.py:74-99`; `distributed_data_parallel.py:287-296, 307-313`); with SP it is also summed over TP (`finalize_model_grads.py:707`). Together these cover every rank that holds a router replica, including all EP peers. The reference's extra EP SUM would over-count, and would crash because the DDP hook sets `.grad = None` (`distributed_data_parallel.py:607`). **Do not port it.** Use `--moe-router-dtype fp32` for fp32 gating. |
| H6 | **Confirmed, exact** | After `reduce_grad`, the owner rows equal what the alltoall backward would have computed on the owner. Then `×1/|dp_cp|` (DDP's own `dp_cp_group`) and SUM over expert-DP when it is >1 equal both DDP modes, `average_in_collective` included (`distributed_data_parallel.py:261-296`). With per-token loss the factor is 1.0, then `×1/clamp(num_tokens,1)` after finalize reduces `num_tokens` (`finalize_model_grads.py:743-759`). The NaN, Inf and large-grad checks mirror DDP's `check_grads` (`param_and_grad_buffer.py:389-425`) (M6). |
| H7 | **Confirmed** | Phase 1 asserts PP=1, no VPP, no EP overlap, no MoE recompute, `mlp_chunks_for_training=1` and no repeated MTP layer. A runtime guard raises if a second grad-tracking forward starts before the previous backward (M5). Phase 2 re-prefetches in combine-backward (§7). |
| H8 | **Confirmed, adopted** | The three `GroupedMMLiveGrad.backward` nodes produce `grad_h`/`grad_w`, so they are autograd ancestors of the dispatch-backward. All of this layer's dW `add_`s are therefore enqueued earlier on the same stream (backward ops run on their forward stream). Peers are fenced by `launch_inter_rank_sync`. Using `ctx.plan` makes the step schedule-agnostic (M5). |
| H9 | **Decided** | **One process-global Buffer** (§1.4). Its memory is ≈ `NvS_padded·H·2` bytes. Example: S=8192, K=8, H=7168, E/R=32 gives NvS=73,664 ≈ 0.98 GiB per Buffer, so **≈60 GiB for 61 per-layer Buffers**. Expert pools are **per layer** in phase 1 (the verified reference layout): per layer and rank that is 72·P bytes versus a baseline of 54·P (P = epn·H·Hi; bf16 own 2 + prefetch 2 + fp32 grad 4 + reduce 4 + master 4 + Adam 8, ×3 projections, versus 18 bytes per element for Megatron). Shared prefetch/reduce pools are phase 2. |
| H10 | **Resolved by ADR-C** | Keys and shapes are identical to the baseline. fp32 masters load bf16 checkpoints exactly, and the reverse direction rounds (DCP `copy_`, verified). Optimizer states are **not** compatible across backends: DistOpt uses its own sharded formats while `FP32Optimizer` uses per-parameter `ShardedTensor`s (M9). |
| H11 | **Confirmed** | Init happens per local expert, fc1 then fc2, like `SequentialMLP`: a `params_dtype` tensor `[2Hi,H]` gets `config.init_method` and a `[H,Hi]` tensor gets `output_layer_init_method`, both inside `get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name())`, then they are **upcast** into the masters. This is exactly Megatron's starting point (`optimizer.py:1061`). Sync runs after init, after load (post-hook), after every step, and in `reload_model_params` (M4, M7). |
| H12 | **Decided** | New field `moe_moonep_max_tokens_per_rank`, derived in `validate_args` from seq, micro-batch, eval micro-batch, CP and SP, plus a runtime assert (M1, M5). |
| H13 | **Confirmed, extended** | §6 has the full list. |
| H14 | **Confirmed, detailed** | Static checks in `__post_init__`; the collective check in `finish_mpu_init` with WORLD consensus; a cached EP-local re-run in `_MoonEPManager.__init__`. Same-host means equal `boot_id`. Fabric info matters only for cross-host pairs. `can_device_access_peer` is **not** used; it is also true for PCIe P2P (standard CUDA semantics) and says nothing about multicast (M2). |
| H15 | **Confirmed, refined** | Per-layer state CLEAN / ACCUMULATING / STALE plus a `finalized` flag; the initial state is CLEAN because storage is zeroed at construction. The STALE check fires on every **gradient-producing** forward (grad enabled and the dispatched input requires grad), not on `self.training`, so an eval-mode forward with grad enabled is covered and `torch.no_grad()` eval never is. Every occurrence is reported (M4, M8). |
| H16 | **Confirmed** | `MegatronModule.zero_moe_grads()`. DDP (`megatron/core/distributed/data_parallel_base.py:11`) and `Float16Module` are `MegatronModule`s, so the call works on wrapped chunks. It is called in `train_step` after `optimizer.zero_grad()`, inside the rerun loop (M8). |

---

## 4. Requirements traceability

| Requirement | Mechanism | Files / functions | Test that proves it |
|---|---|---|---|
| R1 seamless integration | flex backend, manager, experts, process-global Buffer, CLI flags, explicit teardown | `transformer_config.py`, `token_dispatcher.py`, `moe_layer.py`, `moonep_backend/*`, `training.py` | `test_moonep_layer.py::test_parity_vs_alltoall`, `test_moonep_training.py::test_loss_parity`, functional smoke and resume |
| R2 fp32 master + bf16 compute | fp32 masters (ADR-D), bf16 VMM pools, `_apply` guard, `MoonEPOptimizer` with sync after step/load/init | `experts.py`, `storage.py`, `moonep_optimizer.py` | `test_moonep_layer.py::test_mixed_precision_invariants` (dtype and `data_ptr` after `Float16Module`/`.to()`); `test_moonep_training.py::test_pools_equal_masters_after_step` (bitwise) |
| R3 NVLink assert at config/init | static `_validate_moonep` in `__post_init__`; collective `assert_moonep_ep_group_nvlink_coverage` in `finish_mpu_init` (WORLD consensus) plus the manager's cached re-check | `transformer_config.py`, `topology.py`, `initialize.py`, `manager.py` | `test_moonep_config.py`; `test_moonep_topology.py` (passes on 8×B200 for EP=8/4; synthetic and injected failures raise the identical message on all ranks) |
| R4 `zero_grad` never deletes | only `MoonEPOptimizer` holds the masters; its `zero_grad` is a marker; alias detection and re-binding | `moonep_optimizer.py`, `experts.py` | `test_moonep_training.py::test_optimizer_zero_grad_keeps_alias` |
| R5 model zero method + always notify | `MegatronModule.zero_moe_grads`, state machine, `logger.warning` per occurrence with stderr fallback (strict mode raises) | `module.py`, `experts.py`, `training.py` | `test_moonep_training.py::test_notify_every_time`, `::test_no_false_positive_accumulation`, `::test_strict_raises`, `::test_notify_when_warnings_hidden` |
| R6 plan | this document | – | – |

**Interpretation of R3 (please confirm).** The collective NVLink check needs process groups, and `TransformerConfig` is built before `init_process_group`, per micro-batch and in unit tests (§2.5). So R3 is met in two halves: static invariants fail at config construction, and the collective check fails at the first point of initialization where the EP group exists (before seeding, dataset and model build). Library users who build process groups themselves get the same check, EP-local, when the first MoonEP layer is constructed.

---

## 5. Milestones

Conventions for this section:
- Run everything from `/home/ubuntu/Megatron-LM`.
- `PY=/home/ubuntu/MoonEP/.venv/bin/python`.
- `DIST="$PY -m torch.distributed.run --nproc-per-node 8"`.
- Wrap every multi-GPU command in `timeout`: a MoonEP barrier mismatch hangs instead of raising.
- After each change that edits imports, run isort (CLAUDE.md). `uv` is not on PATH here, so use `$PY -m pip install isort==5.13.2` (the `pyproject.toml:187` pin), then `$PY -m isort <files>`.
- Never import `moonep` at module level in Megatron code; import it inside the functions that need it, so Megatron still imports without MoonEP.

### 5.0 Module map and API inventory

New files:

```
megatron/core/transformer/moe/moonep_backend/
  __init__.py   uses_moonep(config), iter_moonep_experts(module)                     (M1/M4)
  topology.py   NVLink-coverage check                                                (M2)
  buffer.py     process-global moonep.Buffer                                        (M3)
  storage.py    MoonEPExpertStorage: per-layer VMM pools                            (M4)
  experts.py    MoonEPGroupedMLP, grad-state machine, _notify, message texts        (M4)
  autograd.py   MoonEPDispatch, MoonEPCombine, GroupedMMLiveGrad                    (M5)
  manager.py    _MoonEPManager                                                      (M5)
  grads.py      reduce_moonep_expert_grads, scale_moonep_expert_grads               (M6)
megatron/core/optimizer/moonep_optimizer.py   MoonEPOptimizer                       (M7)
```

Edited files: `transformer_config.py` and `arguments.py` (M1), `initialize.py` (M2), `parallel_state.py` and `training.py` teardown (M3), `token_dispatcher.py` and `moe_layer.py` (M5), `finalize_model_grads.py` (M6), `optimizer/__init__.py` (M7), `module.py` and `training.py` `train_step` (M8).

| Symbol | Signature | Semantics | Milestone |
|---|---|---|---|
| `uses_moonep` | `(config) -> bool` | `config.moe_token_dispatcher_type == "flex" and config.moe_flex_dispatcher_backend == "moonep"` | M1 |
| `iter_moonep_experts` | `(module) -> Iterator[MoonEPGroupedMLP]` | modules with `_is_moonep_experts` | M4 |
| `TransformerConfig._validate_moonep` | `(self) -> None` | static rules of §6; pure | M1 |
| `assert_moonep_ep_group_nvlink_coverage` | `(ep_group, *, consensus_group=None, force=False) -> None` | collective; cached per EP rank tuple | M2 |
| `evaluate_moonep_nvlink_records` | `(records, p2p_findings, uses_fabric, tmpdir_visible) -> list[str]` | pure; problem lines | M2 |
| `moonep_handle_path` | `(records) -> tuple[bool, list[str]]` | pure mirror of `buffer.py:35-61` | M2 |
| `fabric_health_warnings` | `(records) -> list[str]` | pure; cross-host `LIMITED_CAPACITY` notices | M2 |
| `_reset_moonep_topology_cache` | `() -> None` | clears the cache; call on all ranks | M2 |
| `get_moonep_buffer` | `(group, *, max_tokens_per_rank, hidden_size, topk, num_experts, num_sms, token_padding) -> moonep.Buffer` | collective on first call | M3 |
| `destroy_moonep_buffer` | `() -> None` | collective `Buffer.destroy()` | M3 |
| `release_moonep_buffer` | `() -> None` | drops the reference; never blocks | M3 |
| `MoonEPExpertStorage` | `(ep_group, num_experts, hidden_size, ffn_hidden_size)` | attrs `w, g, rb, lo, hi, E, B, R, rank, epn, layer_number, fwd_outstanding`; methods `fp32_targets(n)`, `zero_grads_()` | M4 |
| `MoonEPGroupedMLP` | `(num_local_experts, config, pg_collection=None, name=None)` | attrs `weight_{gate,up,down}, storage, ep_group, expt_dp_group, layer_number, grad_state, finalized, _is_moonep_experts=True` | M4 |
| `  .forward` | `(h_nvs, cu_seqlens, w_nvs) -> (y, None)` | grouped GEMMs; grad-state check | M5 |
| `  .set_layer_number` | `(n) -> None` | sets `self.layer_number` and `storage.layer_number` | M4 |
| `  .sync_compute_weights` | `() -> None` | `w[n][lo:hi].copy_(master)` | M4 |
| `  .moonep_master_parameters` / `.owned_grads` | `() -> list` | the three masters / `[(name, grad view)]` | M4 |
| `  .check_invariants` / `.raise_on_broken_invariants` | `() -> list[str]` / `(where) -> None` | dtype, `data_ptr`, alias identity, `requires_grad` | M4 |
| `  .repair_grad_aliases_` | `(where) -> None` | notify and re-bind broken aliases | M4 |
| `  .zero_moonep_grads_` / `.mark_zero_point_` / `.mark_grads_consumed_` / `._on_grad_forward_` | `() -> None` | grad-state transitions | M4 |
| `  .sharded_state_dict` | `(prefix="", sharded_offsets=(), metadata=None)` | Megatron-native keys | M9 |
| `  ._apply` | `(fn, recurse=True)` | ignores dtype changes, refuses device moves | M4 |
| `MoonEPDispatch` / `MoonEPCombine` / `GroupedMMLiveGrad` | `torch.autograd.Function`s | see M5 | M5 |
| `_MoonEPManager` | `(group, num_local_experts, router_topk, num_experts, config)`; `bind_storage(storage)` plus the `_DispatchManager` methods | see M5 | M5 |
| `reduce_moonep_expert_grads` / `scale_moonep_expert_grads` | `(model, config)` / `(model, scaling)` | finalize steps; set `finalized` | M6 |
| `MoonEPOptimizer` | `(optimizer, config, init_state_fn, moonep_experts)` | see M7 | M7 |
| `_assert_moonep_optimizer_supported` | `(config, model_chunks, is_mfsdp_v2) -> None` | optimizer-level rules | M7 |
| `MegatronModule.zero_moe_grads` | `(self) -> int` | zero all MoonEP layers in place | M8 |

### M0: Environment and ground truth (no code)
- **Goal**: a runnable test environment and confirmation that MoonEP works here.
- **Steps**:
  - `$PY -m pip install "pytest==9.1.1" pytest-mock click requests`. The conftest imports `tests/test_utils/python_scripts/download_unit_tests_dataset.py` (`tests/unit_tests/conftest.py:13`), which imports `click` and `requests`; the pytest pin is `pyproject.toml:151`. The repository's canonical path is the CI container (see `skills/mcore-build-and-dependency`); on this host venv, install only these test-time packages directly.
  - `cd /home/ubuntu/MoonEP && timeout 1800 $DIST moe_moon_ep.py`. Expect `step replay dw_gate diff 0.000e+00` and dW differences around 1e-1 against a reference max around 40.
  - Probes: Appendix A.
- **Pitfalls**:
  - `tests/unit_tests/conftest.py:107-127` tries to download test data into `/opt/data`; it warns and continues.
  - `uv.lock` is not regenerated. Regenerate it with `uv lock` inside the container before any PR.

### M1: Config surface and static validation
- **Goal**: a MoonEP config builds only in the phase-1 envelope, with no GPU and no collectives.
- **Files**:
  - `megatron/core/transformer/transformer_config.py`: add `'moonep'` to the Literal at line 988; add three fields next to lines 988-992; insert the `moe_enable_deepep` check before line 2061; add a `_validate_moonep()` call at the end of `__post_init__`, after line 3814.
  - `megatron/core/transformer/moe/moonep_backend/__init__.py`: `uses_moonep(config)`.
  - `megatron/training/arguments.py`: S derivation and training-only checks after line 1687.

```python
# transformer_config.py (sketch)
moe_flex_dispatcher_backend: Literal['deepep', 'hybridep', 'ncclep', 'moonep'] = "deepep"
moe_moonep_max_tokens_per_rank: Optional[int] = None   # MoonEP Buffer capacity S (tokens/rank/micro-batch)
moe_moonep_token_padding: int = 128                    # moonep.Buffer(token_padding=...)
moe_moonep_strict_zero_grad: bool = False              # raise instead of warn on MoonEP grad-lifecycle notices

# before line 2061: the deprecated flag would silently turn 'moonep' into 'deepep' (lines 2061-2070)
if self.moe_enable_deepep and self.moe_flex_dispatcher_backend == "moonep":
    raise ValueError("moe_enable_deepep forces the 'deepep' flex backend; remove --moe-enable-deepep "
                     "when using --moe-flex-dispatcher-backend moonep.")

def _validate_moonep(self) -> None:
    if self.moe_flex_dispatcher_backend != "moonep":
        return
    def req(ok: bool, msg: str) -> None:
        if not ok:
            raise ValueError(f"moe_flex_dispatcher_backend='moonep' {msg}")
    req(self.moe_token_dispatcher_type == "flex",
        f"requires moe_token_dispatcher_type='flex', got '{self.moe_token_dispatcher_type}'.")
    # ... one req(...) per row of the table in §6
```

- **Invariants and messages**: every row of the §6 table, with its exact text. Always prefix `moe_flex_dispatcher_backend='moonep' `. Example: `requires pipeline_model_parallel_size == 1 (phase 1 keeps one prefetch/reduce slot set per layer; a second forward of a layer before its backward silently corrupts gradients), got 2.`
- **`validate_args`** (training):

```python
if args.moe_token_dispatcher_type == "flex" and args.moe_flex_dispatcher_backend == "moonep":
    if args.moe_moonep_max_tokens_per_rank is None:
        assert args.seq_length is not None, "MoonEP needs --seq-length to size its token buffer, or pass --moe-moonep-max-tokens-per-rank."
        tokens = args.seq_length * max(args.micro_batch_size, args.eval_micro_batch_size)  # set at arguments.py:736-737
        tokens //= args.context_parallel_size
        if args.sequence_parallel:
            tokens //= args.tensor_model_parallel_size
        args.moe_moonep_max_tokens_per_rank = tokens
    assert args.ckpt_format == "torch_dist", "MoonEP (phase 1) supports only --ckpt-format torch_dist."
    assert not args.data_parallel_random_init, "MoonEP does not support --data-parallel-random-init (it would bypass the bf16 pool sync)."
    assert not (args.use_megatron_fsdp or args.use_torch_fsdp2), "MoonEP requires Megatron DDP (no FSDP)."
    assert not args.freeze_all_layers, ("MoonEP (phase 1) does not support --freeze-all-layers: expert masters are "
        "requires_grad=False by design and would still be stepped (weight decay) by MoonEPOptimizer.")
    assert not args.inprocess_restart, ("MoonEP (phase 1) does not support --inprocess-restart: a MoonEP device trap "
        "poisons the CUDA context, and Buffer teardown is collective.")
    assert not (args.overlap_grad_reduce or args.overlap_param_gather), ("MoonEP (phase 1) requires "
        "--overlap-grad-reduce and --overlap-param-gather off: concurrent NCCL kernels next to MoonEP's cooperative "
        "cross-rank barrier kernels are untested (MOONEP.md §9 item 4).")
```

- **Pitfalls**:
  - Put `_validate_moonep` **last**, so it sees the resolved ETP (`model_parallel_config.py:520-521`), the resolved recompute settings (`transformer_config.py:2392-2403`), the normalized capacity factor (`transformer_config.py:2194-2196`) and `moe_ffn_hidden_size` (`transformer_config.py:2003-2004`).
  - The config is rebuilt per micro-batch (`pretrain_gpt.py:103`): keep it pure. Use `importlib.util.find_spec("moonep")` and never import `moonep` there.
- **Test**: `tests/unit_tests/transformer/moe/test_moonep_config.py`. One valid config (EP=8, TP=1, bf16, SwiGLU, no bias, fp32 router, S set), plus a `pytest.mark.parametrize` over each forbidden override, each checked with `pytest.raises(ValueError, match=...)`, plus the `moe_enable_deepep` case.
- **Command**: `$PY -m pytest -q tests/unit_tests/transformer/moe/test_moonep_config.py`

### M2: NVLink coverage assertion
- **Goal**: fail fast on every rank with one actionable message unless each EP group is a single NVLink/NVSwitch domain that MoonEP's memory-handle path can use. It must pass on this node.
- **Files**:
  - new `megatron/core/transformer/moe/moonep_backend/topology.py`;
  - `megatron/training/initialize.py`: insert after line 124.
- **Rules that shape the code**:
  - **No early raise.** Every local probe (the `moonep._C` import, `boot_id`, device properties, NVML) runs inside `try/except` and turns failures into problem strings, so every rank reaches every collective.
  - **Same host** means equal `boot_id` (`/proc/sys/kernel/random/boot_id` is per kernel instance, shared by containers on one node). The hostname is only used in messages.
  - **Fabric info is only needed for cross-host pairs.** A fabric-info query error is fatal only when the rank is part of a cross-host pair.
  - **Consensus.** With `consensus_group` set, every rank gathers `(ranks, problems)` from the whole group and raises if any EP group failed.
  - **Cache** by EP rank tuple only. A key that includes per-rank state (environment, hostname) could hit on some ranks and miss on others, which would deadlock the next collective. Tests call `_reset_moonep_topology_cache()` or pass `force=True` on every rank.

```python
# topology.py (sketch)
_VERIFIED: set = set()

def _read_boot_id() -> str:            # separate function so tests can monkeypatch it
    with open("/proc/sys/kernel/random/boot_id") as f:
        return f.read().strip()

def _nvml():
    from cuda.bindings import nvml      # ships with torch's cuda-bindings dependency (verified 13.4.3)
    nvml.init_v2()
    return nvml

def _local_record() -> dict:            # never raises
    rec = dict(rank=dist.get_rank(), host=socket.gethostname(), boot_id=None, pci=None, cc=None,
               multicast=None, fabric_handles=None,
               handle_env=os.environ.get("MOONEP_MEM_HANDLE_TYPE", "auto").lower(),
               fabric=None, fabric_error=None, errors=[])
    try: rec["boot_id"] = _read_boot_id()
    except Exception as e: rec["errors"].append(f"cannot read boot_id: {e!r}")
    try:
        dev = torch.cuda.current_device(); p = torch.cuda.get_device_properties(dev)
        rec["pci"] = f"{p.pci_domain_id:08X}:{p.pci_bus_id:02X}:{p.pci_device_id:02X}.0"
        rec["cc"] = tuple(torch.cuda.get_device_capability(dev))
    except Exception as e: rec["errors"].append(f"cannot query the CUDA device: {e!r}")
    try:
        from moonep._C import nvl_fabric_supported, nvl_multicast_supported
        rec["multicast"], rec["fabric_handles"] = bool(nvl_multicast_supported()), bool(nvl_fabric_supported())
    except Exception as e: rec["errors"].append(f"cannot import or query moonep._C: {e!r}")
    try:
        nvml = _nvml(); h = nvml.device_get_handle_by_pci_bus_id_v2(rec["pci"])
    except Exception as e:
        rec["errors"].append(f"NVML is unavailable or cannot see this GPU ({e!r}); install cuda-bindings with "
                             "the cuda.bindings.nvml module (13.4.3 verified) and expose all node GPUs to NVML")
        return rec
    try:
        fi = nvml.device_get_gpu_fabric_info_v(h)
        rec["fabric"] = (int(fi.state), int(fi.status), bytes(fi.cluster_uuid).hex(), int(fi.clique_id),
                         int(fi.health_summary))
    except Exception as e:
        rec["fabric_error"] = repr(e)    # fatal only for cross-host pairs
    return rec

def _probe_same_host_nvlink(records, me) -> list[dict]:     # never raises
    # for every peer j with records[j]["boot_id"] == records[me]["boot_id"]:
    #   status = nvml.device_get_p2p_status(h_me, h_j, nvml.GpuP2PCapsIndex.P2P_CAPS_INDEX_NVLINK)
    #   append {"peer": j, "status": int(status)}  or  {"peer": j, "error": repr(e)}

def _probe_shared_tmpdir(ep_group) -> list[bool]:           # collective; never raises
    # group rank 0: path = tempfile.mkdtemp(prefix="moonep_probe_") (None on failure);
    # broadcast_object_list([path], src=dist.get_global_rank(ep_group, 0), group=ep_group);
    # ok = bool(path) and os.path.isdir(path); all_gather_object(ok); rank 0 then shutil.rmtree(path)
    # (mirrors MoonEP/moonep/buffer.py:134-148)

def assert_moonep_ep_group_nvlink_coverage(ep_group, *, consensus_group=None, force=False):
    ranks = tuple(dist.get_process_group_ranks(ep_group))
    if ranks in _VERIFIED and not force:
        return
    records = [None] * len(ranks); dist.all_gather_object(records, _local_record(), group=ep_group)
    findings = [None] * len(ranks)
    dist.all_gather_object(findings, _probe_same_host_nvlink(records, ep_group.rank()), group=ep_group)
    uses_fabric, problems = moonep_handle_path(records)            # identical on all ranks
    tmp = None if uses_fabric else _probe_shared_tmpdir(ep_group)
    problems += evaluate_moonep_nvlink_records(records, findings, uses_fabric, tmp)
    failed = {ranks: problems} if problems else {}
    if consensus_group is not None:
        gathered = [None] * consensus_group.size()
        dist.all_gather_object(gathered, (ranks, problems), group=consensus_group)
        failed = {r: p for r, p in gathered if p}                   # deduplicated by EP rank tuple
    if failed:
        raise RuntimeError(_format_moonep_topology_error(failed))
    for w in fabric_health_warnings(records):
        logger.warning(w)
    _VERIFIED.add(ranks)

def _reset_moonep_topology_cache() -> None:
    _VERIFIED.clear()
```

- **`moonep_handle_path(records)`** mirrors `buffer.py:35-61`: if the values of `handle_env` differ, add a problem and treat the group as `fd`; an invalid value (not `auto`/`fabric`/`fd`) is a problem; `fd` means no fabric; otherwise fabric iff every rank has `fabric_handles`, and `fabric` with unsupported ranks is a problem.
- **Cross-host pairs are OK only if** all of these hold: both fabric states are 3 (`GpuFabricState.COMPLETED`), both statuses are 0, the cluster UUIDs are equal **and not `'00'*16`**, the clique ids are equal, neither `health_summary` is 2 (UNHEALTHY), `uses_fabric` is true and every rank has `fabric_handles`. A `health_summary` of 3 (LIMITED_CAPACITY) only produces a warning.
- **Exact error text** (one block per failing EP group, then one fix line):

```
MoonEP requires every expert-parallel (EP) group to be covered by a single NVLink/NVSwitch fabric, but EP group ranks {ranks} failed the check:
  - {one line per problem, e.g. "ranks 0 (hostA, GPU 00000000:DB:00.0) and 8 (hostB, GPU ...): different hosts and not in one multi-node NVLink clique (fabric state/status/cluster/clique: 3/0/0000…/0 vs 3/0/0000…/0; an all-zero ClusterUUID means a single-node NVSwitch system)"}
How to fix: keep --expert-model-parallel-size <= the GPUs of one NVLink domain (8 on an HGX B200 node); keep --expert-tensor-parallel-size 1 so EP groups are contiguous rank blocks; launch with torchrun so each node owns a contiguous rank block; expose every GPU of the node to NVML in each container (e.g. NVIDIA_VISIBLE_DEVICES=all) and share /tmp between the EP ranks; for multi-node NVLink (GB200 NVL72) enable IMEX so fabric memory handles work and do not set MOONEP_MEM_HANDLE_TYPE=fd.
```

- Other problem lines to produce:
  - any entry of a record's `errors`;
  - any record field a failed local probe left `None` (`cc`, `pci`, `boot_id`, ...): emit a problem line such as "rank i: compute capability unavailable" instead of comparing it. The evaluator must never raise, or every rank dies with the same `TypeError` instead of this message;
  - compute capability < 9.0 (MoonEP kernels need sm_90+);
  - multicast unsupported (MoonEP asserts it, `buffer.py:303`);
  - same-host pair with P2P status ≠ 0, naming the status (5 = NOT_SUPPORTED usually means PCIe-only);
  - same-host pair whose status could not be read ("cannot verify NVLink between ranks i and j");
  - two EP ranks on one GPU (equal `boot_id` and PCI id);
  - `MOONEP_MEM_HANDLE_TYPE` invalid or different across ranks (mismatched collectives, `buffer.py:40-51`);
  - rank-0 tmpdir not visible on some rank (fd path);
  - cross-host pair with fabric info unavailable, or on the fd path, or without fabric handles on every rank.
- **Call site** (training is exempt from the `parallel_state` rule):

```python
if (getattr(args, "moe_token_dispatcher_type", None) == "flex"
        and getattr(args, "moe_flex_dispatcher_backend", None) == "moonep"
        and mpu.model_parallel_is_initialized()):
    from megatron.core.transformer.moe.moonep_backend.topology import assert_moonep_ep_group_nvlink_coverage
    assert_moonep_ep_group_nvlink_coverage(mpu.get_expert_model_parallel_group(),
                                           consensus_group=torch.distributed.group.WORLD)
```

- **Pitfalls**:
  - Never index NVML by CUDA device index.
  - The check must run before any `Buffer()`, because MoonEP fails with `exit()` or a hang (§1.6).
  - It consumes no RNG, so it can run before seeding.
  - `lazy_mpu_init` returns `finish_mpu_init` to the caller (`initialize.py:161-170`), so the check still runs when the caller invokes it.
- **Tests** (`tests/unit_tests/transformer/moe/test_moonep_topology.py`; an autouse fixture calls `_reset_moonep_topology_cache()` on every rank):
  - pure cases for the evaluator and `moonep_handle_path`: single node passes; single node with `fabric_error` set passes; cross-host with zero UUID fails; cross-host with `fabric_error` fails; MNNVL with a valid clique passes; MNNVL with `health_summary=2` fails; P2P status 5 fails; unreadable P2P fails; env mismatch fails; invalid env fails; no multicast fails; duplicate GPU fails; cc 8.0 fails; fd path across hosts fails; tmpdir not visible fails;
  - live check for EP=8 and EP=4 (passes);
  - live negative: set `MOONEP_MEM_HANDLE_TYPE=fd` on rank 0 only, and expect the identical `RuntimeError` on all ranks;
  - live negative: monkeypatch `topology._read_boot_id` on ranks 4–7 to return another value, and expect the zero-UUID cross-host error on all ranks;
  - live negative with EP=4 and `consensus_group=WORLD`: inject a failure only in group [4–7], and expect all 8 ranks to raise;
  - exception containment: make `_read_boot_id` raise on rank 0 only, and expect an error line on all ranks instead of a hang.
- **Command**: `timeout 900 $DIST -m pytest -q -x tests/unit_tests/transformer/moe/test_moonep_topology.py`

### M3: Process-global Buffer and teardown
- **Files**:
  - `moonep_backend/buffer.py`;
  - `megatron/core/parallel_state.py`, after lines 2511-2516 (same pattern as `nccl_ep_finalize`);
  - `megatron/training/training.py`: after line 2274 (end of `pretrain`) and after the barrier at line 5549 (exit-condition branch of `train()`).
- **Signatures**:
  - `get_moonep_buffer(group, *, max_tokens_per_rank: int, hidden_size: int, topk: int, num_experts: int, num_sms: int | None, token_padding: int) -> "moonep.Buffer"`;
  - `destroy_moonep_buffer() -> None` (collective);
  - `release_moonep_buffer() -> None` (non-collective).

```python
key = (tuple(dist.get_process_group_ranks(group)), max_tokens_per_rank, hidden_size, topk, num_experts, num_sms, token_padding)
if _BUFFER is not None:
    if _BUFFER_KEY != key:
        raise RuntimeError(f"MoonEP: all MoE layers must share one moonep.Buffer, but a layer requested {key} while "
                           f"the live Buffer was built for {_BUFFER_KEY}. Call destroy_moonep_buffer() (collective) first.")
    return _BUFFER
from moonep import Buffer
_BUFFER = Buffer(max_tokens_per_rank, hidden_size, topk, num_experts, group.size(), num_sms=num_sms,
                 token_padding=token_padding, B=num_experts // group.size(), group=group, explicitly_destroy=True)
_BUFFER_KEY = key

def destroy_moonep_buffer():          # collective over the Buffer's EP group
    global _BUFFER, _BUFFER_KEY
    if _BUFFER is not None:
        _BUFFER.destroy()
    _BUFFER, _BUFFER_KEY = None, None

def release_moonep_buffer():          # for destroy_model_parallel and fault paths: never blocks
    global _BUFFER, _BUFFER_KEY
    _BUFFER, _BUFFER_KEY = None, None
```

- **Teardown wiring**:
  - `parallel_state.destroy_model_parallel()`: call `release_moonep_buffer()` and `_reset_moonep_topology_cache()` inside `try/except: pass`. It must not call the collective `destroy()`, because its only production caller is the in-process-restart fault path (§2.13).
  - End of `pretrain` (after line 2274) and the exit-condition branch of `train()` (after line 5549): when MoonEP is enabled, `torch.cuda.synchronize(); destroy_moonep_buffer()` on every rank.
  - Tests: call `destroy_moonep_buffer()` in teardown before `Utils.destroy_model_parallel()`.
  - A process that exits another way (an exception, `sys.exit` elsewhere) only gets the `ResourceWarning` of `Buffer.__del__`; the tensors are released at exit.
  - Per-layer expert storage has no collective teardown: its VMM deleters synchronize the current stream and unmap (`MoonEP/csrc/nvl_shared_buffer.cuh:331-339`). In tests, `torch.cuda.synchronize(); dist.barrier()` before dropping a model.
- **Test**: in `test_moonep_layer.py`, check that the same key returns the same object, a mismatched key raises, `destroy` followed by re-create works, and `release_moonep_buffer()` never blocks.

### M4: Expert storage, `MoonEPGroupedMLP` and the grad-state machine (no comms)
- **Files**: `moonep_backend/storage.py`, `moonep_backend/experts.py`, `moonep_backend/__init__.py` (`iter_moonep_experts`).
- **`MoonEPExpertStorage(ep_group, num_experts, hidden_size, ffn_hidden_size)`**. Port `my_moon_ep.py:247-338` (`_alloc_chunk`, `_gathered_items`, `_local_item`, `_map_items`, `_close_items`, `_composite_weight`, `_grad_buffers`) with the chunk shapes `gate=(epn,Hi,H)`, `up=(epn,Hi,H)`, `down=(epn,H,Hi)`, and call `_use_fabric_for_group(ep_group)` once. Port the zeroing at `my_moon_ep.py:207-210` too.
  - Attributes: `w[n]` bf16 `[E+B,...]`; `g[n]` fp32 `[E+B,...]`; `rb[n]` `[R,B,...]`; `lo`, `hi`, `E`, `B = epn`, `R`, `rank`, `epn`; `layer_number = None`; `fwd_outstanding = False`.
  - Methods: `fp32_targets(n) -> (g[n][lo:hi], g[n][E:])`; `zero_grads_()` zeroes **only** `g[n][lo:hi]` and `g[n][E:]`, and is called at the end of `__init__`.
  - It uses private MoonEP helpers (`moonep.buffer._exchange_ipc_fds`, `moonep.buffer._all_gather_shareables`, `moonep.buffer._use_fabric_for_group`, and `moonep._C.*`). This is risk 1 in §9.
- **`MoonEPGroupedMLP(num_local_experts, config, pg_collection=None, name=None)`**, a `MegatronModule` with `_is_moonep_experts = True` and no child modules:
  - `self.ep_group, self.expt_dp_group = pg_collection.ep, pg_collection.expt_dp`; `self.layer_number = None`; `self.storage = MoonEPExpertStorage(...)`;
  - masters `weight_gate`/`weight_up` `[epn,Hi,H]` and `weight_down` `[epn,H,Hi]`, created as `nn.Parameter(torch.empty(..., dtype=torch.float32, device=cuda), requires_grad=False)` with `allreduce=False`, `moonep_master=True` and `mark_keep_in_fp32`;
  - `_init_masters_()` as in H11 (skipped if `not config.perform_initialization`);
  - `_bind_grad_alias_(n)`: create **one** persistent view `v = g[n][lo:hi]`, call `mark_keep_in_fp32(v)`, then `p.grad = v; p.main_grad = v`, and record `v` and `v.data_ptr()`;
  - `sync_compute_weights()` under `@torch.no_grad`: `w[n][lo:hi].copy_(master)`, called at the end of `__init__`;
  - `register_load_state_dict_post_hook(lambda m, _: m.sync_compute_weights())`;
  - `grad_state = "CLEAN"`, `finalized = False`;
  - `set_layer_number(n)`: `self.layer_number = n; self.storage.layer_number = n`;
  - `check_invariants() -> list[str]`: masters fp32, `requires_grad is False` and unchanged `data_ptr`; each master's `.grad` and `.main_grad` are the recorded view with the recorded `data_ptr` and fp32 dtype; pools bf16 with unchanged `data_ptr`. `raise_on_broken_invariants(where)` raises `RuntimeError(f"MoonEP invariants broken {where}: {problems}")`;
  - `backward_dw()` is a no-op;
  - `__init__` ends with `torch.cuda.synchronize(); torch.distributed.barrier(group=self.ep_group)` (`my_moon_ep.py:238-239`).

```python
def _apply(self, fn, recurse=True):   # ignore dtype conversion, refuse device moves (VMM memory)
    for name, p in self._parameters.items():
        tgt = fn(torch.empty(0, dtype=p.dtype, device=p.device)).device
        if tgt != p.device:
            raise RuntimeError(f"MoonEP: refusing to move '{name}' from {p.device} to {tgt}: MoonEP expert "
                               "tensors live in (or alias) VMM memory mapped across the EP group.")
    return self
```

- **Grad-state machine** (on `MoonEPGroupedMLP`; every notice goes through `_notify(msg, strict=config.moe_moonep_strict_zero_grad)`):

```python
logger = logging.getLogger(__name__)

def _notify(msg: str, *, strict: bool) -> None:   # every occurrence, every rank
    if strict:
        raise RuntimeError(msg)
    if logger.isEnabledFor(logging.WARNING):
        logger.warning(msg)
    else:                                           # --logging-level above WARNING (initialize.py:657-678)
        print(f"WARNING: {msg}", file=sys.stderr, flush=True)

def zero_moonep_grads_(self):
    if self.storage.fwd_outstanding:
        raise RuntimeError(f"MoonEP: zero_moe_grads() called while MoE layer {self.layer_number} has a forward "
                           "without its backward.")
    if self.grad_state == "ACCUMULATING" and self.finalized:
        _notify(DISCARD_MSG, strict=...)
    self.repair_grad_aliases_("zero_moe_grads")
    self.storage.zero_grads_()
    self.grad_state, self.finalized = "CLEAN", False

def mark_zero_point_(self):          # MoonEPOptimizer.zero_grad
    if self.grad_state != "CLEAN":
        self.grad_state = "STALE"
    self.finalized = False

def mark_grads_consumed_(self):      # MoonEPOptimizer.prepare_grads
    self.grad_state, self.finalized = "STALE", False

def _on_grad_forward_(self):         # every gradient-producing forward (M5)
    self.repair_grad_aliases_("forward")
    if self.grad_state == "STALE":
        _notify(STALE_MSG, strict=...)
    self.grad_state = "ACCUMULATING"

def repair_grad_aliases_(self, where):
    # for each master: ok = p.grad is view and view.data_ptr() == recorded_ptr and view.dtype == torch.float32
    # if not ok: _notify(ALIAS_MSG, strict=...) and _bind_grad_alias_(n)  (a fresh view of the same VMM rows)
```

- **Exact texts**:
  - `STALE_MSG`: `f"MoonEP expert gradients of MoE layer {n} (rank {r}) were not zeroed since the last optimizer step / zero_grad(): this micro-batch's gradients will be ADDED to stale values. Call model.zero_moe_grads() at the start of every iteration (megatron/training/training.py:train_step does this right after optimizer.zero_grad())."`
  - `DISCARD_MSG`: `f"MoonEP: zero_moe_grads() discarded finalized expert gradients of MoE layer {n} (rank {r}) that no optimizer step consumed. Call it at the start of an iteration, not between finalize_model_grads and optimizer.step()."`
  - `ALIAS_MSG`: `f"MoonEP expert master '{name}' of MoE layer {n}: .grad was {desc} at {where}; expected the persistent fp32 VMM grad view. Something deleted or replaced it (torch.nn.Module.zero_grad / torch.optim.Optimizer.zero_grad with set_to_none=True?). Re-binding it."`
  - `n` is `self.layer_number` (the 1-based `TransformerLayer.layer_number`), `r` is `torch.distributed.get_rank()`.
- **Asserts** (exact text):
  - `f"MoonEP: expert chunk {shape} bf16 = {n} bytes must be a multiple of the VMM granularity {g} bytes (moonep._C.get_vmm_granularity()); choose num_moe_experts/EP, hidden_size and moe_ffn_hidden_size accordingly."`
- **Pitfalls**:
  - Never write rows of other experts. Zeroing the whole composite would write into peers' memory.
  - No `torch.manual_seed` in `__init__`.
  - Do not register pools as buffers.
  - Do not call `super()._apply`.
  - Test shapes must satisfy the 2 MiB granularity, for example E=16, EP=8, H=Hi=1024 gives 4 MiB.
- **Tests** (`test_moonep_layer.py -k storage`):
  - right after construction, every owned grad row and every local slot row is exactly 0, and `grad_state == "CLEAN"`;
  - after `Float16Module(config, m)`, `.to(torch.float16)` and `.cuda()`: masters fp32, `.grad` fp32 with `data_ptr` equal to `storage.g[n][lo:hi]`, pools unchanged, `check_invariants() == []`;
  - `.cpu()` raises;
  - after sync, `w[n][lo:hi] == masters.bf16()` bitwise;
  - peer check: `all_gather` the masters across EP and compare with the local composite rows `[0,E)`, read over NVLink after a barrier;
  - state machine by direct calls: CLEAN→`_on_grad_forward_`→ACCUMULATING→`mark_grads_consumed_`→STALE→`_on_grad_forward_` warns once→ACCUMULATING; `mark_zero_point_` keeps CLEAN; ACCUMULATING plus `finalized` then `zero_moonep_grads_` warns; strict mode raises; setting `p.grad = None` then `_on_grad_forward_` warns and re-binds.
- **Command**: `timeout 1200 $DIST -m pytest -q -x tests/unit_tests/transformer/moe/test_moonep_layer.py -k storage`

### M5: Autograd functions, manager and `MoELayer` wiring
- **Files**:
  - `moonep_backend/{autograd.py, manager.py}`;
  - `token_dispatcher.py:1835-1867` (new `elif` and the error string);
  - `moe_layer.py:221-224` (`set_layer_number`) and `moe_layer.py:369-375` (experts).
- **Autograd functions** (write `buffer.`/`ctx.buffer.` literally: the determinism manifest pattern matches `\w*buffer\.(dispatch|combine)\(`, M11):

```python
# launch_inter_rank_sync comes from moonep.inter_rank_sync (MoonEP/my_moon_ep.py:17). Per the §5 conventions,
# import it inside backward(), not at module level, so Megatron still imports without MoonEP.

class MoonEPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_sk, topk_sk, tpe, buffer, storage, holder):
        h, w, cu, plan = buffer.dispatch(x.contiguous(), w_sk.contiguous(), topk_sk, tpe, zero_copy=False)
        holder["plan"] = plan; ctx.buffer, ctx.plan, ctx.storage = buffer, plan, storage
        ctx.mark_non_differentiable(cu)
        return h, w, cu
    @staticmethod
    def backward(ctx, grad_h, grad_w, _):
        S = ctx.storage
        gx, gw, _ = ctx.buffer.combine(plan=ctx.plan, hidden_nvsh=grad_h.contiguous(),
                                       route_weights_nvs=None if grad_w is None else grad_w.contiguous())
        launch_inter_rank_sync(ctx.buffer._require_ctx())   # reduce_grad has no entry barrier (grad_reduce.py:468-469)
        ctx.buffer.reduce_grad(plan=ctx.plan, full_gate_grad=S.g["gate"], full_up_grad=S.g["up"], full_down_grad=S.g["down"],
                               gate_reduce_buffer=S.rb["gate"], up_reduce_buffer=S.rb["up"], down_reduce_buffer=S.rb["down"])
        S.fwd_outstanding = False
        return gx, gw, None, None, None, None, None

class MoonEPCombine(torch.autograd.Function):          # fwd combine / bwd plan-reuse dispatch
    @staticmethod
    def forward(ctx, z, buffer, plan):
        assert z.shape[0] == plan.nvs_s
        out, _, _ = buffer.combine(plan=plan, hidden_nvsh=z.contiguous())
        ctx.buffer, ctx.plan = buffer, plan
        return out
    @staticmethod
    def backward(ctx, g):
        assert g.dtype == torch.bfloat16
        grad_z, _, _, _ = ctx.buffer.dispatch(g.contiguous(), plan=ctx.plan)
        return grad_z, None, None

class GroupedMMLiveGrad(torch.autograd.Function):      # weights stored [G, out, in]
    @staticmethod
    def forward(ctx, x, w, offs, offs_live, targets):
        ctx.save_for_backward(x)
        ctx.w, ctx.offs, ctx.offs_live, ctx.targets = w, offs, offs_live, targets   # w is not saved: prefetch rewrites rows in place
        return torch._grouped_mm(x, w.transpose(-2, -1), offs=offs)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        g = g.contiguous()
        gx = torch._grouped_mm(g, ctx.w, offs=ctx.offs)                  # [N, in], all E+B groups
        gw = torch._grouped_mm(g.t(), x, offs=ctx.offs_live)             # [2*epn, out, in] bf16, live groups only
        with torch.no_grad():
            own, slots = ctx.targets
            own.add_(gw[: own.shape[0]]); slots.add_(gw[own.shape[0]:])
        return gx, None, None, None, None
```

- **Manager**:

```python
_FIRST_USE_ALIGNED: set = set()

class _MoonEPManager(_DispatchManager):
    def __init__(self, group, num_local_experts, router_topk, num_experts, config):
        assert importlib.util.find_spec("moonep") is not None, "moe_flex_dispatcher_backend='moonep' needs the moonep package."
        assert group.size() > 1 and num_experts % group.size() == 0
        assert config.moe_moonep_max_tokens_per_rank is not None, ("MoonEP needs moe_moonep_max_tokens_per_rank "
            "(megatron/training derives it in validate_args; library users must set it).")
        self.group, self.num_local_experts, self.router_topk = group, num_local_experts, router_topk
        self.num_experts, self.config = num_experts, config
        self._ranks = tuple(dist.get_process_group_ranks(group))
        assert_moonep_ep_group_nvlink_coverage(group)                     # cached after initialize_megatron's check
        self.buffer = get_moonep_buffer(group, max_tokens_per_rank=config.moe_moonep_max_tokens_per_rank,
                                        hidden_size=config.hidden_size, topk=router_topk, num_experts=num_experts,
                                        num_sms=config.moe_flex_dispatcher_num_sms, token_padding=config.moe_moonep_token_padding)
        assert int(self.buffer._require_ctx()["B"]) == num_experts // group.size(), \
            "MoonEP requires B == num_experts/EP (MoonEP/README.md:58)."
        self.storage = None
        self.token_probs = self.token_indices = self.tokens_per_expert = None
        self.dispatched_probs = self.cu_seqlens = self.plan = self.last_plan = None
    def bind_storage(self, storage): self.storage = storage
    def setup_metadata(self, routing_map, probs):                        # runs under torch.compile
        T = routing_map.shape[0]; rm = routing_map.reshape(T, self.num_experts); p = probs.reshape(T, self.num_experts)
        w, idx = torch.topk(p.masked_fill(~rm, float("-inf")), self.router_topk, dim=-1)
        self.token_probs, self.token_indices = w.float(), idx.to(torch.int32)
        self.tokens_per_expert = rm.sum(dim=0, dtype=torch.int32)
    def dispatch(self, hidden_states, async_finish=False, allocate_on_comm_stream=False):   # phase 1: always synchronous
        S, T = self.storage, hidden_states.shape[0]
        if InferenceMode.is_active(): raise RuntimeError(INFERENCE_MSG)
        if T > self.config.moe_moonep_max_tokens_per_rank: raise RuntimeError(CAPACITY_MSG)
        if hidden_states.dtype != torch.bfloat16: raise RuntimeError(DTYPE_MSG)
        grad_tracking = torch.is_grad_enabled() and (hidden_states.requires_grad or self.token_probs.requires_grad)
        if grad_tracking:
            if S.fwd_outstanding: raise RuntimeError(GUARD_MSG)
            S.fwd_outstanding = True
        sig = (self._ranks, T, grad_tracking)
        if sig not in _FIRST_USE_ALIGNED:           # align hosts before MoonEP JIT-compiles and launches barrier kernels
            torch.cuda.synchronize(); torch.distributed.barrier(group=self.group); _FIRST_USE_ALIGNED.add(sig)
        holder = {}
        h, w, cu = MoonEPDispatch.apply(hidden_states, self.token_probs, self.token_indices,
                                        self.tokens_per_expert, self.buffer, S, holder)
        self.plan = holder["plan"]
        self.buffer.prefetch_weight(plan=self.plan, full_gate_weight=S.w["gate"],
                                    full_up_weight=S.w["up"], full_down_weight=S.w["down"])
        self.dispatched_probs, self.cu_seqlens = w, cu
        return h
    def get_permuted_hidden_states_by_experts(self, h): return h, self.dispatched_probs
    def get_number_of_tokens_per_expert(self): return self.cu_seqlens   # NOTE: cumulative END offsets [E+B]
    def get_restored_hidden_states_by_experts(self, h): return h
    def combine(self, h, async_finish=False, allocate_on_comm_stream=False):
        out = MoonEPCombine.apply(h, self.buffer, self.plan)
        self.last_plan, self.plan, self.dispatched_probs, self.cu_seqlens = self.plan, None, None, None
        return out
```

- **Experts forward**:

```python
def forward(self, h_nvs, cu_seqlens, w_nvs):
    grad_producing = torch.is_grad_enabled() and h_nvs.requires_grad
    if self.training and torch.is_grad_enabled() and not h_nvs.requires_grad:
        raise RuntimeError(NO_GRAD_INPUT_MSG)
    if grad_producing:
        self._on_grad_forward_()                          # M4
    S = self.storage; offs_live = torch.cat((cu_seqlens[S.lo:S.hi], cu_seqlens[S.E:]))
    gmm = GroupedMMLiveGrad.apply
    gate = gmm(h_nvs, S.w["gate"], cu_seqlens, offs_live, S.fp32_targets("gate"))
    up   = gmm(h_nvs, S.w["up"],   cu_seqlens, offs_live, S.fp32_targets("up"))
    act  = (F.silu(gate) * up).float() * torch.nan_to_num(w_nvs).unsqueeze(-1)
    y    = gmm(act.to(torch.bfloat16), S.w["down"], cu_seqlens, offs_live, S.fp32_targets("down"))
    return y, None
```

- **Wiring**:
  - `token_dispatcher.py`: `elif backend == "moonep":` import `_MoonEPManager` inside the branch; `assert self.tp_size == 1 and self.ep_size > 1, "MoonEP dispatcher requires expert_tensor_parallel_size == 1 and expert_model_parallel_size > 1"`; then `self._comm_manager = _MoonEPManager(group=self.ep_group, num_local_experts=self.num_local_experts, router_topk=self.config.moe_router_topk, num_experts=self.config.num_moe_experts, config=self.config)` and `self.cudagraph_attrs = []`. Add `moonep` to the error string at lines 1864-1867.
  - `moe_layer.py:369-375`: if `uses_moonep(self.config)`, build `MoonEPGroupedMLP(self.num_local_experts, self.config, pg_collection=pg_collection, name=...)`, call `self.token_dispatcher._comm_manager.bind_storage(self.experts.storage)`, and if `layer_number is not None` call `self.experts.set_layer_number(layer_number)` (mirrors the router at lines 280-281). Otherwise use the spec builder.
  - `moe_layer.py:221-224`: after `self.router.set_layer_number(layer_number)`, add `if getattr(self.experts, "_is_moonep_experts", False): self.experts.set_layer_number(layer_number)`.
- **Messages** (exact text):
  - Capacity (`CAPACITY_MSG`): `f"MoonEP: MoE layer {n} received {T} tokens on this rank but moe_moonep_max_tokens_per_rank={S}. Raise --moe-moonep-max-tokens-per-rank to at least seq_length*max(micro_batch_size, eval_micro_batch_size)/context_parallel_size (divided by tensor_model_parallel_size with sequence parallelism)."`
  - dtype (`DTYPE_MSG`): `f"MoonEP dispatch requires bf16 hidden states (run with --bf16), got {dtype}."`
  - Guard (`GUARD_MSG`): `f"MoonEP: layer {n} started a second gradient-tracking forward before the backward of the previous one. Phase 1 keeps one prefetch/reduce slot set per layer, so this would silently corrupt gradients. Causes: pipeline/virtual pipeline parallelism, overlap_moe_expert_parallel_comm, MoE activation recompute, mlp_chunks_for_training > 1, repeated MTP layers, or calling the model twice before backward."`
  - Inference (`INFERENCE_MSG`): `"MoonEP (phase 1) does not support Megatron inference engines: MoonEP needs the same token count on every EP rank for every call. Use the alltoall dispatcher or the deepep/hybridep flex backends for inference."`
  - No grad input (`NO_GRAD_INPUT_MSG`): `f"MoonEP: MoE layer {n} runs a training forward with grad enabled, but neither its input nor its router probabilities require grad, so no expert gradient can be produced (MoonEP masters are not autograd leaves). Unfreeze the router or an upstream layer, or run this forward under torch.no_grad()."`
  - In the manager, `n` is `self.storage.layer_number`.
- **Pitfalls**:
  - `Function.forward` runs with grad disabled, so decide `grad_tracking` before calling `apply`.
  - `setup_metadata` is compiled, so no MoonEP calls there (NCCL-EP precedent at `token_dispatcher.py:1701-1703`).
  - Use `ctx.plan`, never a module-level last plan.
  - Plain `torch.topk` on dense probs can select an unrouted expert when a routed prob underflows to 0; the masked top-k avoids this.
  - `ctx.buffer._require_ctx()` is a private MoonEP API.
  - Never reduce over all `nvs_s` rows: tail rows are uninitialized (verified). This also means autograd anomaly mode reports NaNs there.
  - All guards raise before the first barrier-bearing MoonEP call, and their conditions are identical on all EP ranks, so they raise everywhere instead of hanging.
- **Tests** (`test_moonep_layer.py`; each test calls `set_layer_number(0)` like `tests/unit_tests/transformer/moe/test_token_dispatcher.py:151-152`):
  - **Parity** against `alltoall` + `SequentialMLP` (`get_gpt_layer_local_submodules(num_experts=16).mlp`, the pattern at `test_token_dispatcher.py:138-155`). Setup: E=16, K=4, H=Hi=1024, bf16, fp32 router, S = seq·b, 8 GPUs, EP=8. Copy the router, `gate[i]=fc1_i[:Hi]`, `up[i]=fc1_i[Hi:]`, `down[i]=fc2_i`, then sync. Compare output, `x.grad`, router `.grad` and expert grads (MoonEP fp32 `.grad` against baseline bf16 `.grad`) with relative L2 < 1e-2 as a starting threshold to calibrate. Add a TEGroupedMLP cell skipped without TE.
  - **Skew**: make half the experts unattractive and assert `(manager.last_plan.experts_to_copy >= 0).any()`.
  - **Accumulation**: 3 micro-batches without zeroing equal the baseline's accumulated grads, and no MoonEP warning is logged.
  - **Guard**: two forwards without backward raise.
  - **Layer number**: after `set_layer_number(3)`, the guard and STALE texts contain `MoE layer 3`.
  - **Inference guard**: with `InferenceMode.set_active()`, a forward raises the inference message.
  - **Initial weights**: MoonEP masters equal `SequentialMLP` GPU-init weights bitwise when built with the same seed and `use_cpu_initialization=False` **[unverified; if it fails, keep the weight-copy tests]**.
- **Command**: `timeout 1800 $DIST -m pytest -q -x tests/unit_tests/transformer/moe/test_moonep_layer.py`

### M6: Gradient finalization
- **Files**:
  - `moonep_backend/grads.py`;
  - `megatron/core/distributed/finalize_model_grads.py`: after line 686, and inside the `num_tokens` block after line 759.

```python
def reduce_moonep_expert_grads(model, config):
    rsm = get_rerun_state_machine()
    for chunk in model:
        ddp_cfg = getattr(chunk, "ddp_config", None)       # DDP attrs (distributed_data_parallel.py:138, 147)
        dp_cp = getattr(chunk, "dp_cp_group", None)
        for ex in iter_moonep_experts(chunk):
            if ex.finalized:
                raise RuntimeError(FINALIZED_TWICE_MSG)
            for name, g in ex.owned_grads():                 # the persistent views == masters' .grad
                if ddp_cfg is not None and (ddp_cfg.check_for_nan_in_grad or ddp_cfg.check_for_large_grads):
                    n = g.norm(p=2)                           # mirrors param_and_grad_buffer.py:389-425
                    where = f"MoonEP expert grad '{name}' of MoE layer {ex.layer_number} before the expert-data-parallel reduction"
                    if ddp_cfg.check_for_nan_in_grad:
                        rsm.validate_result(result=n, rejection_func=torch.isnan, message=f"found NaN in {where}",
                                            tolerance=0.001, fatal=True)
                        rsm.validate_result(result=n, rejection_func=torch.isinf, message=f"found Inf in {where}",
                                            tolerance=0.001, fatal=True)
                    if ddp_cfg.check_for_large_grads:
                        rsm.validate_result(result=n, rejection_func=partial(rsm.is_unexpectedly_large, threshold=10,
                                            context="moonep_grads"), message=f"found unexpected large grads in {where}",
                                            tolerance=0.001, fatal=False)
                if not config.calculate_per_token_loss and dp_cp is not None:
                    g.mul_(1.0 / dp_cp.size())               # == DDP expert factor with GTP/EGTP off
                if ex.expt_dp_group.size() > 1:
                    torch.distributed.all_reduce(g, group=ex.expt_dp_group)
            ex.finalized = True

def scale_moonep_expert_grads(model, scaling):             # per-token loss, after num_tokens is reduced
    for chunk in model:
        for ex in iter_moonep_experts(chunk):
            for _, g in ex.owned_grads():
                g.mul_(scaling)
```

- **Exact text** (`FINALIZED_TWICE_MSG`): `f"MoonEP: expert gradients of MoE layer {n} were finalized twice without zero_moe_grads() or optimizer.zero_grad() in between; finalize_model_grads must run once per iteration."`
- **Invariants**: touch only the own rows (never the slots). Phase 1 requires `gtp_weight_remat_size == expert_gtp_weight_remat_size == 1`, because DDP's correction factor is `egtp/gtp` (`distributed_data_parallel.py:257-259`). `context="moonep_grads"` keeps the large-value history separate from DDP's `"grads"` history (`rerun_state_machine.py:698-760`).
- **Pitfalls**:
  - Do not all-reduce `num_tokens` again (finalize already does it, `finalize_model_grads.py:749-752`).
  - A fatal NaN/Inf raises only on the ranks that see it, exactly like DDP's check; the others then wait in the next collective until the NCCL timeout.
  - NCCL all-reduce on VMM memory is **[unverified]**; the EP=4 test covers it.
  - A caller-installed `finalize_model_grads_func` (`training.py:4813-4814`) that skips these calls is caught by `MoonEPOptimizer.prepare_grads` (M7).
- **Test** (`test_moonep_training.py::test_finalize_matches_ddp`): DDP with `use_distributed_optimizer=False` (so `main_grad` is complete). Compare MoonEP owned grads with the baseline expert `main_grad` for EP ∈ {8, 4}, `average_in_collective` ∈ {False, True}, `calculate_per_token_loss` ∈ {False, True}; skip the per-token × `average_in_collective` cell, which DDP rejects (`distributed_data_parallel.py:250-253`). Add `test_finalize_inf_is_fatal`: inject `g[0,0,0] = inf` and expect the fatal rerun path.
- **Command**: `timeout 2400 $DIST -m pytest -q -x tests/unit_tests/transformer/moe/test_moonep_training.py -k finalize`

### M7: Optimizer integration
- **Files**:
  - new `megatron/core/optimizer/moonep_optimizer.py`;
  - `megatron/core/optimizer/__init__.py`: a `param_filter` argument for `_get_param_groups` (line 305); the early assert between lines 1081 and 1084; the build step after line 1304.

```python
class MoonEPOptimizer(FP32Optimizer):
    def __init__(self, optimizer, config, init_state_fn, moonep_experts):
        super().__init__(optimizer, config, init_state_fn); self.moonep_experts = list(moonep_experts)
        for ex in self.moonep_experts: ex.raise_on_broken_invariants("at optimizer construction (after wrapping)")
    def zero_grad(self, set_to_none=True):          # NEVER delete or zero: storage is persistent VMM memory
        for ex in self.moonep_experts: ex.mark_zero_point_()
    @torch.no_grad()
    def prepare_grads(self) -> bool:
        for ex in self.moonep_experts:
            ex.repair_grad_aliases_("optimizer.prepare_grads")
            if not ex.finalized:
                raise RuntimeError(NOT_FINALIZED_MSG)
            ex.mark_grads_consumed_()
        return super().prepare_grads()              # binds .grad = .main_grad (optimizer.py:1356-1359): same view
    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        ok = super().step_with_ready_grads()
        for ex in self.moonep_experts: ex.sync_compute_weights()
        return ok
    def reload_model_params(self, state_dict=None):
        for ex in self.moonep_experts: ex.sync_compute_weights()
    def _stage_model_params_from_main_params(self):   # optimizer.py:231-238
        for ex in self.moonep_experts: ex.sync_compute_weights()
    def offload_to_cpu(self): raise NotImplementedError("MoonEP masters alias VMM grad storage; CPU offload is unsupported.")
    def restore_from_cpu(self): raise NotImplementedError("MoonEP masters alias VMM grad storage; CPU offload is unsupported.")
```

```python
# _get_param_groups (optimizer/__init__.py:305-421): add
#   param_filter: Optional[Callable[[torch.nn.Parameter], bool]] = None
# and replace line 339's `if not param.requires_grad: continue` with
#   include = param_filter(param) if param_filter is not None else param.requires_grad
#   if not include: continue
# Everything else (config_overrides incl. MuP, is_expert_parallel, cross-rank alignment) is unchanged.

# get_megatron_optimizer, after the MoE block at line 1304 (call on EVERY rank: _get_param_groups is collective)
moonep_experts = [m for c in model_chunks for m in c.modules() if getattr(m, "_is_moonep_experts", False)]
moonep_groups = _get_param_groups(model_chunks, config, config_overrides, param_group_process_group,
                                  param_filter=lambda p: getattr(p, "moonep_master", False))
if moonep_groups:
    for g in moonep_groups:
        g["is_moonep"] = True                                  # informational; not an identifier key
    base, init_fn = _get_megatron_optimizer_based_on_param_groups(config=config, model_chunks=model_chunks,
                                                                  param_groups=moonep_groups, skip_megatron_wrapping=True)
    opt = MoonEPOptimizer(base, config, init_fn, moonep_experts)   # same config object (optimizer.py:1521)
    opt.grad_stats_parallel_group = expt_tp_pp_with_egtp_remat_group
    # tp_group / expert_tp_group exactly as optimizer/__init__.py:701-707 (same mpu fallback, same comment)
    optimizers.append(opt)                                        # LAST
```

- **Exact text** (`NOT_FINALIZED_MSG`): `f"MoonEP expert gradients of MoE layer {n} reached the optimizer without reduce_moonep_expert_grads() since the last zero point (the 1/|dp_cp| scaling and expert-DP sum are missing). Use megatron.core.distributed.finalize_model_grads as config.finalize_model_grads_func, or call reduce_moonep_expert_grads() from your finalize hook."`
- **Early assert** (`_assert_moonep_optimizer_supported(config, model_chunks, is_mfsdp_v2)`), when MoonEP experts exist:
  - `config.optimizer == "adam"` (emerging optimizers return early at `optimizer/__init__.py:1084-1093` and would silently drop the masters; SGD has `init_state_fn=None`, which `FP32Optimizer.sharded_state_dict(is_loading=True)` calls, `optimizer.py:1436-1443`);
  - no Megatron-FSDP (`model_chunks[0].ddp_config.use_megatron_fsdp` or `is_mfsdp_v2`), precision-aware optimizer, `optimizer_cpu_offload`, `loss_scale`, fp16, `use_layer_wise_distributed_optimizer` or `optimizer_cuda_graph`;
  - `model_chunks[0].ddp_config.overlap_grad_reduce` and `.overlap_param_gather` off (covers library users; `validate_args` covers the CLI).
  - Message: `f"MoonEP requires {name}={expected} ({reason}), got {value}."`, where the reason is "MoonEP masters are stepped by MoonEPOptimizer, an FP32Optimizer" for optimizer settings and "concurrent NCCL next to MoonEP barrier kernels is untested" for the DDP overlap flags.
- **Pitfalls**:
  - Do not add `is_moonep` to `param_group_identifier_keys` (`optimizer.py:139-149`).
  - Every rank has MoonEP masters in phase 1 (PP=1), so the base `count_zeros` is safe; do not add an early return that skips its collective.
  - With the torch fallback the step is `torch.optim.AdamW` (foreach), which allocates full-size temporaries; `AdamW(fused=True)` is a later optimization.
- **Tests** (`test_moonep_training.py`; each runs finalize through `forward_backward_no_pipelining` with `config.finalize_model_grads_func = finalize_model_grads`):
  - `test_optimizer_zero_grad_keeps_alias`: `optimizer.zero_grad()` with the default `set_to_none=True` leaves every master `.grad` with the same `data_ptr`.
  - `test_grad_norm_matches_baseline`: relative difference < 2e-2.
  - `test_pools_equal_masters_after_step`: bitwise `w[n][lo:hi] == masters.bf16()`.
  - `test_module_zero_grad_repaired`: `model.zero_grad()`, then a forward warns, re-binds, and training continues.
  - `test_step_without_finalize_raises`: set `config.finalize_model_grads_func = None`, run a step, expect `NOT_FINALIZED_MSG`.
  - `test_mup_overrides_reach_masters`: with MuP overrides, the MoonEP group's `max_lr` and `eps` equal the baseline expert group's.
- **Command**: `timeout 2400 $DIST -m pytest -q -x tests/unit_tests/transformer/moe/test_moonep_training.py -k optimizer`

### M8: `zero_moe_grads()` and always-notify wiring
- **Files**:
  - `megatron/core/transformer/module.py`: new method on `MegatronModule` (class at `module.py:66`);
  - `megatron/training/training.py`: after line 3320.

```python
def zero_moe_grads(self) -> int:                       # MegatronModule; DDP/Float16Module inherit it
    """Zero MoonEP expert gradients in place (they are never set to None). Returns #MoonEP layers."""
    n = 0
    for m in self.modules():
        if getattr(m, "_is_moonep_experts", False): m.zero_moonep_grads_(); n += 1
    return n
```

- **`train_step`**:

```python
optimizer.zero_grad()
for model_chunk in model:        # MoonEP grads are persistent VMM storage: zero in place, never None
    model_chunk.zero_moe_grads()
```

- **Pitfalls**:
  - Keep the call inside the rerun loop (`training.py:3314`).
  - The notice is intentionally loud: one line per MoonEP layer per rank per occurrence. Strict mode (`--moe-moonep-strict-zero-grad`) is the guaranteed channel.
  - Mirror the call at `megatron/core/transformer/cuda_graphs.py:2803-2805` and `megatron/core/transformer/moe/paged_stash.py:1201-1203` only if those features are ever enabled; both are asserted off.
- **Tests**:
  - `test_notify_every_time` (caplog): skip zeroing on iterations 2 and 3, and expect one warning per MoonEP layer on **each** of those iterations.
  - `test_no_false_positive_accumulation`: 4 micro-batches with zeroing, no warning.
  - `test_strict_raises`.
  - `test_eval_no_transition`: under `torch.no_grad()` the state does not change.
  - `test_eval_mode_with_grad_counts`: zero; eval-mode forward and backward with grad enabled; `optimizer.zero_grad()`; a train-mode forward without zeroing warns once.
  - `test_notify_when_warnings_hidden` (capfd): with `logging.getLogger().setLevel(logging.ERROR)`, the STALE text still reaches stderr.
- **Command**: `timeout 2400 $DIST -m pytest -q -x tests/unit_tests/transformer/moe/test_moonep_training.py -k "notify or strict or accumulation or eval"`

### M9: Checkpointing
- **Files**: `experts.py` (`sharded_state_dict`); `MoonEPOptimizer` (above).

```python
def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
    metadata = ensure_metadata_has_dp_cp_group(metadata)          # transformer/utils.py:293
    if metadata.get("singleton_local_shards", False):
        raise NotImplementedError("MoonEP (phase 1) does not support singleton_local_shards checkpoints.")
    A = len(sharded_offsets); ep_r, ep_n = self.ep_group.rank(), self.ep_group.size()
    kw = dict(replica_id=(0, 0, self.expt_dp_group.rank()), prepend_axis_num=len(sharded_offsets))
    fc1, fc2 = f"{prefix}experts.linear_fc1.weight", f"{prefix}experts.linear_fc2.weight"
    return {
      f"{prefix}weight_gate": ShardedTensor.from_rank_offsets(fc1, self.weight_gate, *sharded_offsets, (A, ep_r, ep_n), (A + 1, 0, 2), **kw),
      f"{prefix}weight_up":   ShardedTensor.from_rank_offsets(fc1, self.weight_up,   *sharded_offsets, (A, ep_r, ep_n), (A + 1, 1, 2), **kw),
      f"{prefix}weight_down": ShardedTensor.from_rank_offsets(fc2, self.weight_down, *sharded_offsets, (A, ep_r, ep_n), **kw)}
```

- **Invariants**:
  - `.data` **is** the master parameter, which the optimizer mapping requires (`dist_checkpointing/optimizer.py:66-67`).
  - Global shapes are `[NL,E,2Hi,H]` and `[NL,E,H,Hi]`, dropping `NL` for non-homogeneous layers, which matches `SequentialMLP`. The two fc1 fragments reproduce the SwiGLU factory's `(axis, rank, 2)` split for ETP=1 (`mlp.py:503-508`).
- **Pitfalls**:
  - Pools are refreshed by the post-load hook, and by `reload_model_params` on the no-optimizer path.
  - `torch_dcp` is broken for `ChainedOptimizer` (`checkpointing.py:2882` versus `optimizer.py:1644`), which is one more reason to require `torch_dist`.
  - Loading across chunkings (per-expert chunks versus `[epn,...]` chunks) relies on DCP resharding **[unverified here]**.
  - The MoonEP optimizer must stay the last chain entry: its checkpoint keys carry `chained_{idx}.` (§2.12).
- **Tests** (`test_moonep_checkpoint.py`, using `dist_checkpointing.save`/`load`, `serialization.py:74, 339`):
  - MoonEP → MoonEP: masters, pools and Adam states are equal;
  - EP=8 save → EP=4 load;
  - SequentialMLP save → MoonEP load: masters equal the baseline bf16 values upcast, exactly;
  - MoonEP save → SequentialMLP load: weights equal `masters.bf16()`;
  - a strict `load_state_dict` of the MoonEP model's state dict raises nothing (no missing or unexpected keys).
- **Command**: `timeout 1800 $DIST -m pytest -q -x tests/unit_tests/transformer/moe/test_moonep_checkpoint.py`

### M10: Model-level training parity, functional smoke and resume
- **Test** (`test_moonep_training.py::test_loss_parity`):
  - model: `GPTModel(num_layers=2, all MoE, hidden_size=1024, heads=8, E=16, K=4, seq=256, mbs=2)`, local spec, `Float16Module`, DDP (DistOpt on), `get_megatron_optimizer`;
  - baseline: `alltoall` with the same seed; copy weights **before** building the optimizers;
  - run 5 steps with 4 micro-batches each through `forward_backward_no_pipelining`, with `config.finalize_model_grads_func = finalize_model_grads`;
  - check the relative difference of the per-step loss < 2e-2 and of the grad norm < 5e-2, and assert that no MoonEP warning was logged.
- **Functional smoke** (on this VM it needs `pybind11` and `python3-config`: my baseline run failed in `_compile_dependencies` with `pybind11/pybind11.h: No such file`, because the dataset `Makefile` uses `python3-config` and `python3 -m pybind11`, see `megatron/core/datasets/Makefile`):

```
source /home/ubuntu/MoonEP/.venv/bin/activate && pip install pybind11   # plus python3.12-dev for python3-config
timeout 3600 python -m torch.distributed.run --nproc-per-node 8 pretrain_gpt.py \
 --num-layers 4 --hidden-size 1024 --num-attention-heads 8 --seq-length 1024 --max-position-embeddings 1024 \
 --micro-batch-size 2 --global-batch-size 64 --train-iters 50 --lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine \
 --lr-warmup-iters 5 --clip-grad 1.0 --weight-decay 0.01 --bf16 --swiglu --disable-bias-linear \
 --num-experts 16 --moe-ffn-hidden-size 1024 --moe-router-topk 4 --moe-router-dtype fp32 --moe-aux-loss-coeff 1e-2 \
 --expert-model-parallel-size 8 --expert-tensor-parallel-size 1 \
 --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend moonep \
 --use-distributed-optimizer --transformer-impl local --no-gradient-accumulation-fusion --attention-backend unfused \
 --mock-data --tokenizer-type NullTokenizer --vocab-size 32000 --log-interval 1 --eval-interval 25 --eval-iters 2 \
 --ckpt-format torch_dist --seed 1234
```

  - Every flag except the new `moonep` value parses with Megatron's parser today (verified with `--moe-flex-dispatcher-backend deepep`).
  - Rerun with `--moe-token-dispatcher-type alltoall` and compare the `lm loss` lines.
  - `--no-gradient-accumulation-fusion` is required without apex (`layers.py:1249-1258`; the flag defaults to True, `arguments.py:3144-3148`).
- **Resume** (exercises `training.py`'s load path, which the M9 unit tests bypass):
  - Run 50 iterations with `--save <dir> --save-interval 25`.
  - Rerun with `--load <dir>`: iterations 26–50 must match the uninterrupted run. Start with an exact comparison of `lm loss`; loosen only if the alltoall twin is not bitwise reproducible on this VM either.
  - Repeat once with `--no-load-optim` to exercise `reload_model_params` (`checkpointing.py:3218-3226`).
  - The log must not contain `load_return:` (the non-strict fallback print, `checkpointing.py:3071`).

### M11: Teardown, eval, determinism, docs
- **Teardown**: the collective `destroy_moonep_buffer` at the end of `pretrain` and in the exit-condition branch of `train()`, the non-collective release in `destroy_model_parallel` (M3); synchronize and barrier before dropping models in tests.
- **Eval**: covered by `test_eval_no_transition` and the functional run's `--eval-interval`.
- **Determinism**:
  - Add `tests/unit_tests/determinism/kernels/test_moonep_kernels.py`. It runs the same forward/backward twice with `zero_moe_grads()` in between (**not** the harness's `zero_grads`, which calls `model.zero_grad(set_to_none=True)`, `tests/unit_tests/determinism/utils.py:113`) and asserts bitwise-equal outputs, `x.grad` and master grads. It must import `megatron.core.transformer.moe.moonep_backend.autograd`, because the manifest test requires kernel tests to import a registered source (`tests/unit_tests/determinism/kernels/test_manifest.py:78-95`).
  - Register it: a `KernelEntry(name="moe_moonep", sources=("megatron/core/transformer/moe/moonep_backend/autograd.py",), tests=(K+"test_moonep_kernels.py",), kind="external-lib", ...)`. `buffer.dispatch(` and `ctx.buffer.combine(` match the kernel pattern (`tests/unit_tests/determinism/kernels/manifest.py:72`).
  - Changing `token_dispatcher.py`/`moe_layer.py`, which are registered sources (`manifest.py:299-308`), and `optimizer/__init__.py` (`manifest.py:449-457`) needs a matching test change or the `determinism-exempt` label (`manifest.py:35`; `tools/check_kernel_determinism_coverage.py:9-16`). Add a MoonEP cell to `test_moe_kernels.py`.
  - Do not use `--deterministic-mode` with MoonEP yet: it calls `torch.use_deterministic_algorithms(True)` (`megatron/training/determinism.py:211`), which is **[unverified]** with this path.
- **Docs**: a flex-backend section in `megatron/core/transformer/moe/README.md`.
- **Command**: `timeout 1200 $DIST -m pytest -q tests/unit_tests/determinism/kernels/test_moonep_kernels.py`

---

## 6. Config and CLI surface

New or changed fields become flags automatically (`arguments.py:2528`; `argument_utils.py:151-154`):

| Field / flag | Type (default) | Meaning |
|---|---|---|
| `moe_flex_dispatcher_backend` / `--moe-flex-dispatcher-backend moonep` | Literal (`deepep`) | enables MoonEP |
| `moe_moonep_max_tokens_per_rank` / `--moe-moonep-max-tokens-per-rank` | `Optional[int]` (None, derived in `validate_args`) | Buffer capacity S |
| `moe_moonep_token_padding` / `--moe-moonep-token-padding` | `int` (128) | `Buffer(token_padding=...)` |
| `moe_moonep_strict_zero_grad` / `--moe-moonep-strict-zero-grad` | `bool` (False) | raise instead of warn on MoonEP grad-lifecycle notices |
| (reused) `moe_flex_dispatcher_num_sms` | `Optional[int]` | `Buffer(num_sms=...)`; None becomes 32 |

**Static rules in `_validate_moonep`.** Each is a `ValueError` whose text starts `moe_flex_dispatcher_backend='moonep' requires `. The `moe_enable_deepep` conflict is checked earlier (M1).

| Rule | Why |
|---|---|
| `moe_token_dispatcher_type == 'flex'` | the backend lives in the flex dispatcher |
| `num_moe_experts` set, `EP > 1`, `E % EP == 0`, `EP <= 128` | `api.py:277-289`; `planning.py:1542-1544` |
| `expert_tensor_parallel_size == 1` (pass it explicitly when TP>1) and `gtp_weight_remat_size == expert_gtp_weight_remat_size == 1` | MoonEP moves whole experts; DDP factor parity (`distributed_data_parallel.py:257-259`) |
| `pipeline_model_parallel_size == 1`, `virtual_pipeline_model_parallel_size is None` | H7 |
| `bf16 and not fp16` | bf16 pools and kernels (`dispatch.py:897`; `combine.py`) |
| `gated_linear_unit and activation_func is F.silu`, `glu_linear_offset == 0`, `activation_func_clamp_value`, `activation_func_tanh_clamp_scale` and `activation_func_tanh_clamp_scale_linear` None, `moe_mlp_glu_interleave_size is None` | SwiGLU layout (§2.3) |
| `not add_bias_linear` | no expert bias |
| capacity factor None, no pad-to-capacity, no router padding for quantization, `moe_pad_experts_for_cuda_graph_inference=False` (`transformer_config.py:1016`) | `sum(tpe) == s·K`, otherwise planning traps (`MoonEP/README.md:83`) |
| `fp8 is None and fp4 is None and not fp8_param` | phase 1 |
| `moe_router_dtype == 'fp32'` | fp32 gating logits and probs, as in the MoonEP reference (`my_moon_ep.py:175-178, 382-388`). The Buffer's fp32 route-weight contract (`api.py:870-876`) is met by the `.float()` in `setup_metadata` anyway, so this is a numerics choice, not an API requirement |
| `moe_latent_size is None`, `not moe_apply_probs_on_input`, `moe_num_hash_layers == 0` | latent projections are TE-only and change the expert widths; probs-on-input moves the scaling point; hash routing assigns experts from a lookup table and is untested with MoonEP |
| `not moe_shared_expert_overlap` | the flex dispatcher runs shared-expert fc1/fc2 on `SharedExpertMLP.stream` around `manager.dispatch`/`combine` for any backend (`token_dispatcher.py:1972-1979, 2031-2032, 2048-2051`); interleaving that with MoonEP's barrier kernels is untested |
| no `overlap_moe_expert_parallel_comm`, `delay_wgrad_compute` or `overlap_dispatch_backward_with_experts_wgrad` | H7; dW is computed inline |
| `cuda_graph_impl == 'none'` | graphs valid for one `(s, total)` only (`MoonEP/README.md:86`) |
| no `'full'` recompute; `selective` without `'moe'` or `'moe_act'` | `'full'` and `'moe'` re-run dispatch, planning and prefetch inside backward (`moe_layer.py:752-768`); `'moe_act'` exists only in TEGroupedMLP (`experts.py:319-321`), so MoonEP experts would silently ignore it |
| no `hybrid_context_parallel`, `sequence_packing_scheduler` or `variable_seq_lengths` | equal s on all ranks |
| `mlp_chunks_for_training == 1`, `not mtp_use_repeated_layer` | one call per layer per micro-batch |
| no `cpu_offloading`, `fine_grained_activation_offloading` or `moe_paged_stash` | untested with VMM |
| `not init_model_with_meta_device`, `not use_cpu_initialization`, `transformer_impl != 'inference_optimized'` | VMM allocation and GPU init at construction; the builder path skips `.cuda()` under CPU init (`dist_utils.py:190-192`), and `lazy_mpu_init` forces CPU init (`initialize.py:163`) |
| `not freeze_base_model_for_mtp` | it sets `requires_grad=True` on MTP-layer masters before DDP wraps (`training.py:2502`), which registers them with DDP (ADR-D) |
| `hidden_size % 128 == 0`, `moe_ffn_hidden_size % 128 == 0`, `moe_router_topk <= 32` | `prefetch.py:401-402`; `grad_reduce.py:497-498`; `planning.py:1551` |
| `moe_moonep_max_tokens_per_rank` None or > 0, `moe_moonep_token_padding >= 1` | Buffer arguments |
| `importlib.util.find_spec('moonep')` | the package must be installed |

- **Training-only** (`validate_args`, M1): `torch_dist`; no `data_parallel_random_init`; no FSDP; no `--freeze-all-layers`; no `--inprocess-restart`; no `--overlap-grad-reduce` or `--overlap-param-gather`.
- **Optimizer-level** (`get_megatron_optimizer`, M7): Adam only; no MFSDP, precision-aware optimizer, CPU offload, loss scale, fp16, layer-wise DistOpt or optimizer CUDA graph; DDP overlap flags off.
- **Runtime** (M5): inference engines (`InferenceMode`), capacity, dtype, second forward, training forward without grad inputs.
- **Recommended bring-up flags**: `--transformer-impl local` and `--no-gradient-accumulation-fusion`, because TE and apex are not installed here.

---

## 7. Phase-1 unsupported features and roadmap
- **PP/VPP.**
  - Re-run `prefetch_weight(ctx.plan)` at the start of `MoonEPCombine.backward`. It is idempotent, and the source rows change only in sync (`MoonEP/benchmarks/bench_vs_deepep.py:270-274`).
  - Keep `reduce_grad` in dispatch-backward.
  - Relax the guard into a per-layer counter.
  - EP ranks share a PP stage (`parallel_state.py:903-906`), so the barrier order stays symmetric.
- **Shared prefetch/reduce pools** (MoonEP's design, `MoonEP/README.md:54, 68`): needs the same re-prefetch, plus mapping one physical chunk into every layer's composite. This needs a MoonEP API (§9 item 1). It cuts expert memory from 72·P to about 54·P.
- **MoE recompute**: the recompute forward builds a fresh plan and prefetch consumed via `ctx.plan`. It needs deterministic routing and a relaxed guard; test it.
- **EP comm overlap**: combined 1F1B runs F(mb i+1, layer j), including its prefetch into layer j's slots, before or concurrently with B(mb i, layer j) (`model_chunk_schedule_plan.py:575-589`), and dispatch/combine nodes run on the comm stream while the expert GEMMs run on the compute stream, each waiting only on its own plan's event (`model_chunk_schedule_plan.py:159-165`; `pipeline_parallel/utils.py:336-345`). Re-prefetching alone is not enough: phase 2 needs either two slot sets per layer alternated by micro-batch parity, or a re-prefetch in combine-backward on the compute stream plus an explicit event ordering it after the concurrent forward's prefetch.
- **CUDA graphs**: bucket `s` and warm up the JIT before capture (`api.py:20-24`).
- **MNNVL**: validate the fabric path on GB200 (MoonEP marks it untested).
- **fp8/mxfp4**: MoonEP prefetch already accepts `uint8`/`int8` with scales (`api.py:1037-1045`).
- **THD/packing**: pass `total_num_tokens`, which costs one host `all_gather` per step (`api.py:822-828`).
- **fp32 dW**: blocked by `_grouped_mm` requiring `out_dtype` equal to the input dtype (verified); upcast inputs if needed.
- **SGD**: pass a no-op `init_state_fn` to `MoonEPOptimizer`.
- **Construction-time warm-up**: one collective dummy dispatch/prefetch/combine/reduce per Buffer and pool shape, so every MoonEP kernel is compiled before training starts.
- **ETP > 1**: out of scope.

---

## 8. Test plan (repository conventions)
- **Location**: `tests/unit_tests/transformer/moe/`, which is globbed by the CI MoE bucket (`tests/test_utils/recipes/h100/unit-tests.yaml:269-275`).
  - Use `Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=Utils.world_size)` and, in teardown, `destroy_moonep_buffer()` then `Utils.destroy_model_parallel()`.
  - Markers: `pytest.mark.internal`, plus module-level `skipif` for "no moonep", "compute capability < 9" and "fewer than 2 GPUs".
- **Files**:

| File | Contents | Milestone |
|---|---|---|
| `test_moonep_config.py` | static rules, `moe_enable_deepep` conflict | M1 |
| `test_moonep_topology.py` | evaluator cases, live pass/negatives, consensus, exception containment | M2 |
| `test_moonep_grouped_mm.py` | single GPU: empty-group zero dW, native-layout correctness, rows past `offs[-1]` never read | M4/M5 assumptions |
| `test_moonep_layer.py` | Buffer, storage, state machine, parity, skew, accumulation, guards, layer number, mixed precision | M3–M5 |
| `test_moonep_training.py` | finalize, optimizer, notify, loss parity | M6–M8, M10 |
| `test_moonep_checkpoint.py` | checkpointing | M9 |
| `tests/unit_tests/determinism/kernels/test_moonep_kernels.py` | bitwise replay | M11 |

- **Baselines**: `alltoall` + `SequentialMLP` is always available here (TE is missing; `megatron/core/models/backends.py:221-230`). The TEGroupedMLP cells skip without TE.
- **Functional tests**:
  - `tests/functional_tests/test_cases/moe/gpt3_mcore_moonep_ep8_1node/model_config.yaml`, laid out like `gpt3_mcore_te_tp2_pp1_te_8experts_etp1_ep4/model_config.yaml`, with the M10 flags (EP=8, TP=1, local impl, no `--deterministic-mode`) and `TEST_TYPE: regular`;
  - a resume twin `gpt3_mcore_moonep_ep8_resume_torch_dist_1node` with `--save`, `--load`, `--save-interval` and `TEST_TYPE: ckpt-resume`, laid out like `gpt3_moe_mcore_te_ep8_resume_torch_dist_dist_optimizer_1node/model_config.yaml:23-24, 37, 58, 66`;
  - an alltoall twin;
  - recipe entries in `tests/test_utils/recipes/h100/moe.yaml` with `scope: [mr-github-broken]` until the CI image ships moonep (skills/mcore-testing);
  - golden values come from CI via `download_golden_values.py`.
- **Tolerances**: start from the verified MoonEP-versus-dense noise (dW ≈4e-3 relative) and calibrate.

---

## 9. Risks and open questions for the MoonEP author
1. **Public pool API.** Expert-pool allocation exists only in `my_moon_ep.py`, which uses private `moonep.buffer._*` helpers and `Buffer._require_ctx()`. Please expose a public API, ideally one that supports a pool shared across layers (`MoonEP/README.md:54, 68`).
2. **B200 coverage.** It ran here (§1.8), but the repository documents only H200/H20. Please run the MoonEP pytest suite on sm_100.
3. **Barrier timeout.** The 100 s device timeout is fixed (`_common.py:36-37`). Any host stall of one EP rank longer than that while its peers sit in a MoonEP kernel traps the device: a slow data loader, a slow checkpoint write on one rank, or JIT-compile skew. NCCL would tolerate the same stall up to its watchdog timeout. The first-use barrier in M5 aligns hosts before new shapes compile; a cheap NCCL all-reduce over EP before the first MoonEP layer of each forward would turn remaining stalls into NCCL waits (phase-2 hardening). Please make the timeout configurable or add AOT compile / warm-up.
4. **Concurrent NCCL.** Cooperative dedup kernels default to all SMs (`api.py:285-286`). Is there a deadlock risk with NCCL running concurrently (DDP `overlap_grad_reduce`/`overlap_param_gather`, or `CUDA_DEVICE_MAX_CONNECTIONS=1`)? **[unverified]** Phase 1 asserts both overlap flags off; `MOONEP_NUM_SMS_DEDUP` (`api.py:100-120`) is the knob to test before lifting that.
5. **Fabric auto-selection comment.** It picks fabric whenever all ranks support it; the comment says multi-node only (`buffer.py:29-31` versus `buffer.py:61`). Which predicate is intended for MNNVL: is clique id enough, or do the v4 per-type cliques matter?
6. **Tail rows.** Rows past `cu_seqlens[-1]` are uninitialized on B200, contrary to `my_moon_ep.py:424-426`. Is "never read" a guaranteed contract?
7. **Determinism.** Replay was bit-identical here. Is determinism guaranteed across runs and routing patterns?
8. **Reference defects** from §1.7: plan attribute name, `reference_grads` bugs, and the per-micro-batch EP all-reduce of the router grad.
9. **dW precision.** dW is rounded to bf16 per micro-batch, like Megatron's local path without fused wgrad accumulation. Is an fp32-accumulating grouped GEMM planned?
10. **Megatron side, unrelated to MoonEP.** `megatron/core/recompute.py:322-323`, changed in the merge `1912044d7` (both parents, `917740c76` and `ec806b20a`, used `tensor_parallel.checkpoint` there), calls `cf(*args)` without checkpointing in the bf16 branch. bf16 full recompute is currently a no-op for baselines.
11. **Other [unverified] items**:
    - `cuda.bindings.nvml` in the CI container's cuda-bindings (torch only requires `cuda-bindings>=13.0.3`); if it is missing, add a `pynvml` adapter (`nvidia-ml-py` is not installed here, so its names are unchecked);
    - DCP resharding between per-expert and `[epn,...]` chunks (M9 test);
    - NCCL on VMM memory (M6 EP=4 test);
    - identical initialization against SequentialMLP (M5 test);
    - VMM lifetime when peers unmap at different times (CUDA reference-counting semantics);
    - MoonEP under `torch.use_deterministic_algorithms(True)` (M11).

---

## Appendix A: What I ran on this VM (read-only; no repository file changed except this one)
- `cd /home/ubuntu/MoonEP && timeout 1500 $DIST my_moon_ep.py`: 86.2 ms/step; consistency checks True.
- `cd /home/ubuntu/MoonEP && timeout 1800 $DIST moe_moon_ep.py`: parity numbers in §1.8.
- **`torch._grouped_mm` probe** (GPU 0): 8 groups, sizes `[0,128,0,256,128,0,0,128]`, 64 tail rows, NaN-poisoned allocator. Checked the empty-group dW in both layouts, `out_dtype=fp32` (raises), the `W^T`-view forward, dX, tail rows and fp32 inputs, plus timing at E=64, B=8, H=2048, Hi=1024, N=16384 (both layouts ≈0.057 ms).
- **NVML / MoonEP probe**: `get_vmm_granularity()`, `get_multicast_granularity(8)`, `nvl_multicast_supported()`, `nvl_fabric_supported()`, `device_get_p2p_status(h_i, h_j, P2P_CAPS_INDEX_NVLINK)` for all 56 ordered pairs, `device_get_gpu_fabric_info_v` (state, status, cluster UUID, clique, `health_summary`) on all 8 GPUs, `device_get_handle_by_pci_bus_id_v2`, the `GpuP2PStatus` and `GpuFabricHealthSummary` enums, and the torch PCI-id formatting under `CUDA_VISIBLE_DEVICES=3`.
- **`_apply` override probe** (CPU): masters fp32, alias kept, sibling module converted to bf16, `.to('meta')` rejected, `load_state_dict` in place.
- **DCP probe** (CPU, `no_dist=True`): bf16→fp32 load exact; fp32→bf16 load equals `.bfloat16()`.
- **Parser probe** (CPU): `add_megatron_arguments(argparse.ArgumentParser())` parses every M10 flag (with `deepep` in place of `moonep`, plus `--save/--load/--save-interval/--no-load-optim/--logging-level`); the backend choices are `('deepep', 'hybridep', 'ncclep')` today.
- **Import probe** (CPU): `tests.unit_tests.test_utilities`, `tests.unit_tests.dist_checkpointing`, `megatron.core.transformer.moe.moe_layer`, `megatron.core.optimizer` and `megatron.training.training` import in the venv; only the pytest-side packages of M0 are missing.
- **Baseline `pretrain_gpt.py`** (alltoall, `--transformer-impl local`): failed in dataset-helper compilation (missing `pybind11` / `python3-config`); M10 lists the prerequisites.
