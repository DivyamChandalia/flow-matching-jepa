# LeWM Flow-Matching + Shortcut-Forcing — Results

## Setup
- **Models**: ViT-tiny, 192-D latent, img_size=112, history_size=3, num_preds=1
- **Training**: 100 epochs, effective batch 128 (batch=32, accum=4), AdamW lr=5e-5, SIGReg weight=0.09, 16-mixed precision
- **Dataset**: PushT expert (500 episodes, ~50,500 frames, frameskip=5)
- **GPU**: RTX 3050 Ti Laptop 4 GB VRAM (peak ~1.08 GB allocated)
- **Checkpoints**: `~/.stable_worldmodel/checkpoints/<run_name>/`

## Offline Latent Dynamics Evaluation

64 held-out sequences, fixed noise seed 12345. Reported: latent MSE (↓), cosine similarity (↑).

| Model | Target Space | 1-step MSE | 3-step MSE | 5-step MSE | 1-step Cos | 5-step Cos |
|-------|-------------|-----------|-----------|-----------|-----------|-----------|
| LeWM (MSE baseline) | — | **0.008 ± 0.0004** | **0.036 ± 0.011** | **0.070 ± 0.015** | **0.996** | **0.955** |
| Flow-LeWM | latent | 0.317 ± 0.007 | 0.643 ± 0.003 | 0.816 ± 0.001 | 0.865 | 0.385 |
| Flow-LeWM | delta | 0.086 ± 0.012 | 0.230 ± 0.039 | 0.369 ± 0.046 | 0.942 | 0.765 |
| Shortcut-LeWM | latent | 0.371 ± 0.013 | 0.726 ± 0.008 | 0.904 ± 0.008 | 0.801 | 0.324 |
| Shortcut-LeWM | delta | **0.084 ± 0.027** | **0.202 ± 0.063** | **0.332 ± 0.087** | **0.962** | **0.867** |

## Conditioning Dependence (1-step Shuffle Ratios)

Shuffle ratio >1 indicates the model uses that input. Higher = stronger conditioning.

| Model | Target Space | Action Shuffle Ratio | Context Shuffle Ratio |
|-------|-------------|---------------------|---------------------|
| LeWM (MSE baseline) | — | **6.3 ×** ✅ | **225.7 ×** ✅ |
| Flow-LeWM | latent | 1.07 × ❌ | 3.4 × ⚠️ |
| Flow-LeWM | delta | 1.00 × ❌ | 19.0 × ✅ |
| Shortcut-LeWM | latent | 1.05 × ❌ | 3.5 × ⚠️ |
| Shortcut-LeWM | delta | 1.00 × ❌ | 14.9 × ✅ |

## Key Findings

1. **Baseline MSE works well** — excellent latent dynamics (0.008 MSE, 0.996 cos) and strong conditioning on both actions (6.3×) and visual context (225×).

2. **Flow with `target_space=latent` fails** — predicts the SIGReg Gaussian marginal instead of conditioned dynamics. Action shuffle ratio ~1.07× (effectively ignores actions), context shuffle only 3.4×.

3. **`target_space=delta` helps but doesn't fix action conditioning** — switching to delta-space targets improves MSE 3–4× (0.086 vs 0.317) and context conditioning jumps to 19×. However, **action shuffle ratio remains ~1.0×** — the flow head ignores the ARPredictor context that carries action information.

4. **Shortcut-LeWM matches Flow-LeWM** — similar performance to flow delta, no improvement from shortcut bootstrapping. Same action conditioning collapse.

## Suspected Root Cause

The `ShortcutFlowHead` uses `noisy_proj(x) + context_proj(x)` — a residual addition that the MLP can learn to ignore. The noise projection dominates because it directly processes the target signal, while context is a small additive bias. The MSE baseline proves the ARPredictor *does* carry action information (6.3× shuffle ratio), so the bottleneck is in the flow head's ability to use that context.

**Recommended fix**: Concatenate `[noisy_proj; context_proj]` instead of adding, or use adaLN-style modulation in the flow head.

---
## Ablation: Flow-Matching with Isolated Encoder

Three controlled experiments testing whether flow conditioning can be stabilized by keeping the proven AR+encoder training path unchanged.

### Experiment A — Frozen Encoder, AR-Residual Flow (10 epochs)

**Setup**: Load LeWM (MSE) checkpoint; freeze encoder/projector/action encoder/ARPredictor/pred_proj; train only `ShortcutFlowHead` on `r = z_{t+1} - ARPredictor(z_t, a_t)`.

**Evaluation note**: AR rollout repeats the last action (inference-time setting). LeWM MSE (100ep) reference uses oracle actions (teacher-forced). Flow evaluation also uses repeated actions, so the direct comparison is Flow vs AR (both repeated actions).

| Model | 1-step MSE | 3-step MSE | 5-step MSE | 10-step MSE | Action Ratio | Context Ratio |
|-------|-----------|-----------|-----------|------------|-------------|--------------|
| AR (repeated actions) | 0.0074 | 0.120 | 0.272 | 0.538 | 8.1× | 213× |
| **Flow (NFE=4, repeated actions)** | 0.0083 | **0.037** | **0.068** | **0.197** | **5.5×** | **237×** |

| Reference (teacher-forced) | 1-step MSE | 3-step MSE | 5-step MSE | 10-step MSE |
|---------------------------|-----------|-----------|-----------|------------|
| LeWM MSE (100ep, oracle actions) | 0.008 | 0.036 | 0.070 | 0.194 |
| Exp A AR (oracle actions) | 0.008 | 0.036 | 0.070 | 0.194 |
| **Exp A Flow (NFE=4, repeated actions)** | 0.008 | **0.037** | **0.068** | **0.197** |

**Result**: With teacher forcing (oracle actions), AR and Flow match at 1-step, but Flow's advantage is only visible at 3+ steps with imperfect actions. At 3-step with repeated actions, Flow gives **3.2× improvement** over AR (0.037 vs 0.120). At 5-step: **4.0×** (0.068 vs 0.272). The flow denoising corrects the action error that would otherwise compound in a pure AR rollout. Multi-step performance **matches the 100-epoch MSE baseline** with teacher forcing, despite training only the flow head for 10 epochs.

**Conclusion**: The AR-residual flow approach works when the encoder is stable. The flow corrects both AR prediction errors AND action errors during inference.

### Experiment B — Joint Training, Flow Isolated (10 epochs)

*(In progress)*

### Experiment C — Flow Gradients into Predictor

*(If needed)*

### Experiment D — Full-Transformer Flow, Joint from Scratch (10 epochs)

**Motivation**: Rule out head capacity as the cause of flow conditioning collapse by making the AR predictor itself a full-transformer flow model (Flow-JEPA-style), trained jointly with the encoder from scratch.

**Architecture** (`module.py`, `jepa.py`):
- `FlowTransformerPredictor`: 6-layer transformer, heads=16, mlp_dim=2048, dim_head=64 (same budget as `ARPredictor`). Per block: causal self-attn over noisy tokens + cross-attn to `[hist latents; action tokens]` + MLP, all AdaLN-zero modulated by continuous flow time `t` (sinusoidal embedding). Frame positions via sinusoidal embeddings. Outputs velocity (`v-pred`).
- `FlowJEPA`: no separate flow head. `flow_loss` = `MSE(v_pred, target - noise)` on `x_t = (1-t)*noise + t*target`, `t ~ U[0,1]`; encoder gradients flow through history AND target latents (fully joint). Euler sampling, autoregressive rollout, seeded common-noise CEM cost. Source is pure `N(0,I)` (no `noisy_current`).
- `eval_dynamics.py` extended with a `sample_next_flow` branch so FlowJEPA evaluates under the identical protocol (no protocol change). New tests in `tests/test_flow_transformer.py` (4 tests: shape, joint grads, rollout/cost determinism + seed sensitivity). Full suite: 29 passed.

**Training** — identical hyperparams to prior runs except noted deviations:
```bash
python train.py data=pusht model=flow_transformer_lewm objective=flow_transformer \
  seed=3072 img_size=112 loader.batch_size=32 loader.num_workers=4 \
  +trainer.accumulate_grad_batches=4 trainer.max_epochs=10 \
  trainer.precision=16-mixed output_model_name=pusht_flow_transformer_10ep
```
(seed 3072, 192-D, H=3/P=1, eff batch 128, AdamW lr 5e-5, SIGReg 0.09. Deviation: `loader.num_workers=4` instead of default 6 — math-neutral, required to avoid the host OOM-killer on 13 GB RAM. Checkpoint: `pusht_flow_transformer_10ep`, final val `flow_loss=0.392`, `sigreg=2.55`. ~30 min on RTX 3050 Ti laptop.)
```bash
python eval_dynamics.py --config-name=pusht \
  policy=pusht_flow_transformer_10ep/weights_best.pt eval.img_size=112 +nfe=4
```

**Full results** (64 sequences, seed 12345, oracle actions):

| Horizon | NFE=4 MSE | NFE=4 Cos | NFE=8 MSE | NFE=8 Cos |
|---------|----------|----------|----------|----------|
| 1-step | 0.414 ± 0.037 | 0.726 ± 0.022 | 0.417 | 0.731 |
| 3-step | 0.489 ± 0.016 | 0.690 ± 0.006 | 0.497 | 0.694 |
| 5-step | 0.538 ± 0.013 | 0.650 ± 0.006 | 0.547 | 0.655 |
| 10-step | 0.722 ± 0.024 | 0.525 ± 0.014 | 0.736 | 0.529 |
| 15-step | 0.831 ± 0.026 | 0.447 ± 0.017 | 0.851 | 0.449 |

Conditioning (NFE=4): action shuffle 1.004× (ignores actions), context shuffle 4.04×. NFE=8: 1.010× / 3.71×. Noise sensitivity: prediction variance 0.29 across 8 noise draws (vs ~0.001 for collapsed MLP runs) — outputs swing with noise instead of anchoring to conditioning. Flow action-output effect 0.0018.

**Comparison**:

| Model (NFE=4) | 1-step MSE | 3-step MSE | 5-step MSE | 1-step Cos | Action Ratio | Context Ratio |
|---------------|-----------|-----------|-----------|-----------|-------------|--------------|
| LeWM MSE (100ep, ref) | 0.008 | 0.036 | 0.070 | 0.996 | 6.3× | 225× |
| Flow-MLP latent (100ep, ref) | 0.317 | 0.643 | 0.816 | 0.865 | 1.07× | 3.4× |
| Flow-MLP delta (100ep, ref) | 0.086 | 0.230 | 0.369 | 0.942 | 1.00× | 19.0× |
| Exp A frozen AR-residual (10ep, ref) | 0.008 | 0.037 | 0.068 | — | 5.5× | 237× |
| **Flow-Transformer joint scratch (10ep)** | **0.414** | **0.489** | **0.538** | **0.726** | **1.00×** | **4.0×** |

**Result**: Head capacity was **not** the bottleneck. A full transformer in the flow loop, trained jointly from scratch, reproduces the same disease as the MLP head: action conditioning collapses (~1.0×), context conditioning stays weak (~4×, same band as MLP-latent 3.4×, far below delta 19× and MSE 225×). NFE 4→8 barely moves metrics — integration is converged, conditioning is the problem, not solver error. The only configuration that preserves conditioning is Exp A — frozen good encoder + residual target. Implication: the failure is in joint representation/flow dynamics (flow loss lets the encoder drift to a marginal the flow can match unconditionally), not in the flow head's expressivity.

**Caveats**: 10ep vs 100ep (possible undertraining — but MLP-latent at 100ep is no better at 0.317); latent scales differ across runs so cross-run MSE is indicative only (cosine confirms the ranking). Next candidates: (1) frozen-encoder + full-transformer residual flow (capacity × stability), (2) `noisy_current` source, (3) concat/AdaLN conditioning instead of cross-attn-only.

### Experiment E — Direct Flow-JEPA Reproduction (10 epochs)

**Motivation**: Reproduce the external Flow-JEPA method (repo: `git@github.com:HuoYanchen/Flow-JEPA.git`, https://github.com/HuoYanchen/Flow-JEPA) faithfully and test it under our apples-to-apples protocol, to see whether its exact design choices (trajectory predictor, one-shot horizon, H=1/P=5, `v-pred`, `standard_noise` source) avoid the conditioning collapse.

**Port** (`module.py`: `RMSNormMLP`, `FJTrajectoryBlock`, `FlowTrajectoryPredictor`; `jepa.py`: `TrajectoryFlowJEPA`; `train.py`: `fjepa_forward`; configs `model/fjepa.yaml` + `objective/fjepa.yaml`; `tests/test_fjepa.py`, 33-test suite green):
- Predictor math matches their repo: self-attn (no causal mask, per their config) + cross-attn with hist-full/action-causal mask + MLP, AdaLN-zero time modulation, sinusoidal `t` + frame positions (positions on pred/action tokens only — hist has none, as in theirs), RMSNormMLP output head, `pred_proj=null`. Simplification: our port fixes K=1 (ViT CLS token, `(B,T,D)`) where theirs is token-generic `(B,T,K,D)` — identical math for the ViT-CLS case used here (masks/positions reduce to the same matrices).
- Loss: unweighted `MSE(v_pred, target - source)`, `t ~ U[0,1]`, pure-Gaussian `standard_noise` source (their default). `noisy_current` source also implemented (`flow_source` config) but not yet run.
- Rollout is one-shot over the full horizon (no autoregression); CEM cost compares final predicted vs goal latent (their criterion). Seeded-RNG inference (`set_flow_seed`) ported.
- Training: H=1 history → P=5 joint trajectory (`future_act = act[:, H-1:H-1+P]`, `target = emb[:, H:H+P]`), encoder joint from scratch, `flow_loss + SIGReg`, no AR loss.

**Training** — same budget as Exp D / our own flow exps on our data (PushT expert, 500 episodes / ~50,500 frames, frameskip=5; seed 3072, img 112, 192-D ViT-tiny, eff batch 128 = 32×accum4, AdamW lr=5e-5 wd=1e-3, SIGReg 0.09, 16-mixed, 10 epochs; H=1/P=5 per their method — the only intentional deviation from our H=3/P=1):
```bash
python train.py data=pusht model=fjepa objective=fjepa \
  seed=3072 img_size=112 history_size=1 num_preds=5 \
  loader.batch_size=32 loader.num_workers=4 \
  +trainer.accumulate_grad_batches=4 trainer.max_epochs=10 \
  trainer.precision=16-mixed output_model_name=pusht_fjepa_repro_10ep
```
```bash
python eval_dynamics.py --config-name=pusht \
  policy=pusht_fjepa_repro_10ep/weights_best.pt eval.img_size=112 \
  +eval.history_size=1 +nfe=8
```
(`eval_dynamics.py` gained a one-shot trajectory branch + `eval.history_size` override; single-step noise/action diagnostics are skipped for one-shot models. Checkpoint: `pusht_fjepa_repro_10ep`.)

**Results** (64 sequences, seed 12345, oracle actions, NFE=8; verified by re-run 2026-09-16 — dynamics reproduce exactly):

| Horizon | MSE | Cos |
|---------|-----|-----|
| 1-step | 0.716 ± 0.020 | 0.387 ± 0.035 |
| 3-step | 0.805 ± 0.002 | 0.233 ± 0.011 |
| 5-step | 0.830 ± 0.053 | 0.208 ± 0.023 |
| 10-step | 0.940 ± 0.006 | 0.093 ± 0.001 |
| 15-step | 0.963 ± 0.002 | 0.082 ± 0.000 |

NFE=4 check (same checkpoint, direct comparison to Exp D's NFE=4):

| Horizon | MSE | Cos |
|---------|-----|-----|
| 1-step | 0.688 ± 0.018 | 0.387 ± 0.034 |
| 3-step | 0.762 ± 0.003 | 0.233 ± 0.011 |
| 5-step | 0.784 ± 0.047 | 0.208 ± 0.023 |
| 10-step | 0.891 ± 0.005 | 0.092 ± 0.000 |
| 15-step | 0.917 ± 0.003 | 0.082 ± 0.001 |

Conditioning: action shuffle 1.001× (stable across re-runs: 1.0006–1.001×), context shuffle 1.3–1.7× across re-runs (1.33× / 1.64× / 1.70× — the shuffle uses unseeded `randperm`, so expect ±0.2 jitter; always weak).

**Result**: The faithful reproduction collapses too — and harder than our variants (worse than Exp D's 0.414/4.0× at NFE=4 and MLP-latent-100ep's 0.317/3.4×; NFE 4→8 barely moves it: 0.688→0.716 at 1-step, so solver error is not the cause). Action conditioning is exactly 1.0×; even context conditioning is barely above chance (1.3–1.7×). H=1 (single-frame, no motion history) gives the flow even less to hold onto than H=3, and the joint encoder + unconditional-noise setup drifts the same way.

**Caveats**: H=1/P=5 differs from our H=3/P=1 runs by method design (not a same-shapes comparison — the point is method-level); 10ep vs their 20ep; img 112 vs their 224. `noisy_current` source (their obs-anchored option, the one thing we haven't tried) remains untested and is now the most interesting follow-up — it's the only design difference left that could ground the source.

### Experiment E2 — Direct-clone check: upstream code trained as-is on our data/settings

**Motivation**: Rule out porting error — train the actual upstream repo (temporary local clone, commit `e9172e7`) directly, with our data/epochs/batch, and eval under the same protocol.

**Equivalence test** (one-off script, not retained): upstream `FlowTrajectoryPredictor` weights loaded into the port via explicit key mapping → forward outputs **bit-identical** (max abs diff 0.0, all keys map, cross masks equal for K=1). The port is mathematically faithful; any result difference is run noise, not a port bug.

**Direct training** — upstream `train.py` unmodified, only Hydra overrides (same budget as Exp E: PushT expert, seed 3072, img 112, 192-D, batch 32×accum4 = eff 128, AdamW 5e-5, SIGReg 0.09, 16-mixed, 10 epochs, H=1/P=5). One adaptation was required: upstream `data.dataset.name` (`pusht_expert_train.h5`) doesn't resolve against our cache layout, so the full local path was passed:
```bash
cd Flow-JEPA  # upstream clone
python train.py data=pusht data.dataset.name=$HOME/.stable_worldmodel/pusht_expert_train.h5 \
  img_size=112 loader.batch_size=32 loader.num_workers=4 \
  +trainer.accumulate_grad_batches=4 trainer.max_epochs=10 \
  trainer.precision=16-mixed output_model_name=pusht_fjepa_direct_10ep
```
(~40 min on RTX 3050 Ti; checkpoints `pusht_fjepa_direct_10ep/weights_epoch_10.pt`. Upstream saves per-epoch weights, no best/latest.)

**Config parity** (verified by diffing the run's resolved Hydra config against upstream defaults — `git status` in the clone is clean apart from `outputs/`/`lightning_logs/`): the *only* deltas are the overrides above, i.e. `output_model_name`, `img_size` 224→112, `max_epochs` 20→10, `precision` bf16→16-mixed, `+accumulate_grad_batches` 4, `batch_size` 128→32 (micro-batch; effective 128 with accumulation), `num_workers` 6→4, and the dataset path (same file content). Everything else is byte-for-byte upstream: seed 3072, 192-D, H=1/P=5, 8 flow steps, `flow_matching.source=standard_noise`, `source_noise_scale=1.0`, AdamW 5e-5/wd 1e-3, SIGReg 0.09, split 0.9 — so noise sampling (`randn_like` source, `t~U[0,1]`), loss, and architecture are exactly theirs. Two honest caveats: (1) micro-batch 32 vs 128 changes `BatchNorm1d` statistics in the projector (accumulation does not fix BN) — same setup as our port/Exp D runs, so direct-vs-port is apples-to-apples, but vs their paper batch it's a gap; (2) same-seed runs still diverge because SIGReg/flow noise draws are unseeded — that explains the ≤0.05 direct-vs-port gaps, not a code difference.

**Direct eval** (one-off script, not retained — mirrors our one-shot protocol exactly: 64 seqs, index RNG 0, noise seed 12345, oracle actions; handles upstream 4D `(B,T,1,D)` latents):

| Horizon | NFE=8 MSE | NFE=8 Cos | NFE=4 MSE | NFE=4 Cos |
|---------|----------|----------|----------|----------|
| 1-step | 0.677 ± 0.079 | 0.388 ± 0.071 | 0.652 ± 0.073 | 0.384 ± 0.074 |
| 3-step | 0.777 ± 0.002 | 0.253 ± 0.014 | 0.740 ± 0.004 | 0.251 ± 0.015 |
| 5-step | 0.867 ± 0.006 | 0.143 ± 0.015 | 0.818 ± 0.006 | 0.141 ± 0.015 |
| 10-step | 0.898 ± 0.011 | 0.104 ± 0.016 | 0.852 ± 0.009 | 0.101 ± 0.016 |
| 15-step | 0.964 ± 0.005 | 0.070 ± 0.007 | 0.919 ± 0.006 | 0.069 ± 0.007 |

Conditioning: NFE=8 action 1.000× / context 1.70×; NFE=4 action 1.000× / context 1.61×.

**Result**: The direct clone lands in the **same collapse band** as the port (port NFE=8: 0.716/0.805/0.830, act 1.00×, ctx 1.3–1.7×). Residual gaps (≤0.05 MSE) are expected run-to-run noise — both use seed 3072 but SIGReg draws unseeded random projections each step, so trajectories diverge. Port exonerated: the method itself, not our re-implementation, fails to condition under our data/settings (10ep, img 112, joint-from-scratch). The `noisy_current` source remains the only untested design difference.

### Paper-claims check (arXiv:2608.29029, "Flow-JEPA: Flow Matching for Robust Latent Dynamics in JEPA World Models")

**Their claims**: MPC planning success (CEM, 5 action blocks, terminal-latent cost, 50 rollouts), not latent MSE. Mean 86%→92% clean and 67%→86% under localized-Gaussian visual noise across Two-Room/Reacher/Push-T/OGBench-Cube. Best config: 8 Euler steps; Push-T F-JEPA uses 10 CEM iterations vs LeWM baseline's 30 (matched-budget ablation: LeWM-noisy 60% vs F-JEPA-noisy 90% at 10 iters).

**The critical detail (App. A.3, "Training details")**: *"For Push-T, the selected model initializes the visual encoder and projector from a trained LeWM checkpoint and keeps both components frozen during training; predictor dropout is set to 0."* Reacher uses LeWM init as trainable warm start; only the rest train joint-from-scratch. I.e., **their Push-T result uses the frozen-encoder recipe — exactly our Exp A configuration, the one setup where flow worked for us too** (0.037 vs 0.120 at 3-step, conditioning preserved). Our Exp E/E2 benchmark used joint-from-scratch (their config default), a configuration the paper itself does not claim works on Push-T. No contradiction.

**Remaining gaps before calling the repo broken**: (1) metric — we never ran their MPC eval (clean or noisy); planning only needs good goal-cost ranking, which can survive bad 1-step MSE; (2) budget — they train 20ep/img 224/true-batch 128 on an RTX 6000 Ada; we ran 10ep/img 112/eff-batch on a 4 GB 3050 Ti; (3) their robustness claim lives in the noisy regime, which we never tested. **Verdict: not safe to say the repo doesn't work.** The fair replication of their Push-T claim is frozen-LeWM-encoder init + dropout 0 + their MPC eval (clean + noisy) — that is the natural next experiment, and it coincides with the one flow configuration our own results endorse.

**Env-generality note**: our collapse evidence is PushT-only, so universal collapse is unproven — but note the paper's pattern is consistent with env-dependent difficulty: both manipulation envs (PushT, Reacher) use LeWM-encoder init (frozen / warm-start), while pure joint-from-scratch is only claimed on Two-Room (simple navigation, easy latents) and OGBench-Cube (where their gain is smallest: +8 noisy, clean 82 vs LeWM 74). Two cheap decisive tests: (1) their frozen PushT recipe + MPC eval — should work per both paper and Exp A; (2) joint-from-scratch on Two-Room — if that collapses too, the disease is general, not PushT-specific.

## Runtime & Memory

| Model | Peak GPU Allocated | Peak GPU Reserved |
|-------|-------------------|-------------------|
| LeWM (MSE) | ~1.08 GB | ~1.13 GB |
| Flow-LeWM | ~1.08 GB | ~1.13 GB |
| Shortcut-LeWM | ~1.08 GB | ~1.13 GB |
| Flow-Transformer joint (Exp D, 10ep) | ~1.14 GB | ~1.20 GB |
| Flow-JEPA repro (Exp E, 10ep, H=1/P=5) | ~1.56 GB | ~1.60 GB |
