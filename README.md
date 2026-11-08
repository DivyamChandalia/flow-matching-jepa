# Flow-Matching JEPA

**Can a JEPA world model's encoder and a flow-matching predictor be trained together?**

This repository extends [LeWorldModel](https://github.com/lucas-maes/le-wm) (Maes, Le Lidec et al., 2026), a JEPA world model trained end-to-end from pixels, by replacing or augmenting its deterministic predictor with flow matching. The short answer from these experiments: **on Push-T, only when the encoder is frozen.** When the encoder trains jointly with the flow objective, the predictor learns to ignore actions.

The original LeWorldModel README (installation, data, training, planning) is preserved in [`UPSTREAM_README.md`](UPSTREAM_README.md). All credit for the base model and codebase goes to its authors.

## Why this question

This came out of an earlier project, [Differentiable Flow World Models](https://github.com/DivyamChandalia/differentiable-flow-world-models).

I started from drone racing. A PPO policy I trained could fly through the first gates, but the competition simulator could not be run in parallel easily, so learning in it would be slow. A learned world model offered a way around that: a fast, parallel proxy for the simulator, learned from the simulator's own data. I tested the idea on Cartpole first. A JEPA-style world model predicted chunks of future latents, and the actor was trained by backpropagating value through those imagined chunks.

The reason for backpropagating through the model: if the world model knows what will happen next, it should be able to train the policy directly, without sampling random actions as MPC or policy-gradient methods do.

Predicting several timesteps in one pass was hard for the world model, so I tried flow matching. The flow model then ignored the actions, some form of collapse appeared, and the JEPA and flow objectives seemed to work against each other. Direct gradients also overestimated value in regions the model had not seen. Too many problems were tangled together to solve at once. So I isolated the world-model part: **getting flow matching and JEPA to train together**, on a strong existing JEPA world model, so that any fix could later be carried back to the chunked models. I wanted joint training specifically, because a jointly trained model can be used in any environment without a separate pretraining stage.

**Why flow matching.** Here I dropped chunking and kept plain one-step autoregressive prediction, so flow matching was the only change. The motivation is that the world is not fully predictable. In chaotic systems, small differences lead to very different outcomes. A deterministic predictor like LeWM's can only output one future, and under uncertainty it is pushed toward the average of the possible ones. Flow matching can represent a distribution over futures instead. A sample may not be the future that actually happens, but it should be a plausible one, and a policy trained on plausible futures may hold up better in the real world than one trained on averages. Most world models that sample multiple futures are trained to reconstruct pixels or tokens rather than to predict in a JEPA-style embedding space, so a generative JEPA world model seemed worth testing as a different approach.

## Setup

- **Base:** LeWorldModel, ViT-tiny encoder, 192-D latent, 112 px images, 3 history frames.
- **Data:** Push-T expert dataset (500 episodes, ~50,500 frames, frameskip 5).
- **Hardware:** one RTX 3050 Ti laptop GPU (4 GB VRAM).
- **Flow parameterization:** the flow heads (results 1–2) use x-prediction: the head predicts the clean target and sampling converts it to a velocity. The full flow transformer and the Flow-JEPA runs (results 3–4) use v-prediction, as Flow-JEPA does.
- **Evaluation:** offline latent rollout error on 64 held-out sequences (MSE and cosine to encoder targets at 1–15 steps), plus **shuffle ratios**. A shuffle ratio is how much the error grows when the actions (or the visual context) are shuffled across the batch. A ratio near 1.0× means the model ignores that input.

## Results

Full tables, commands and caveats are in [`results.md`](results.md).

### 1. Flow heads trained jointly ignore actions

Flow-matching and shortcut-forcing heads on top of LeWM's predictor, trained jointly for 100 epochs:

| Model | 1-step MSE | 5-step MSE | Action shuffle | Context shuffle |
|---|---:|---:|---:|---:|
| LeWM (MSE predictor) | 0.008 | 0.070 | **6.3×** | 225.7× |
| Flow head, latent target | 0.317 | 0.816 | 1.07× | 3.4× |
| Flow head, delta target | 0.086 | 0.369 | 1.00× | 19.0× |
| Shortcut head, delta target | 0.084 | 0.332 | 1.00× | 14.9× |

The deterministic predictor uses actions; every flow variant ignores them. Predicting deltas instead of latents helps the error, but not action conditioning.

### 2. Frozen encoder, residual flow: conditioning survives

Experiment A loads a trained LeWM, freezes the encoder and predictor, and trains only a flow head on the predictor's residual, `z_{t+1} - ARPredictor(z_t, a_t)`. Both models roll out with the last action repeated (the inference-time setting):

| Model (repeated actions) | 3-step MSE | 5-step MSE | 10-step MSE | Action shuffle |
|---|---:|---:|---:|---:|
| AR predictor | 0.120 | 0.272 | 0.538 | 8.1× |
| **AR + residual flow (4 steps)** | **0.037** | **0.068** | **0.197** | **5.5×** |

With the encoder frozen, the flow stays action-sensitive and corrects compounding rollout error, matching the teacher-forced baseline after 10 epochs.

### 3. It is not head capacity

Experiment D replaces the predictor with a full 6-layer flow transformer (the same parameter budget as LeWM's predictor), generating **one step at a time autoregressively**, and trains it jointly with the encoder from scratch. It collapses the same way: action shuffle **1.00×**, context shuffle 4.0×. Doubling the integration steps barely changes anything, so it is not solver error either. The failure is in joint training, not in the predictor's expressiveness.

### 4. Reproducing Flow-JEPA

[Flow-JEPA](https://github.com/HuoYanchen/Flow-JEPA) (arXiv:2608.29029) also combines flow matching with JEPA world models. Unlike Experiment D, it is **not autoregressive**: from one history frame and five future actions, a single flow integration generates all five future latents together. That is the one-pass chunked prediction I had attempted in the Cartpole project. I ported its predictor (verified bit-identical to the upstream code by loading the same weights) and also trained the upstream repository directly, both under the evaluation above. Trained jointly from scratch, both land in the same collapse band: action shuffle 1.00×, context shuffle ~1.3–1.7×.

This is **not** evidence that Flow-JEPA fails. Its paper reports planning success rather than latent error, used a larger training budget, and, importantly, states that **its Push-T model initializes the encoder from a trained LeWM checkpoint and keeps it frozen**. That is the same recipe as Experiment A, the one configuration that worked here. I did not run its planning evaluation.

Taken together, a one-step autoregressive flow (Experiment D) and a one-shot multi-step flow (Flow-JEPA) both lose action conditioning when trained jointly from scratch on Push-T, and both work with a frozen encoder. The rollout style is not what decides it. Training the encoder jointly is.

## What I think is happening

When the encoder trains with the flow objective, nothing stops the latent space from drifting toward something the flow can match without using the actions. A frozen encoder removes that freedom. The flow then has a fixed target it can only fit by using its conditioning.

## Next

- **EMA target encoder.** A frozen encoder works, but it needs a pretrained model first, which is what joint training was meant to avoid. If embedding drift is what breaks joint training, a slowly updated target encoder may sit between "frozen" and "fully joint". Planned, not yet run.
- **x-prediction with a full, jointly trained flow transformer.** My x-prediction runs used a small head and my full-transformer runs used v-prediction, so this combination is untested. The x-prediction head also lost action conditioning when trained jointly, so I don't expect it to fix the collapse on its own.
- Frozen-encoder Flow-JEPA on Push-T with its MPC planning evaluation, a fair replication of its claim.
- Joint-from-scratch on a simpler environment (Two-Room) to test whether the collapse is specific to Push-T.
- Carrying a working recipe back to chunked world models for policy learning.

## Limitations

- Push-T only, single seeds, and mostly 10-epoch runs for the ablations (the 100-epoch flow runs are no better).
- Offline latent-prediction metrics only. No planning or control results.
- Latent scales differ across separately trained encoders, so cross-run MSE is indicative. Cosine similarity gives the same ranking.

## Code map

| File | What it adds |
|---|---|
| `module.py` | `ShortcutFlowHead` (flow/shortcut head), `FlowTransformerPredictor` (Exp D), `FlowTrajectoryPredictor` (Flow-JEPA port, Exp E) |
| `jepa.py` | `ShortcutJEPA` (flow and shortcut variants), `FlowJEPA` (Exp D), `TrajectoryFlowJEPA` (Exp E) |
| `shortcut.py` | Shortcut-forcing utilities |
| `eval_dynamics.py` | Offline latent-rollout evaluation and shuffle-ratio diagnostics |
| `train.py` | Objectives for flow, shortcut, frozen-encoder residual flow and Flow-JEPA |
| `config/train/` | Model/objective configs for every experiment, plus a 4 GB-GPU config |
| `tests/` | Unit tests for heads, rollouts and the Flow-JEPA port |
| `results.md` | All experiments with commands, settings and caveats |

Install, data and training follow the [upstream instructions](UPSTREAM_README.md). Experiment commands are listed in [`results.md`](results.md).

## Citation

This work builds directly on LeWorldModel:

```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
