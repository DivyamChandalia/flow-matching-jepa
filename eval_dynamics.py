import json
import os
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig

import stable_worldmodel as swm
from utils import get_column_normalizer, get_img_preprocessor


def collate_sequences(dataset, indices, device):
    rows = [dataset[idx] for idx in indices]
    pixels = torch.stack([row["pixels"] for row in rows]).to(device)
    actions = torch.stack([row["action"] for row in rows]).to(device)
    return pixels, actions


def _flow_single_step(model, ctx_emb, ctx_act, base_state, noise, nfe):
    """One denoised next latent. Supports FlowJEPA (predictor-is-flow) and
    ShortcutJEPA (AR context + small flow head) models."""
    if hasattr(model, "sample_next_flow"):
        return model.sample_next_flow(ctx_emb, ctx_act, noise=noise, nfe=nfe)
    context = model.predict_context(ctx_emb, ctx_act)[:, -1:]
    return model.sample_next_from_context(
        context=context, base_state=base_state, noise=noise, nfe=nfe,
    )


def _is_flow_model(model):
    return (
        hasattr(model, "sample_next_from_context")
        or hasattr(model, "sample_next_flow")
        or hasattr(model, "flow_source")
    )


def _ar_rollout(model, ctx_emb, ctx_act, steps, history_size=3):
    """Autoregressive rollout using AR predictor only (no flow)."""
    pred_embs = []
    for t in range(steps):
        _ctx_emb = ctx_emb[:, -history_size:]
        _ctx_act = ctx_act[:, -history_size:]
        ar_pred = model.predict_context(_ctx_emb, _ctx_act)[:, -1:]
        pred_embs.append(ar_pred)
        ctx_emb = torch.cat([ctx_emb, ar_pred], dim=1)
        ctx_act = torch.cat([ctx_act, ctx_act[:, -1:]], dim=1)
    return torch.cat(pred_embs, dim=1)


def evaluate_ar_dynamics(model, dataset, horizons, noise_seed=12345, num_samples=64, device="cuda", history_size=3):
    """Evaluate AR-only (deterministic) latent dynamics without flow."""
    model = model.to(device).eval()
    model.requires_grad_(False)
    all_metrics = {h: {"mse": [], "cos_sim": []} for h in horizons}
    rnd = np.random.RandomState(0)
    total = len(dataset)
    batch_size = min(32, num_samples)
    for start in range(0, num_samples, batch_size):
        batch_indices = rnd.randint(0, total, batch_size).tolist()
        pixels, action = collate_sequences(dataset, batch_indices, device)
        info = {"pixels": pixels, "action": action}
        info = model.encode(info)
        emb = info["emb"]
        act_emb = info["act_emb"]
        for h in horizons:
            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            pred_embs = _ar_rollout(model, ctx_emb, ctx_act, h, history_size)
            true_embs = emb[:, history_size:history_size + h]
            if true_embs.size(1) == 0:
                continue
            mse = (pred_embs - true_embs).pow(2).mean().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                pred_embs.reshape(-1, pred_embs.size(-1)),
                true_embs.reshape(-1, true_embs.size(-1)),
            ).mean().item()
            all_metrics[h]["mse"].append(mse)
            all_metrics[h]["cos_sim"].append(cos_sim)
    results = {}
    for h in horizons:
        if all_metrics[h]["mse"]:
            results[h] = {
                "mse_mean": float(np.mean(all_metrics[h]["mse"])),
                "mse_std": float(np.std(all_metrics[h]["mse"])),
                "cos_sim_mean": float(np.mean(all_metrics[h]["cos_sim"])),
                "cos_sim_std": float(np.std(all_metrics[h]["cos_sim"])),
            }
    return results


def evaluate_ar_conditioning_shuffle(model, dataset, noise_seed=12345, num_samples=32, device="cuda", history_size=3):
    """Action/context shuffle test using AR predictor output."""
    model = model.to(device).eval()
    model.requires_grad_(False)
    normal_losses = []
    shuffle_action_losses = []
    shuffle_context_losses = []
    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed)
    rnd = np.random.RandomState(0)
    total = len(dataset)
    batch_size = min(32, num_samples)
    for start in range(0, num_samples, batch_size):
        batch_indices = rnd.randint(0, total, batch_size).tolist()
        pixels, action = collate_sequences(dataset, batch_indices, device)
        B = pixels.size(0)
        info = {"pixels": pixels, "action": action}
        info = model.encode(info)
        emb = info["emb"]
        act_emb = info["act_emb"]
        ctx_emb = emb[:, :history_size]
        ctx_act = act_emb[:, :history_size]
        true_emb = emb[:, history_size:history_size + 1]
        noise = torch.randn(B, 1, emb.size(-1), generator=generator, device=device)

        if hasattr(model, "predict_context"):
            ar_pred = model.predict_context(ctx_emb, ctx_act)[:, -1:]
        else:
            ar_pred = model.predict(ctx_emb, ctx_act)[:, -1:]
        normal_loss = (ar_pred - true_emb).pow(2).mean().item()
        normal_losses.append(normal_loss)

        perm = torch.randperm(B, device=device)
        ctx_act_shuf = ctx_act[perm]
        if hasattr(model, "predict_context"):
            ar_pred = model.predict_context(ctx_emb, ctx_act_shuf)[:, -1:]
        else:
            ar_pred = model.predict(ctx_emb, ctx_act_shuf)[:, -1:]
        shuffle_action_losses.append((ar_pred - true_emb).pow(2).mean().item())

        perm = torch.randperm(B, device=device)
        ctx_emb_shuf = ctx_emb[perm]
        if hasattr(model, "predict_context"):
            ar_pred = model.predict_context(ctx_emb_shuf, ctx_act)[:, -1:]
        else:
            ar_pred = model.predict(ctx_emb_shuf, ctx_act)[:, -1:]
        shuffle_context_losses.append((ar_pred - true_emb).pow(2).mean().item())

    normal = float(np.mean(normal_losses))
    shuffle_act = float(np.mean(shuffle_action_losses))
    shuffle_ctx = float(np.mean(shuffle_context_losses))
    return {
        "normal_loss": normal,
        "shuffled_action_loss": shuffle_act,
        "shuffled_context_loss": shuffle_ctx,
        "action_shuffle_ratio": shuffle_act / normal if normal > 0 else 1.0,
        "context_shuffle_ratio": shuffle_ctx / normal if normal > 0 else 1.0,
    }


def evaluate_flow_dynamics(model, dataset, horizons, noise_seed=12345, num_samples=64, device="cuda", history_size=3, nfe=4):
    """Evaluate latent dynamics. Uses flow sampling if available, else AR predictor."""
    is_flow = _is_flow_model(model)
    is_traj = hasattr(model, "flow_source")
    model = model.to(device).eval()
    model.requires_grad_(False)
    all_metrics = {h: {"mse": [], "cos_sim": []} for h in horizons}
    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed)
    rnd = np.random.RandomState(0)
    total = len(dataset)
    batch_size = min(32, num_samples)
    for start in range(0, num_samples, batch_size):
        batch_indices = rnd.randint(0, total, batch_size).tolist()
        pixels, action = collate_sequences(dataset, batch_indices, device)
        B = pixels.size(0)
        info = {"pixels": pixels, "action": action}
        info = model.encode(info)
        emb = info["emb"]
        act_emb = info["act_emb"]
        for h in horizons:
            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            if is_traj:
                # one-shot horizon prediction with oracle future actions
                fut_act = act_emb[:, history_size - 1:history_size - 1 + h]
                noise = torch.randn(B, h, emb.size(-1), generator=generator, device=device)
                pred_embs = model.predict(ctx_emb, fut_act, horizon=h, noise=noise, num_steps=nfe)
                true_embs = emb[:, history_size:history_size + h]
                if true_embs.size(1) == 0:
                    continue
                mse = (pred_embs - true_embs).pow(2).mean().item()
                cos_sim = torch.nn.functional.cosine_similarity(
                    pred_embs.reshape(-1, pred_embs.size(-1)),
                    true_embs.reshape(-1, true_embs.size(-1)),
                ).mean().item()
                all_metrics[h]["mse"].append(mse)
                all_metrics[h]["cos_sim"].append(cos_sim)
                continue
            pred_embs = []
            for t in range(h):
                _ctx_emb = ctx_emb[:, -history_size:]
                _ctx_act = ctx_act[:, -history_size:]
                noise = torch.randn(B, 1, emb.size(-1), generator=generator, device=device)
                if is_flow:
                    pred = _flow_single_step(
                        model, _ctx_emb, _ctx_act, _ctx_emb[:, -1:], noise, nfe,
                    )
                else:
                    pred = model.predict(_ctx_emb, _ctx_act)[:, -1:]
                pred_embs.append(pred)
                ctx_emb = torch.cat([ctx_emb, pred], dim=1)
                next_act = act_emb[:, history_size + t:history_size + t + 1]
                ctx_act = torch.cat([ctx_act, next_act], dim=1)
            pred_embs = torch.cat(pred_embs, dim=1)
            true_embs = emb[:, history_size:history_size + h]
            if true_embs.size(1) == 0:
                continue
            mse = (pred_embs - true_embs).pow(2).mean().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                pred_embs.reshape(-1, pred_embs.size(-1)),
                true_embs.reshape(-1, true_embs.size(-1)),
            ).mean().item()
            all_metrics[h]["mse"].append(mse)
            all_metrics[h]["cos_sim"].append(cos_sim)
    results = {}
    for h in horizons:
        if all_metrics[h]["mse"]:
            results[h] = {
                "mse_mean": float(np.mean(all_metrics[h]["mse"])),
                "mse_std": float(np.std(all_metrics[h]["mse"])),
                "cos_sim_mean": float(np.mean(all_metrics[h]["cos_sim"])),
                "cos_sim_std": float(np.std(all_metrics[h]["cos_sim"])),
            }
    return results


def evaluate_flow_conditioning_shuffle(model, dataset, noise_seed=12345, num_samples=32, device="cuda", history_size=3, nfe=4):
    """Action/context shuffle test. Uses flow if available, else AR predictor."""
    is_flow = _is_flow_model(model)
    is_traj = hasattr(model, "flow_source")
    model = model.to(device).eval()
    model.requires_grad_(False)
    normal_losses = []
    shuffle_action_losses = []
    shuffle_context_losses = []
    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed)
    rnd = np.random.RandomState(0)
    total = len(dataset)
    batch_size = min(32, num_samples)
    for start in range(0, num_samples, batch_size):
        batch_indices = rnd.randint(0, total, batch_size).tolist()
        pixels, action = collate_sequences(dataset, batch_indices, device)
        B = pixels.size(0)
        info = {"pixels": pixels, "action": action}
        info = model.encode(info)
        emb = info["emb"]
        act_emb = info["act_emb"]
        ctx_emb = emb[:, :history_size]
        ctx_act = act_emb[:, :history_size]
        true_emb = emb[:, history_size:history_size + 1]
        noise = torch.randn(B, 1, emb.size(-1), generator=generator, device=device)
        if is_traj:
            # one-shot single-step prediction (P=1)
            fut_act = ctx_act[:, -1:]
            pred = model.predict(
                ctx_emb, fut_act, horizon=1,
                noise=noise.expand(B, 1, emb.size(-1)), num_steps=nfe,
            )
        elif is_flow:
            pred = _flow_single_step(model, ctx_emb, ctx_act, ctx_emb[:, -1:], noise, nfe)
        else:
            pred = model.predict(ctx_emb, ctx_act)[:, -1:]
        normal_loss = (pred - true_emb).pow(2).mean().item()
        normal_losses.append(normal_loss)

        perm = torch.randperm(B, device=device)
        ctx_act_shuf = ctx_act[perm]
        if is_traj:
            pred = model.predict(
                ctx_emb, ctx_act_shuf[:, -1:], horizon=1,
                noise=noise.expand(B, 1, emb.size(-1)), num_steps=nfe,
            )
        elif is_flow:
            pred = _flow_single_step(model, ctx_emb, ctx_act_shuf, ctx_emb[:, -1:], noise, nfe)
        else:
            pred = model.predict(ctx_emb, ctx_act_shuf)[:, -1:]
        shuffle_action_losses.append((pred - true_emb).pow(2).mean().item())

        perm = torch.randperm(B, device=device)
        ctx_emb_shuf = ctx_emb[perm]
        if is_traj:
            pred = model.predict(
                ctx_emb_shuf, ctx_act, horizon=1,
                noise=noise.expand(B, 1, emb.size(-1)), num_steps=nfe,
            )
        elif is_flow:
            pred = _flow_single_step(model, ctx_emb_shuf, ctx_act, ctx_emb_shuf[:, -1:], noise, nfe)
        else:
            pred = model.predict(ctx_emb_shuf, ctx_act)[:, -1:]
        shuffle_context_losses.append((pred - true_emb).pow(2).mean().item())

    normal = float(np.mean(normal_losses))
    shuffle_act = float(np.mean(shuffle_action_losses))
    shuffle_ctx = float(np.mean(shuffle_context_losses))
    return {
        "normal_loss": normal,
        "shuffled_action_loss": shuffle_act,
        "shuffled_context_loss": shuffle_ctx,
        "action_shuffle_ratio": shuffle_act / normal if normal > 0 else 1.0,
        "context_shuffle_ratio": shuffle_ctx / normal if normal > 0 else 1.0,
    }


def evaluate_noise_sensitivity(model, dataset, num_samples=16, device="cuda", history_size=3, nfe=4):
    """Measure how much flow predictions vary with different noise."""
    model = model.to(device).eval()
    model.requires_grad_(False)
    noise_effects_1step = []
    pred_variances_8 = []
    rnd = np.random.RandomState(0)
    total = len(dataset)
    batch_size = min(8, num_samples)
    for start in range(0, num_samples, batch_size):
        batch_indices = rnd.randint(0, total, batch_size).tolist()
        pixels, action = collate_sequences(dataset, batch_indices, device)
        B = pixels.size(0)
        info = {"pixels": pixels, "action": action}
        info = model.encode(info)
        emb = info["emb"]
        act_emb = info["act_emb"]
        ctx_emb = emb[:, :history_size]
        ctx_act = act_emb[:, :history_size]
        base_state = ctx_emb[:, -1:]
        noise_1 = torch.randn(B, 1, emb.size(-1), device=device)
        noise_2 = torch.randn(B, 1, emb.size(-1), device=device)
        pred_1 = _flow_single_step(model, ctx_emb, ctx_act, base_state, noise_1, nfe)
        pred_2 = _flow_single_step(model, ctx_emb, ctx_act, base_state, noise_2, nfe)
        noise_effects_1step.append((pred_1 - pred_2).pow(2).mean().item())
        all_preds = []
        for _ in range(8):
            n = torch.randn(B, 1, emb.size(-1), device=device)
            p = _flow_single_step(model, ctx_emb, ctx_act, base_state, n, nfe)
            all_preds.append(p)
        stacked = torch.stack(all_preds, dim=0)
        pred_variances_8.append(stacked.var(dim=0).mean().item())
    return {
        "mean_noise_effect_1step": float(np.mean(noise_effects_1step)),
        "pred_variance_across_8_samples": float(np.mean(pred_variances_8)),
    }


def evaluate_action_sensitivity(model, dataset, num_samples=16, device="cuda", history_size=3, nfe=4):
    """Direct action sensitivity for both AR and flow outputs."""
    model = model.to(device).eval()
    model.requires_grad_(False)
    has_ar = hasattr(model, "predict_context")
    ar_action_effects = []
    flow_action_effects = []
    rnd = np.random.RandomState(0)
    total = len(dataset)
    batch_size = min(8, num_samples)
    for start in range(0, num_samples, batch_size):
        batch_indices = rnd.randint(0, total, batch_size).tolist()
        pixels, action = collate_sequences(dataset, batch_indices, device)
        B = pixels.size(0)
        info = {"pixels": pixels, "action": action}
        info = model.encode(info)
        emb = info["emb"]
        act_emb = info["act_emb"]
        ctx_emb = emb[:, :history_size]
        ctx_act = act_emb[:, :history_size]
        noise = torch.randn(B, 1, emb.size(-1), device=device)
        perm = torch.randperm(B, device=device)

        if has_ar:
            ar_correct = model.predict_context(ctx_emb, ctx_act)[:, -1:]
            ar_shuffled = model.predict_context(ctx_emb, ctx_act[perm])[:, -1:]
            ar_action_effects.append((ar_correct - ar_shuffled).pow(2).mean().item())

        flow_correct = _flow_single_step(
            model, ctx_emb, ctx_act, ctx_emb[:, -1:], noise, nfe,
        )
        flow_shuffled = _flow_single_step(
            model, ctx_emb, ctx_act[perm], ctx_emb[:, -1:], noise, nfe,
        )
        flow_action_effects.append((flow_correct - flow_shuffled).pow(2).mean().item())

    return {
        "ar_action_output_effect": float(np.mean(ar_action_effects)) if ar_action_effects else None,
        "flow_action_output_effect": float(np.mean(flow_action_effects)),
    }


def _resolve_dataset_path(name, cfg):
    local_dir = os.environ.get("LOCAL_DATASET_DIR")
    if local_dir:
        candidates = [Path(local_dir) / name, Path(local_dir) / f"{name}.h5", Path(local_dir) / f"{name}.lance"]
    else:
        cache_dir = swm.data.utils.get_cache_dir()
        candidates = [Path(cache_dir) / name, Path(cache_dir) / f"{name}.h5", Path(cache_dir) / f"{name}.lance"]
    for c in candidates:
        if c.exists():
            return str(c)
    return name


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    history_size = cfg.eval.get("history_size", 3)
    horizons = [1, 3, 5, 10, 15]
    num_steps = history_size + max(horizons)

    dataset_name = _resolve_dataset_path(cfg.eval.dataset_name, cfg)
    dataset = swm.data.load_dataset(
        dataset_name,
        num_steps=num_steps,
        frameskip=cfg.eval.get("dynamics_frameskip", 5),
        keys_to_load=["pixels", "action"],
        keys_to_cache=cfg.dataset.keys_to_cache,
    )

    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.eval.get("img_size", 224))]
    for col in dataset.column_names:
        if col.startswith("pixels"):
            continue
        normalizer = get_column_normalizer(dataset, col, col)
        transforms.append(normalizer)
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = swm.wm.utils.load_pretrained(cfg.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)

    is_flow_model = _is_flow_model(model)
    has_ar = hasattr(model, "predict_context")
    has_step_api = (
        hasattr(model, "predict_context")
        or hasattr(model, "sample_next_from_context")
        or hasattr(model, "sample_next_flow")
    )
    if is_flow_model:
        sampling_cfg = cfg.get("sampling", {})
        model.configure_sampling(
            nfe=sampling_cfg.get("nfe", 4),
            num_model_samples=sampling_cfg.get("num_model_samples", 1),
            seed=sampling_cfg.get("seed", 0),
        )

    nfe = cfg.get("nfe", 4)
    use_nfe = getattr(cfg, "nfe", 4)

    results = {}

    if is_flow_model:
        if has_ar:
            print("=== AR (deterministic) dynamics ===")
            ar_dynamics = evaluate_ar_dynamics(
                model, dataset, horizons=horizons, noise_seed=12345, num_samples=64, device=device,
            )
            print(json.dumps(ar_dynamics, indent=2, default=str))
            results["ar_dynamics"] = ar_dynamics

            print("=== AR conditioning shuffle ===")
            ar_shuffle = evaluate_ar_conditioning_shuffle(
                model, dataset, noise_seed=12345, num_samples=32, device=device,
            )
            print(json.dumps(ar_shuffle, indent=2, default=str))
            results["ar_conditioning"] = ar_shuffle

        print(f"=== Flow dynamics (NFE={use_nfe}) ===")
        flow_dynamics = evaluate_flow_dynamics(
            model, dataset, horizons=horizons, noise_seed=12345, num_samples=64, device=device, nfe=use_nfe,
            history_size=history_size,
        )
        print(json.dumps(flow_dynamics, indent=2, default=str))
        results["flow_dynamics"] = flow_dynamics

        print("=== Flow conditioning shuffle ===")
        flow_shuffle = evaluate_flow_conditioning_shuffle(
            model, dataset, noise_seed=12345, num_samples=32, device=device, nfe=use_nfe,
            history_size=history_size,
        )
        print(json.dumps(flow_shuffle, indent=2, default=str))
        results["flow_conditioning"] = flow_shuffle

        print("=== Noise sensitivity ===")
        if has_step_api:
            noise = evaluate_noise_sensitivity(
                model, dataset, num_samples=16, device=device, nfe=use_nfe,
            )
        else:
            noise = {"note": "single-step API unsupported for one-shot trajectory model"}
        print(json.dumps(noise, indent=2, default=str))
        results["noise_sensitivity"] = noise

        print("=== Action sensitivity ===")
        if has_step_api:
            action_sens = evaluate_action_sensitivity(
                model, dataset, num_samples=16, device=device, nfe=use_nfe,
            )
        else:
            action_sens = {"note": "single-step API unsupported for one-shot trajectory model"}
        print(json.dumps(action_sens, indent=2, default=str))
        results["action_sensitivity"] = action_sens

        if has_ar:
            ar_1 = ar_dynamics.get("1", {}).get("mse_mean", 0)
            flow_1 = flow_dynamics.get("1", {}).get("mse_mean", 0)
            flow_gain = (ar_1 - flow_1) / ar_1 if ar_1 > 0 else 0
            results["flow_gain"] = flow_gain
            print(f"=== Flow gain (1-step): {flow_gain:.4f} (positive = flow improves over AR) ===")

            print("=== AR vs Flow comparison ===")
            ar_act_ratio = ar_shuffle.get("action_shuffle_ratio", 0)
            flow_act_ratio = flow_shuffle.get("action_shuffle_ratio", 0)
            print(f"  AR action ratio: {ar_act_ratio:.4f}")
            print(f"  Flow action ratio: {flow_act_ratio:.4f}")
    else:
        print("=== Deterministic (MSE) dynamics ===")
        results["dynamics"] = evaluate_flow_dynamics(
            model, dataset, horizons=horizons, noise_seed=12345, num_samples=64, device=device, nfe=1,
        )
        print(json.dumps(results["dynamics"], indent=2, default=str))
        results["conditioning"] = evaluate_flow_conditioning_shuffle(
            model, dataset, noise_seed=12345, num_samples=32, device=device, nfe=1,
        )
        print(json.dumps(results["conditioning"], indent=2, default=str))

    results_file = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent if cfg.policy != "random" else Path.cwd()
    results_file = results_file / "dynamics_eval.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    run()
