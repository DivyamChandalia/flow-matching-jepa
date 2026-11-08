import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

try:
    from importlib.metadata import PackageNotFoundError, version as _md_version
    from packaging.version import Version
except ImportError:
    Version = None

if Version is not None:
    _MIN_SWM_VER = Version("0.1.1")
    try:
        _swm_ver = Version(_md_version("stable-worldmodel"))
    except PackageNotFoundError:
        _swm_ver = Version("0")
    if _swm_ver < _MIN_SWM_VER or not hasattr(swm.data, "load_dataset"):
        raise RuntimeError(
            "This repository requires stable-worldmodel==0.1.1 "
            "with swm.data.load_dataset available.\n"
            'Install with: python -m pip install --force-reinstall '
            '"stable-worldmodel[all]==0.1.1"'
        )

from module import SIGReg
from shortcut import (
    sample_finest_flow_batch,
    flow_xpred_loss,
    sample_shortcut_grid,
    shortcut_bootstrap_loss,
    _make_weight,
)
from utils import get_column_normalizer, get_img_preprocessor, GPUMetricsCallback, SaveCkptCallback


def limit_dataset(dataset, count, seed):
    count = min(count, len(dataset))
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:count].tolist()
    return torch.utils.data.Subset(dataset, indices)


def lewm_forward(self, batch, stage, cfg):
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def flow_forward(self, batch, stage, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :cfg.history_size]
    ctx_act = act_emb[:, :cfg.history_size]
    target = emb[:, 1:cfg.history_size + 1]

    # Baseline LeWM AR loss (unchanged — trains encoder + predictor exactly as before)
    ar_pred = self.model.predict_context(ctx_emb, ctx_act)
    ar_loss = (ar_pred - target).pow(2).mean()
    output["ar_loss"] = ar_loss.detach()

    # SIGReg (baseline — pushes encoder toward unit-variance latents)
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    # --- Flow branch (isolated from encoder) ---
    flow_grad_mode = cfg.objective.get("flow_grad_mode", "isolated")
    flow_weight = cfg.objective.get("flow_weight", 1.0)
    weighting = cfg.objective.get("flow_weighting", "original")
    force_tau0_prob = cfg.objective.get("force_tau0_prob", 0.0)

    # Flow target is always detached — flow cannot rewrite the representation.
    flow_target = (target - ar_pred).detach()

    # Flow context determines whether flow can influence the predictor.
    if flow_grad_mode == "frozen" or flow_grad_mode == "isolated":
        flow_context = ar_pred.detach()
    elif flow_grad_mode == "predictor":
        flow_context = ar_pred
    else:
        raise ValueError(f"Unknown flow_grad_mode: {flow_grad_mode}")

    sample = sample_finest_flow_batch(
        flow_target, cfg.objective.k_max,
        weighting=weighting, force_tau0_prob=force_tau0_prob,
    )
    pred_target = self.model.flow_predict(
        sample["x_t"], flow_context, sample["signal_idx"], sample["step_idx"]
    )
    flow_loss = flow_xpred_loss(pred_target, flow_target, sample["weight"])
    output["flow_loss"] = flow_loss.detach()

    lambd = cfg.loss.sigreg.weight
    output["loss"] = ar_loss + lambd * output["sigreg_loss"] + flow_weight * flow_loss
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def flow_transformer_forward(self, batch, stage, cfg):
    """Joint encoder + full-transformer flow training from scratch.

    The predictor IS the flow model: v(noisy, hist_emb, hist_act, t).
    Encoder gradients flow through both target and history latents (no detach).
    Loss = flow_loss + sigreg (no AR MSE loss).
    """
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    hist_emb = emb[:, :cfg.history_size]
    hist_act = act_emb[:, :cfg.history_size]
    target_emb = emb[:, 1:cfg.history_size + 1]

    flow_loss = self.model.flow_loss(hist_emb, hist_act, target_emb)
    output["flow_loss"] = flow_loss.detach()

    lambd = cfg.loss.sigreg.weight
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = flow_loss + lambd * output["sigreg_loss"]
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def fjepa_forward(self, batch, stage, cfg):
    """Faithful Flow-JEPA training: joint H=1 -> P trajectory flow + SIGReg.

    future actions act[:, H-1:H-1+P], targets emb[:, H:H+P]. No AR loss.
    """
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    H = cfg.history_size
    P = cfg.num_preds
    hist_emb = emb[:, :H]
    future_act = act_emb[:, H - 1:H - 1 + P]
    target_emb = emb[:, H:H + P]

    flow_loss = self.model.flow_loss(hist_emb, future_act, target_emb)
    output["flow_loss"] = flow_loss.detach()

    lambd = cfg.loss.sigreg.weight
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = flow_loss + lambd * output["sigreg_loss"]
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def shortcut_forward(self, batch, stage, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    B = emb.size(0)
    ctx_emb = emb[:, :cfg.history_size]
    ctx_act = act_emb[:, :cfg.history_size]
    target = emb[:, 1:cfg.history_size + 1]
    context = self.model.predict_context(ctx_emb, ctx_act)
    if cfg.objective.target_space == "latent":
        flow_target = target
    elif cfg.objective.target_space == "delta":
        flow_target = target - ctx_emb
    else:
        raise ValueError(cfg.objective.target_space)
    detach_enc = cfg.objective.get("bootstrap_detach_encoder", True)
    use_alternate = cfg.objective.alternate_batches and stage == "train"
    bootstrap_started = self.global_step >= cfg.objective.bootstrap_start_steps
    if use_alternate and bootstrap_started:
        use_shortcut = (self.global_step % 4 == 0)
    else:
        use_shortcut = False
    if use_shortcut:
        skw = sample_shortcut_grid(flow_target.shape[:2], cfg.objective.k_max, flow_target.device)
        shortcut_ctx = ctx_emb.detach() if detach_enc else ctx_emb
        shortcut_act = ctx_act
        shortcut_flow = flow_target.detach()
        shortcut_context = self.model.predict_context(shortcut_ctx, shortcut_act)
        noise = torch.randn_like(shortcut_flow)
        x_t = (1.0 - skw["tau"][..., None]) * noise + skw["tau"][..., None] * shortcut_flow
        shortcut_loss = shortcut_bootstrap_loss(
            self.model.flow_head, x_t, shortcut_context,
            skw["signal_idx"], skw["step_idx"],
            skw["tau"], skw["d"], cfg.objective.k_max,
        )
        output["shortcut_loss"] = shortcut_loss.detach()
        dynamics_loss = shortcut_loss
        output["shortcut_ratio"] = torch.tensor(1.0, device=shortcut_loss.device)
    elif cfg.objective.self_fraction > 0.0 and bootstrap_started and not use_alternate and stage == "train":
        if B < 2:
            raise ValueError(
                "Split-batch shortcut training requires batch_size >= 2. "
                "Use objective.alternate_batches=true for batch_size=1."
            )
        B_self = max(1, int(B * cfg.objective.self_fraction))
        B_emp = B - B_self
        sample = sample_finest_flow_batch(flow_target[:B_emp], cfg.objective.k_max)
        pred_target = self.model.flow_predict(
            sample["x_t"], context[:B_emp], sample["signal_idx"], sample["step_idx"]
        )
        flow_loss = flow_xpred_loss(pred_target, flow_target[:B_emp], sample["weight"])
        output["flow_loss"] = flow_loss.detach()
        shortcut_ctx = ctx_emb[B_emp:].detach() if detach_enc else ctx_emb[B_emp:]
        shortcut_act = ctx_act[B_emp:]
        shortcut_flow = flow_target[B_emp:].detach()
        shortcut_context = self.model.predict_context(shortcut_ctx, shortcut_act)
        noise = torch.randn_like(shortcut_flow)
        skw = sample_shortcut_grid(shortcut_flow.shape[:2], cfg.objective.k_max, shortcut_flow.device)
        tau = skw["tau"][..., None]
        x_t = (1.0 - tau) * noise + tau * shortcut_flow
        shortcut_loss = shortcut_bootstrap_loss(
            self.model.flow_head, x_t, shortcut_context,
            skw["signal_idx"], skw["step_idx"],
            skw["tau"], skw["d"], cfg.objective.k_max,
        )
        output["shortcut_loss"] = shortcut_loss.detach()
        dynamics_loss = (B_emp * flow_loss + B_self * shortcut_loss) / B
        output["shortcut_ratio"] = torch.tensor(B_self / B, device=flow_loss.device)
    else:
        sample = sample_finest_flow_batch(flow_target, cfg.objective.k_max)
        pred_target = self.model.flow_predict(
            sample["x_t"], context, sample["signal_idx"], sample["step_idx"]
        )
        dynamics_loss = flow_xpred_loss(pred_target, flow_target, sample["weight"])
        output["flow_loss"] = dynamics_loss.detach()
        output["shortcut_loss"] = torch.tensor(0.0, device=dynamics_loss.device)
        output["shortcut_ratio"] = torch.tensor(0.0, device=dynamics_loss.device)
    lambd = cfg.loss.sigreg.weight
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = dynamics_loss + lambd * output["sigreg_loss"]
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    local_dataset_dir = os.environ.get("LOCAL_DATASET_DIR")
    if local_dataset_dir:
        candidate = Path(local_dataset_dir) / dataset_name
        if not candidate.exists():
            candidate = Path(local_dataset_dir) / f"{dataset_name}.h5"
        if not candidate.exists():
            candidate = Path(local_dataset_dir) / f"{dataset_name}.lance"
        if candidate.exists():
            dataset_name = str(candidate)
    else:
        cache_dir = swm.data.utils.get_cache_dir()
        candidate = Path(cache_dir) / dataset_name
        if not candidate.exists():
            candidate = Path(cache_dir) / f"{dataset_name}.h5"
        if not candidate.exists():
            candidate = Path(cache_dir) / f"{dataset_name}.lance"
        if candidate.exists():
            dataset_name = str(candidate)
    dataset = swm.data.load_dataset(
        dataset_name, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    if cfg.get("debug_subset", {}).get("enabled", False):
        train_set = limit_dataset(train_set, cfg.debug_subset.train_examples, cfg.seed)
        val_set = limit_dataset(val_set, cfg.debug_subset.val_examples, cfg.seed + 1)

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    flow_grad_mode = cfg.get("objective", {}).get("flow_grad_mode", "isolated")
    if flow_grad_mode == "frozen":
        pretrained_path = cfg.objective.get("load_pretrained", None)
        assert pretrained_path is not None, "frozen mode requires load_pretrained path"
        state = torch.load(pretrained_path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        world_model.load_state_dict(state, strict=False)
        for name, param in world_model.named_parameters():
            if not name.startswith("flow_head"):
                param.requires_grad = False
        n_trainable = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
        print(f"[frozen] Trainable params: {n_trainable:,} (only flow_head)")

    objective_name = cfg.get("objective", {}).get("name", "mse")
    if objective_name == "mse":
        forward_fn = lewm_forward
    elif objective_name == "flow":
        forward_fn = flow_forward
    elif objective_name == "flow_transformer":
        forward_fn = flow_transformer_forward
    elif objective_name == "fjepa":
        forward_fn = fjepa_forward
    elif objective_name == "shortcut":
        forward_fn = shortcut_forward
    else:
        forward_fn = lewm_forward

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(forward_fn, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    metrics_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints')
    ) / cfg.output_model_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    gpu_metrics_path = metrics_dir / f"{cfg.output_model_name}_gpu_metrics.json"

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model,
    )
    gpu_metrics_callback = GPUMetricsCallback(output_path=gpu_metrics_path)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, gpu_metrics_callback],
        logger=logger,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
