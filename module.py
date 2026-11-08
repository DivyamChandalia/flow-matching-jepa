import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class ShortcutFlowHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 2048, k_max: int = 8):
        super().__init__()
        assert k_max > 0 and (k_max & (k_max - 1)) == 0
        self.dim = dim
        self.k_max = k_max
        self.max_step_idx = int(math.log2(k_max))
        self.noisy_proj = nn.Linear(dim, dim)
        self.context_proj = nn.Linear(dim, dim)
        self.signal_embed = nn.Embedding(k_max + 1, dim)
        self.step_embed = nn.Embedding(self.max_step_idx + 1, dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, noisy_target, context, signal_idx, step_idx):
        x = self.noisy_proj(noisy_target)
        x = x + self.context_proj(context)
        x = x + self.signal_embed(signal_idx)
        x = x + self.step_embed(step_idx)
        return self.net(x)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal embedding for continuous diffusion/flow timesteps. (B,) -> (B, dim)"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = timesteps.float().reshape(-1, 1) * freqs.reshape(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class CrossAttention(nn.Module):
    """Multi-head cross-attention (queries from x, keys/values from context)."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context):
        x = self.norm_q(x)
        context = self.norm_kv(context)
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in self.to_kv(context).chunk(2, dim=-1))
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class FlowTransformerBlock(nn.Module):
    """Self + cross attention block modulated by flow time (AdaLN-zero)."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out_sa = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.to_out_ca = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, context, t_cond, causal=True):
        (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(t_cond).chunk(9, dim=-1)

        # self attention (causal over prediction tokens)
        h = modulate(self.norm1(x), shift_sa, scale_sa)
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        sa = F.scaled_dot_product_attention(
            q * self.scale, k, v, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = x + gate_sa * self.to_out_sa(rearrange(sa, "b h n d -> b n (h d)"))

        # cross attention to history + action context (full visibility)
        h = modulate(self.norm2(x), shift_ca, scale_ca)
        q = rearrange(self.to_q(h), "b n (h d) -> b h n d", h=self.heads)
        kv = self.to_kv(self.norm_kv(context)).chunk(2, dim=-1)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in kv)
        ca = F.scaled_dot_product_attention(
            q * self.scale, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + gate_ca * self.to_out_ca(rearrange(ca, "b h n d -> b n (h d)"))

        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FlowTransformerPredictor(nn.Module):
    """Full-transformer flow predictor: the AR predictor IS the flow model.

    forward(noisy_emb (B,P,D), hist_emb (B,H,D), act_emb (B,F,D), t (B,))
    returns velocity (B,P,D) for the Gaussian -> target flow.
    """

    def __init__(
        self,
        *,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        action_dim=None,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        causal_self_attention=True,
    ):
        super().__init__()
        action_dim = action_dim or input_dim
        output_dim = output_dim or input_dim
        self.hidden_dim = hidden_dim
        self.causal_self_attention = bool(causal_self_attention)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.hist_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.act_proj = nn.Linear(action_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList(
            [FlowTransformerBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, noisy_emb, hist_emb, act_emb, t):
        B, P, _ = noisy_emb.shape
        H = hist_emb.size(1)
        F_ = act_emb.size(1)
        dtype = noisy_emb.dtype
        device = noisy_emb.device

        # frame positions: history 0..H-1, predictions/actions continue after
        pred_pos = torch.arange(H, H + P, device=device, dtype=torch.float32)
        hist_pos = torch.arange(H, device=device, dtype=torch.float32)
        act_pos = torch.arange(H, H + F_, device=device, dtype=torch.float32)
        pred_pe = timestep_embedding(pred_pos, self.hidden_dim).to(dtype=dtype)
        hist_pe = timestep_embedding(hist_pos, self.hidden_dim).to(dtype=dtype)
        act_pe = timestep_embedding(act_pos, self.hidden_dim).to(dtype=dtype)

        x = self.input_proj(noisy_emb) + pred_pe.unsqueeze(0)
        hist_ctx = self.hist_proj(hist_emb) + hist_pe.unsqueeze(0)
        act_ctx = self.act_proj(act_emb) + act_pe.unsqueeze(0)
        context = torch.cat([hist_ctx, act_ctx], dim=1)
        x = self.dropout(x)

        t = t.reshape(-1)
        assert t.size(0) == B, f"t must have shape (B,), got {tuple(t.shape)}"
        t_cond = self.time_mlp(timestep_embedding(t, self.hidden_dim).to(dtype=dtype)).unsqueeze(1)

        causal = self.causal_self_attention and P > 1
        for layer in self.layers:
            # broadcast t_cond over prediction tokens
            tc = t_cond.expand(B, P, -1)
            x = layer(x, context, tc, causal=causal)

        return self.output_proj(self.norm(x))


class RMSNormMLP(nn.Module):
    """Token-local nonlinear projection without batch-dependent statistics."""

    def __init__(self, input_dim, hidden_dim, output_dim=None, eps=1e-6, act_fn=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.RMSNorm(hidden_dim, eps=eps),
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class FJTrajectoryBlock(nn.Module):
    """Faithful port of Flow-JEPA's FlowTrajectoryBlock (self + cross + MLP)."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out_sa = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.to_out_ca = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def _split(self, t):
        return rearrange(t, "b n (h d) -> b h n d", h=self.heads)

    def forward(self, x, context, t_cond, self_mask=None, cross_mask=None):
        (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(t_cond).chunk(9, dim=-1)
        drop = self.dropout if self.training else 0.0

        h = modulate(self.norm1(x), shift_sa, scale_sa)
        q, k, v = (self._split(t) for t in self.to_qkv(h).chunk(3, dim=-1))
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self_mask is not None:
            scores = scores.masked_fill(~self_mask.bool()[None, None], -torch.finfo(scores.dtype).max)
        sa = (scores.softmax(dim=-1))
        sa = F.dropout(sa, p=drop, training=self.training)
        x = x + gate_sa * self.to_out_sa(rearrange(torch.matmul(sa, v), "b h n d -> b n (h d)"))

        h = modulate(self.norm2(x), shift_ca, scale_ca)
        q = self._split(self.to_q(h))
        k, v = (self._split(t) for t in self.to_kv(context).chunk(2, dim=-1))
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if cross_mask is not None:
            scores = scores.masked_fill(~cross_mask.bool()[None, None], -torch.finfo(scores.dtype).max)
        ca = scores.softmax(dim=-1)
        ca = F.dropout(ca, p=drop, training=self.training)
        x = x + gate_ca * self.to_out_ca(rearrange(torch.matmul(ca, v), "b h n d -> b n (h d)"))

        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FlowTrajectoryPredictor(nn.Module):
    """Faithful port of Flow-JEPA's FlowTrajectoryPredictor (K=1 frame tokens).

    forward(noisy_emb (B,P,D), hist_emb (B,H,D), act_emb (B,F,A), t (B,))
    returns velocity (B,P,D).
    """

    def __init__(
        self,
        *,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        action_dim=None,
        output_dim=None,
        output_proj_hidden_dim=None,
        causal_self_attention=False,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        action_dim = action_dim or input_dim
        output_dim = output_dim or input_dim
        self.hidden_dim = hidden_dim
        self.causal_self_attention = bool(causal_self_attention)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.context_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList(
            [FJTrajectoryBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = RMSNormMLP(
            hidden_dim, output_proj_hidden_dim or 4 * hidden_dim, output_dim
        )

    def _positional_embedding(self, positions, dtype):
        return timestep_embedding(positions, self.hidden_dim).to(dtype=dtype)

    def _build_cross_mask(self, hist_len, pred_len, action_len, device):
        query_future_idx = torch.arange(pred_len, device=device)
        hist_mask = torch.ones(pred_len, hist_len, device=device, dtype=torch.bool)
        action_idx = torch.arange(action_len, device=device)
        if pred_len == 1 and action_len > 1:
            action_mask = torch.ones(pred_len, action_len, device=device, dtype=torch.bool)
        else:
            action_mask = action_idx[None, :] <= query_future_idx[:, None]
        return torch.cat([hist_mask, action_mask], dim=-1)

    def _build_causal_self_mask(self, pred_len, device):
        frame_index = torch.arange(pred_len, device=device)
        return frame_index[None, :] <= frame_index[:, None]

    def forward(self, noisy_emb, hist_emb, act_emb, t):
        B, P, _ = noisy_emb.shape
        H = hist_emb.size(1)
        F_ = act_emb.size(1)
        dtype = noisy_emb.dtype
        device = noisy_emb.device

        action_pos = torch.arange(H, H + F_, device=device, dtype=torch.float32)
        if P == F_:
            pred_pos = action_pos
        elif P == 1:
            pred_pos = action_pos[-1:].clone()
        else:
            pred_pos = action_pos[:P]
        pred_pos_emb = self._positional_embedding(pred_pos, dtype)
        action_pos_emb = self._positional_embedding(action_pos, dtype)

        x = self.input_proj(noisy_emb) + pred_pos_emb.unsqueeze(0)
        x = self.dropout(x)
        hist_ctx = self.context_proj(hist_emb)
        act_ctx = self.action_proj(act_emb) + action_pos_emb.unsqueeze(0)
        context = torch.cat([hist_ctx, act_ctx], dim=1)

        t = t.reshape(-1)
        assert t.size(0) == B, f"t must have shape (B,), got {tuple(t.shape)}"
        t_cond = self.time_mlp(timestep_embedding(t, self.hidden_dim).to(dtype=dtype)).unsqueeze(1)
        t_cond = t_cond.expand(B, P, -1)

        self_mask = self._build_causal_self_mask(P, device) if self.causal_self_attention else None
        cross_mask = self._build_cross_mask(H, P, F_, device)
        for layer in self.layers:
            x = layer(x, context, t_cond, self_mask=self_mask, cross_mask=cross_mask)
        return self.output_proj(self.norm(x))
