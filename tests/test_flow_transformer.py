import torch

from jepa import FlowJEPA
from module import FlowTransformerPredictor


class MockEncoder(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.config = type("obj", (object,), {"hidden_size": dim})()
        self.register_buffer("_fixed_emb", torch.randn(256, 1, dim))

    def forward(self, pixels, interpolate_pos_encoding=True):
        B = pixels.size(0)
        idx = torch.arange(B) % self._fixed_emb.size(0)

        class MockOutput:
            last_hidden_state = self._fixed_emb[idx]

        return MockOutput()


class MockActionEncoder(torch.nn.Module):
    def __init__(self, emb_dim=32):
        super().__init__()
        self.proj = torch.nn.Linear(4, emb_dim)

    def forward(self, x):
        return self.proj(x.float())


def make_model(dim=32):
    encoder = MockEncoder(dim)
    predictor = FlowTransformerPredictor(
        depth=2, heads=4, mlp_dim=64, input_dim=dim, hidden_dim=dim,
        action_dim=dim, output_dim=dim, dim_head=16,
    )
    action_encoder = MockActionEncoder(dim)
    return FlowJEPA(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        projector=torch.nn.Identity(), num_flow_steps=4,
    )


def test_predictor_velocity_shape():
    torch.manual_seed(0)
    B, H, P, D = 2, 3, 3, 32
    pred = FlowTransformerPredictor(
        depth=2, heads=4, mlp_dim=64, input_dim=D, hidden_dim=D,
        action_dim=D, output_dim=D, dim_head=16,
    )
    noisy = torch.randn(B, P, D)
    hist = torch.randn(B, H, D)
    act = torch.randn(B, H, D)
    t = torch.rand(B)
    v = pred(noisy, hist, act, t)
    assert v.shape == (B, P, D)
    assert v.isfinite().all()


def test_flow_loss_finite_and_joint_grads():
    torch.manual_seed(0)
    model = make_model()
    hist = torch.randn(2, 3, 32, requires_grad=True)
    act = torch.randn(2, 3, 32, requires_grad=True)
    target = torch.randn(2, 3, 32, requires_grad=True)
    loss = model.flow_loss(hist, act, target)
    assert loss.isfinite()
    loss.backward()
    # joint training: grads must reach history, action-window and target latents
    assert hist.grad is not None and hist.grad.isfinite().all()
    assert target.grad is not None and target.grad.isfinite().all()
    assert any(p.grad is not None for p in model.predictor.parameters())


def test_sample_next_flow_shape():
    torch.manual_seed(0)
    model = make_model()
    hist = torch.randn(2, 3, 32)
    act = torch.randn(2, 3, 32)
    out = model.sample_next_flow(hist, act, nfe=2)
    assert out.shape == (2, 1, 32)
    assert out.isfinite().all()


def test_rollout_and_cost_deterministic():
    torch.manual_seed(0)
    model = make_model()
    B, S, H, T = 2, 3, 3, 6
    info = {
        "pixels": torch.randn(B, H, 3, 16, 16),
        "goal": torch.randn(B, 1, 3, 16, 16),
        "action": torch.randn(B, H, 4),
    }
    actions = torch.randn(B, S, T, 4)
    model.configure_sampling(nfe=2, seed=123)
    c1 = model.get_cost({k: v.clone() for k, v in info.items()}, actions.clone())
    c2 = model.get_cost({k: v.clone() for k, v in info.items()}, actions.clone())
    assert c1.shape == (B, S)
    assert torch.allclose(c1, c2), "planning cost must be deterministic for fixed seed"
    model.configure_sampling(seed=999)
    c3 = model.get_cost({k: v.clone() for k, v in info.items()}, actions.clone())
    assert not torch.allclose(c1, c3), "changing seed should change stochastic cost"
