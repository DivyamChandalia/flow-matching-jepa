import torch

from jepa import TrajectoryFlowJEPA
from module import FlowTrajectoryPredictor


def make_predictor(dim=32):
    return FlowTrajectoryPredictor(
        depth=2, heads=4, mlp_dim=64, input_dim=dim, hidden_dim=dim,
        action_dim=dim, output_dim=dim, output_proj_hidden_dim=64,
        causal_self_attention=False, dim_head=16,
    )


def make_model(dim=32, source="standard_noise"):
    # tiny ViT stand-in: encode() is bypassed; model methods take latents directly
    encoder = torch.nn.Identity()
    predictor = make_predictor(dim)
    action_encoder = torch.nn.Linear(4, dim)
    model = TrajectoryFlowJEPA.__new__(TrajectoryFlowJEPA)
    torch.nn.Module.__init__(model)
    model.encoder = encoder
    model.predictor = predictor
    model.action_encoder = action_encoder
    model.projector = torch.nn.Identity()
    model.pred_proj = torch.nn.Identity()
    model.num_flow_steps = 4
    model.flow_source = source
    model.flow_source_noise_scale = 1.0
    model.sampling_nfe = 4
    model._flow_seed = None
    model._flow_generators = {}
    return model


def test_predictor_velocity_shape_and_masks():
    torch.manual_seed(0)
    pred = make_predictor()
    B, H, P, D = 2, 1, 5, 32
    v = pred(torch.randn(B, P, D), torch.randn(B, H, D), torch.randn(B, P, D), torch.rand(B))
    assert v.shape == (B, P, D)
    assert v.isfinite().all()
    # mask builders
    m = pred._build_cross_mask(1, 5, 5, device="cpu")
    assert m.shape == (5, 6) and m[:, :1].all()
    assert m[0, 1:].sum() == 1 and m[-1, 1:].all()  # causal over actions
    cm = pred._build_causal_self_mask(3, device="cpu")
    assert (cm == cm.tril()).all() and cm.diag().all()


def test_sources():
    torch.manual_seed(0)
    m_std = make_model(source="standard_noise")
    m_nc = make_model(source="noisy_current")
    hist = torch.randn(2, 1, 32)
    noise = torch.randn(2, 5, 32)
    s_std = m_std._flow_source_from_noise(hist, noise)
    assert torch.allclose(s_std, noise)  # scale 1.0
    s_nc = m_nc._flow_source_from_noise(hist, noise)
    assert torch.allclose(s_nc, hist[:, -1:].expand(2, 5, 32) + noise)
    try:
        TrajectoryFlowJEPA(
            encoder=torch.nn.Identity(), predictor=make_predictor(),
            action_encoder=torch.nn.Linear(4, 32), flow_source="bogus",
        )
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_flow_loss_joint_grads():
    torch.manual_seed(0)
    m = make_model()
    hist = torch.randn(2, 1, 32, requires_grad=True)
    act = torch.randn(2, 5, 32, requires_grad=True)
    tgt = torch.randn(2, 5, 32, requires_grad=True)
    loss = m.flow_loss(hist, act, tgt)
    assert loss.isfinite()
    loss.backward()
    assert hist.grad is not None and tgt.grad is not None
    assert any(p.grad is not None for p in m.predictor.parameters())


def test_predict_rollout_cost_and_seed():
    torch.manual_seed(0)
    m = make_model()
    hist = torch.randn(2, 1, 32)
    act = torch.randn(2, 5, 32)
    out = m.predict(hist, act, horizon=5, num_steps=2, noise=torch.randn(2, 5, 32))
    assert out.shape == (2, 5, 32) and out.isfinite().all()

    m2 = make_model()
    # rollout/get_cost need encode(); use a mock CLS encoder (fixed buffer => deterministic)
    class MockEnc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_fixed", torch.randn(256, 1, 32))

        def forward(self, pixels, interpolate_pos_encoding=True):
            B = pixels.size(0)
            idx = torch.arange(B) % self._fixed.size(0)
            return type("O", (), {"last_hidden_state": self._fixed[idx]})()

    for mm in (m, m2):
        mm.encoder = MockEnc()
    B, S, T = 1, 2, 4
    info = {
        "pixels": torch.randn(B, 1, 3, 16, 16),
        "goal": torch.randn(B, 1, 1, 3, 16, 16),
        "action": torch.randn(B, 1, 4),
    }
    actions = torch.randn(B, S, T, 4)
    m2.set_flow_seed(7)
    r = m2.rollout({k: v.clone() for k, v in info.items()}, actions.clone())
    assert r["predicted_emb"].shape == (B, S, T, 32)
    m2.set_flow_seed(7)
    c1 = m2.get_cost({k: v.clone() for k, v in info.items()}, actions.clone())
    m2.set_flow_seed(7)
    c2 = m2.get_cost({k: v.clone() for k, v in info.items()}, actions.clone())
    assert c1.shape == (B, S) and torch.allclose(c1, c2)
