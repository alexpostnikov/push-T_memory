import torch
from src.models.ctm_act_policy import CTMACTPolicy

def test_ctm_act_policy_shapes():
    model = CTMACTPolicy(obs_dim=10, act_dim=4, embed_dim=16, num_heads=2, mlp_dim=32, n_blocks=2, ticks=2)
    x = torch.randn(5, 7, 10)  # [batch, seq, obs_dim]
    out = model(x)
    assert out.shape == (5, 7, 4)

def test_ctm_act_policy_tick_traces():
    model = CTMACTPolicy(obs_dim=8, act_dim=3, embed_dim=12, num_heads=2, mlp_dim=24, n_blocks=1, ticks=2)
    x = torch.randn(2, 3, 8)
    model(x)
    traces = model.get_all_tick_traces()
    assert isinstance(traces, list) and len(traces) == 1
    assert 'attn' in traces[0] and 'ffn' in traces[0]