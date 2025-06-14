import torch
from src.ctm.ctm_synapse import CTMSynapseWrapper

def test_ctm_synapse_forward_shapes():
    dummy_linear = torch.nn.Linear(16, 32)
    wrapper = CTMSynapseWrapper(dummy_linear, num_neurons=32, ticks=3, sync_threshold=0.0)
    x = torch.randn(4, 16)
    out = wrapper(x)
    assert out.shape == (4, 32)
    # Check state history shape
    assert wrapper.state_history.shape == (4, 32, 3)

def test_ctm_synapse_tick_trace_and_sync_mask():
    dummy_linear = torch.nn.Linear(8, 8)
    wrapper = CTMSynapseWrapper(dummy_linear, num_neurons=8, ticks=2, sync_threshold=0.0)
    x = torch.ones(2, 8)
    out = wrapper(x)
    trace = wrapper.get_tick_trace()
    mask = wrapper.get_sync_mask()
    assert trace.shape == (2, 8, 2)
    assert mask.shape == (2, 8)