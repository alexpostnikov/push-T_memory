import torch
from src.ctm.ctm_transformer import CTMTransformerBlock

def test_ctm_transformer_block_output_shape():
    block = CTMTransformerBlock(embed_dim=16, num_heads=2, mlp_dim=32, ticks=2, sync_threshold=0.0)
    x = torch.randn(3, 5, 16)  # [batch, seq, embed_dim]
    out = block(x)
    assert out.shape == (3, 5, 16)

def test_ctm_transformer_tick_traces():
    block = CTMTransformerBlock(embed_dim=12, num_heads=2, mlp_dim=20, ticks=2, sync_threshold=0.0)
    x = torch.randn(1, 4, 12)
    block(x)
    traces = block.get_tick_traces()
    assert 'attn' in traces and 'ffn' in traces