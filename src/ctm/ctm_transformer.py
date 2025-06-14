import torch
import torch.nn as nn
from src.ctm.ctm_synapse import CTMSynapseWrapper

class CTMTransformerBlock(nn.Module):
    """
    Example Transformer block using CTM synapse wrappers.
    Replace the standard Attention and FFN with tick-synchronous CTM modules.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, ticks=1, sync_threshold=0.8, device=None):
        super().__init__()
        from torch.nn import MultiheadAttention

        self.attn = CTMSynapseWrapper(
            MultiheadAttention(embed_dim, num_heads, batch_first=True),
            num_neurons=embed_dim,
            ticks=ticks,
            sync_threshold=sync_threshold,
            device=device,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = CTMSynapseWrapper(
            nn.Linear(embed_dim, mlp_dim),
            num_neurons=mlp_dim,
            ticks=ticks,
            sync_threshold=sync_threshold,
            device=device,
        )
        self.norm2 = nn.LayerNorm(mlp_dim)
        self.out_proj = nn.Linear(mlp_dim, embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        # x: [B, T, D]
        # Attention
        attn_out = self.attn(x)
        x = x + attn_out
        x = self.norm1(x)
        # FFN
        ffn_out = self.ffn(x)
        ffn_out = self.norm2(ffn_out)
        out = self.out_proj(ffn_out)
        x = x + out
        x = self.norm3(x)
        return x

    def get_tick_traces(self):
        return {
            "attn": self.attn.get_tick_trace(),
            "ffn": self.ffn.get_tick_trace(),
        }