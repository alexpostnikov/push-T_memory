import torch
import torch.nn as nn
from src.ctm.ctm_transformer import CTMTransformerBlock

class CTMACTPolicy(nn.Module):
    """
    CTM-ACT Policy: ACT transformer policy with CTM synapse integration.
    - Wraps each transformer block in CTM-style tick/sync modules.
    - Use as a drop-in replacement for ACT or baseline policies.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        embed_dim=128,
        num_heads=4,
        mlp_dim=256,
        n_blocks=4,
        ticks=1,
        sync_threshold=0.8,
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.embed = nn.Linear(obs_dim, embed_dim)
        self.blocks = nn.ModuleList([
            CTMTransformerBlock(
                embed_dim,
                num_heads,
                mlp_dim,
                ticks=ticks,
                sync_threshold=sync_threshold,
                device=self.device,
            )
            for _ in range(n_blocks)
        ])
        self.head = nn.Linear(embed_dim, act_dim)

    def forward(self, obs):
        """
        Args:
            obs: [batch, seq_len, obs_dim]
        Returns:
            act: [batch, seq_len, act_dim]
        """
        x = self.embed(obs)
        for block in self.blocks:
            x = block(x)
        act = self.head(x)
        return act

    def set_ticks(self, ticks):
        """Set number of ticks for all CTM transformer blocks."""
        for block in self.blocks:
            block.attn.ticks = ticks
            block.ffn.ticks = ticks

    def get_all_tick_traces(self):
        """Get tick traces for all blocks (for visualization)."""
        return [blk.get_tick_traces() for blk in self.blocks]

    def reset_all_ticks(self, batch_size):
        """Reset tick state for new sequences/batches."""
        for block in self.blocks:
            block.attn.tick_reset(batch_size)
            block.ffn.tick_reset(batch_size)