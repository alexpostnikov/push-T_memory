import torch
import torch.nn as nn
import torch.nn.functional as F


class CTMSynapseWrapper(nn.Module):
    """
    CTM Synapse wrapper for transformer layers.
    Adds internal tick mechanism, per-neuron state history, and synchronization gating.
    """

    def __init__(self, module, num_neurons, ticks=1, sync_threshold=0.8, device=None):
        """
        Args:
            module: nn.Module to wrap (e.g., nn.Linear/Attention).
            num_neurons: Number of output neurons (for state buffer).
            ticks: Number of internal ticks per step.
            sync_threshold: Threshold for sync gating.
            device: torch device.
        """
        super().__init__()
        self.module = module
        self.num_neurons = num_neurons
        self.ticks = ticks
        self.sync_threshold = sync_threshold
        self.device = device or torch.device("cpu")

        # Internal state buffer: shape [batch, num_neurons, ticks]
        self.register_buffer("state_history", torch.zeros(1, num_neurons, ticks))
        # Sync gating mask (updated every forward)
        self.sync_mask = None

    def tick_reset(self, batch_size):
        """Reset state history for a new sequence/batch."""
        self.state_history = torch.zeros(batch_size, self.num_neurons, self.ticks, device=self.device)
        self.sync_mask = torch.zeros(batch_size, self.num_neurons, device=self.device)

    def forward(self, x):
        """
        Args:
            x: Input tensor [..., num_neurons]
        Returns:
            Output tensor, possibly synchronized across ticks.
        """
        batch_size = x.shape[0]
        # Reset state if shape mismatch
        if self.state_history.shape[0] != batch_size:
            self.tick_reset(batch_size)

        tick_outputs = []
        for t in range(self.ticks):
            # Pass through wrapped module
            out = self.module(x)
            out = F.relu(out)
            # Update state history
            self.state_history[:, :, t] = out.detach()
            # Compute sync (e.g., stddev or range across tick history)
            if t > 0:
                sync_metric = torch.std(self.state_history[:, :, :t+1], dim=-1)
            else:
                sync_metric = torch.zeros_like(out)
            # Gating: fire if sync_metric > threshold
            fired = (sync_metric > self.sync_threshold).float()
            self.sync_mask = fired
            # Only keep output where neurons "fire"
            gated_out = out * fired
            tick_outputs.append(gated_out)

            # Stop early if all neurons have fired (optional)
            if torch.all(fired > 0):
                break

        # Aggregate outputs (e.g., sum, mean, or last nonzero)
        agg_out = torch.stack(tick_outputs, dim=1)  # [B, ticks, N]
        # Take last nonzero per neuron (fallback to last tick if all zero)
        mask = (agg_out != 0).float()
        idx = mask.cumsum(1).argmax(1)
        final_out = agg_out[torch.arange(batch_size).unsqueeze(-1), idx, torch.arange(self.num_neurons)]
        return final_out

    def get_tick_trace(self):
        """Return neuron activation traces for visualization."""
        return self.state_history.detach().cpu().numpy()

    def get_sync_mask(self):
        return self.sync_mask.detach().cpu().numpy()