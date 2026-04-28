import torch
import torch.nn as nn
import numpy as np
from collections import deque


class RE3(nn.Module):
    """
    Random Encoders for Efficient Exploration (Seo et al. 2021).

    Intrinsic reward at state s_t:
        r^i(s_t) = log( ||y_t - y_t^{k-NN}||_2 + 1 )

    where y_t = f_theta(s_t) is a fixed random encoder embedding,
    and y_t^{k-NN} is the k-th nearest neighbor of y_t in a stored
    set of past embeddings (a small replay buffer maintained even
    for on-policy algorithms; the paper uses 10K for MiniGrid).

    The encoder is randomly initialized and FROZEN throughout training
    -- there are no learnable parameters in this module.
    """

    def __init__(self, state_dim, encoding_size=64, num_layers=2,
                 k=3, buffer_size=10000):
        super(RE3, self).__init__()
        self.state_dim = int(state_dim)
        self.k = int(k)
        self.buffer_size = int(buffer_size)

        # Build random encoder
        layers = []
        layers.append(nn.Linear(state_dim, encoding_size))
        nn.init.normal_(layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / state_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(encoding_size, encoding_size))
            nn.init.normal_(layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / encoding_size))
            layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*layers)

        # Freeze encoder: no gradients, no updates
        for p in self.encoder.parameters():
            p.requires_grad = False

        # FIFO buffer of past embeddings on CPU as a list of 1D tensors
        self.embedding_buffer = deque(maxlen=self.buffer_size)

    @torch.no_grad()
    def encode(self, state):
        """Encode a batch of states. Returns embeddings on the encoder's device."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.encoder(state)

    def add_to_buffer(self, embeddings):
        """Append a batch of embeddings (Tensor [B, d]) to the FIFO buffer (CPU)."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu()
        for i in range(embeddings.shape[0]):
            self.embedding_buffer.append(embeddings[i])

    def _stacked_buffer(self):
        """Return the buffer as a single Tensor [N, d] on CPU; None if empty."""
        if len(self.embedding_buffer) == 0:
            return None
        return torch.stack(list(self.embedding_buffer), dim=0)

    @torch.no_grad()
    def compute_intrinsic_reward(self, states, update_buffer=True):
        """
        Compute RE3 intrinsic reward for a batch of states.

        Args:
            states:        Tensor [B, state_dim]
            update_buffer: if True, also append the produced embeddings to the FIFO buffer.

        Returns:
            intrinsic:     Tensor [B] on the same device as `states`,
                           with values log(||y_t - y_t^{k-NN}||_2 + 1).
                           If the buffer has fewer than k+1 elements, returns zeros.
        """
        device = states.device
        y = self.encode(states).cpu()                         # [B, d]
        buf = self._stacked_buffer()                          # [N, d] or None
        B = y.shape[0]

        if update_buffer:
            self.add_to_buffer(y)

        if buf is None or buf.shape[0] < self.k + 1:
            # Not enough samples yet; return zeros
            return torch.zeros(B, device=device)

        # Pairwise L2 distances: [B, N]
        # Use cdist for numerical stability
        dists = torch.cdist(y, buf, p=2)                      # [B, N]

        # Exclude the trivial self-distance for entries that are themselves in the
        # buffer: pick the (k+1)-th smallest if 0 is present, else k-th smallest.
        # In practice, since y was just appended to the buffer the trivial 0 will
        # be among the distances; selecting kth_value(k+1) handles this cleanly.
        eff_k = min(self.k + 1, dists.shape[1])
        knn_dist, _ = torch.kthvalue(dists, eff_k, dim=1)      # [B]

        intrinsic_cpu = torch.log(knn_dist + 1.0)
        return intrinsic_cpu.to(device)

    def reset_buffer(self):
        """Clear the embedding buffer (e.g., between independent training runs)."""
        self.embedding_buffer.clear()
