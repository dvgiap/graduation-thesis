# ride.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class RIDE(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 encoding_size=256,
                 num_layers=2,
                 episodic_memory_size=1000,
                 global_memory_size=50000,
                 k_neighbors=10,
                 sample_memory_size=1024):
        """
        Fixed & improved RIDE implementation:
         - correct Linear init ordering
         - normalize encodings before novelty
         - compute novelty with torch.no_grad() and sample memory if too large
         - encode always returns [B, enc_size]
         - supports discrete actions (one-hot). Continuous actions would need change.
        """
        super(RIDE, self).__init__()
        self.action_dim = action_dim
        self.encoding_size = encoding_size
        self.k_neighbors = k_neighbors
        self.sample_memory_size = sample_memory_size

        # Build encoder: initialize Linear then append activation (fixed order)
        encoder_layers = []
        l = nn.Linear(state_dim, encoding_size)
        nn.init.normal_(l.weight, mean=0.0, std=np.sqrt(1.0 / max(1, state_dim)))
        nn.init.zeros_(l.bias)
        encoder_layers.append(l)
        encoder_layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            l = nn.Linear(encoding_size, encoding_size)
            nn.init.normal_(l.weight, mean=0.0, std=np.sqrt(1.0 / max(1, encoding_size)))
            nn.init.zeros_(l.bias)
            encoder_layers.append(l)
            encoder_layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*encoder_layers)

        # forward & inverse models
        self.forward_model = nn.Sequential(
            nn.Linear(encoding_size + action_dim, 256),
            nn.Tanh(),
            nn.Linear(256, encoding_size)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(encoding_size * 2, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )

        # losses (created once)
        self.inverse_loss_fn = nn.CrossEntropyLoss()

        # memories stored on CPU (to avoid GPU OOM)
        self.episodic_memory = deque(maxlen=episodic_memory_size)
        self.global_memory = deque(maxlen=global_memory_size)

    def encode(self, state):
        """
        Accepts:
          - state: Tensor [B, state_dim] or [state_dim] or numpy/list
        Returns:
          - encoded: Tensor [B, encoding_size] on model device (float)
        """
        device = next(self.parameters()).device

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        state = state.to(device).float()
        # ensure batch dim
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, state_dim]

        encoded = self.encoder(state)  # [B, enc]
        # normalize for stable distance computations
        encoded = F.normalize(encoded, p=2, dim=-1)
        return encoded

    def _sample_memory_list(self, mem_deque):
        """Return list of memory tensors sampled (on CPU) up to sample_memory_size."""
        mem_list = list(mem_deque)
        M = len(mem_list)
        if M == 0:
            return []
        if M > self.sample_memory_size:
            idx = np.random.choice(M, size=self.sample_memory_size, replace=False)
            mem_list = [mem_list[i] for i in idx]
        return mem_list

    def compute_episodic_novelty(self, encoded_state):
        """
        encoded_state: [B, enc] on device
        returns: episodic_novelty [B] on same device
        """
        B = encoded_state.shape[0]
        device = encoded_state.device

        if len(self.episodic_memory) < 1:
            return torch.ones(B, device=device)

        with torch.no_grad():
            mem_list = self._sample_memory_list(self.episodic_memory)
            if len(mem_list) == 0:
                return torch.ones(B, device=device)
            memory_tensor = torch.stack(mem_list).to(device)  # [M, enc]
            k = min(self.k_neighbors, memory_tensor.shape[0])
            # distances
            distances = torch.cdist(encoded_state, memory_tensor, p=2)  # [B, M]
            k_distances, _ = torch.topk(distances, k=k, largest=False, dim=1)  # [B, k]
            episodic_novelty = k_distances.mean(dim=1)  # [B]
            # avoid zero
            episodic_novelty = episodic_novelty + 1e-8
        return episodic_novelty

    def compute_global_novelty(self, encoded_state):
        """
        Same as episodic but on global memory
        """
        B = encoded_state.shape[0]
        device = encoded_state.device

        if len(self.global_memory) < 1:
            return torch.ones(B, device=device)

        with torch.no_grad():
            mem_list = self._sample_memory_list(self.global_memory)
            if len(mem_list) == 0:
                return torch.ones(B, device=device)
            memory_tensor = torch.stack(mem_list).to(device)  # [M, enc]
            k = min(self.k_neighbors, memory_tensor.shape[0])
            distances = torch.cdist(encoded_state, memory_tensor, p=2)
            k_distances, _ = torch.topk(distances, k=k, largest=False, dim=1)
            global_novelty = k_distances.mean(dim=1)
            global_novelty = global_novelty + 1e-8
        return global_novelty

    def add_to_episodic_memory(self, encoded_state):
        """
        Accepts encoded_state: [enc] or [B, enc] (tensor on device)
        Stores CPU copies into deque
        """
        with torch.no_grad():
            if encoded_state.dim() == 1:
                self.episodic_memory.append(encoded_state.detach().cpu().clone())
            else:
                for i in range(encoded_state.shape[0]):
                    self.episodic_memory.append(encoded_state[i].detach().cpu().clone())

    def add_to_global_memory(self, encoded_state):
        """
        Add encoded states to global memory (typically at episode end).
        Accepts [B, enc] or [enc]
        """
        with torch.no_grad():
            if encoded_state.dim() == 1:
                self.global_memory.append(encoded_state.detach().cpu().clone())
            else:
                for i in range(encoded_state.shape[0]):
                    self.global_memory.append(encoded_state[i].detach().cpu().clone())

    def clear_episodic_memory(self):
        self.episodic_memory.clear()

    def forward(self, state, next_state, action):
        """
        Args:
            state: [B, state_dim] or [state_dim]
            next_state: [B, state_dim] or [state_dim]
            action: [B] (discrete indices) or list/np array
        Returns:
            intrinsic_reward: [B]
            inverse_loss: scalar tensor
            forward_loss_mean: scalar tensor
        """
        device = next(self.parameters()).device

        # encode (ensures shape [B, enc])
        encoded_state = self.encode(state)         # [B, enc]
        encoded_next_state = self.encode(next_state)  # [B, enc]

        # action to LongTensor [B]
        if not torch.is_tensor(action):
            action = torch.tensor(action)
        action = action.long().view(-1).to(device)

        # inverse model
        inverse_input = torch.cat([encoded_state, encoded_next_state], dim=-1)  # [B, 2*enc]
        predicted_action_logits = self.inverse_model(inverse_input)  # [B, action_dim]
        inverse_loss = self.inverse_loss_fn(predicted_action_logits, action)

        # forward model: one-hot for discrete
        action_one_hot = F.one_hot(action, num_classes=self.action_dim).float().to(device)
        forward_input = torch.cat([encoded_state, action_one_hot], dim=-1)  # [B, enc+action_dim]
        predicted_next_encoded_state = self.forward_model(forward_input)  # [B, enc]
        predicted_next_encoded_state = F.normalize(predicted_next_encoded_state, p=2, dim=-1)

        # per-sample forward MSE (controllable state error)
        mse_per_dim = F.mse_loss(predicted_next_encoded_state, encoded_next_state.detach(), reduction='none')  # [B, enc]
        forward_mse_per_sample = 0.5 * mse_per_dim.mean(dim=-1)  # [B]
        forward_loss_mean = forward_mse_per_sample.mean()

        # novelty computations (no grad)
        episodic_novelty = self.compute_episodic_novelty(encoded_next_state)  # [B]
        global_novelty = self.compute_global_novelty(encoded_next_state)      # [B]

        # add next state encoding to episodic memory (store CPU)
        self.add_to_episodic_memory(encoded_next_state.detach())

        # intrinsic reward per-step: forward_error * episodic_novelty * global_novelty
        intrinsic_reward = forward_mse_per_sample.detach() * episodic_novelty * global_novelty  # [B]
        # optional: scale/clamp later upstream

        return intrinsic_reward, inverse_loss, forward_loss_mean
