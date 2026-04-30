import torch
import torch.nn as nn
import numpy as np


class RIDE(nn.Module):
    """
    RIDE: Rewarding Impact-Driven Exploration (Raileanu & Rocktäschel, ICLR 2020).

    Intrinsic reward:
        r_intr(s_t, s_{t+1}) = ||phi(s_{t+1}) - phi(s_t)||_2 / sqrt(N_episode(s_{t+1}))

    where phi is an encoder learned via inverse + forward dynamics (same training
    objective as ICM, but trained INDEPENDENTLY of any other ICM module), and
    N_episode is a per-episode visitation count via a random-projection hash.

    Episode boundaries are honored at compute time via the `is_terminals` buffer:
    each call to `compute_intrinsic_reward(...)` resets a LOCAL episodic count
    dict whenever it crosses a transition where is_terminals[t] is True.
    """

    def __init__(self, state_dim, action_dim,
                 encoding_size=256, num_layers=2,
                 hash_dim=32):
        super(RIDE, self).__init__()
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.hash_dim = int(hash_dim)

        # Encoder phi
        encoder_layers = []
        encoder_layers.append(nn.Linear(state_dim, encoding_size))
        nn.init.normal_(encoder_layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / state_dim))
        encoder_layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            encoder_layers.append(nn.Linear(encoding_size, encoding_size))
            nn.init.normal_(encoder_layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / encoding_size))
            encoder_layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_layers)

        # Inverse model: predict action from (phi(s), phi(s'))
        self.inverse_model = nn.Sequential(
            nn.Linear(encoding_size * 2, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )

        # Forward model: predict phi(s') from (phi(s), a)
        self.forward_model = nn.Sequential(
            nn.Linear(encoding_size + action_dim, 256),
            nn.Tanh(),
            nn.Linear(256, encoding_size)
        )

        # Random projection for episodic state hashing.
        # Stored as numpy array (used on CPU at hash time).
        self.projection_matrix = np.random.randn(state_dim, self.hash_dim).astype(np.float32)

    # ---------------- encode ----------------
    def encode(self, state):
        return self.encoder(state)

    # ---------------- hash ----------------
    def _hash_one(self, state_np):
        """Hash a single 1D numpy state to a string key."""
        projected = state_np @ self.projection_matrix
        binned = np.round(projected / 0.1).astype(np.int32)
        return binned.tobytes()

    # ---------------- forward (training) ----------------
    def forward(self, state, next_state, action):
        """
        Train signal for the encoder via inverse + forward dynamics.
        Mirrors ICM.forward() so that training pipelines are interchangeable.

        Returns:
            intrinsic_reward: forward-model prediction error per sample [batch]
                              (NOT the RIDE intrinsic reward; that requires
                              episodic counts -- see compute_intrinsic_reward).
            inverse_loss: scalar
            forward_loss: scalar (averaged)
        """
        encoded_state = self.encode(state)
        encoded_next_state = self.encode(next_state)

        # Inverse loss: predict a from (phi(s), phi(s'))
        inverse_input = torch.cat([encoded_state, encoded_next_state], dim=-1)
        predicted_action_logits = self.inverse_model(inverse_input)
        inverse_loss = nn.CrossEntropyLoss()(predicted_action_logits, action)

        # Forward loss: predict phi(s') from (phi(s).detach(), a)
        action_one_hot = nn.functional.one_hot(action, num_classes=self.action_dim).float()
        forward_input = torch.cat([encoded_state.detach(), action_one_hot], dim=-1)
        predicted_next_encoded_state = self.forward_model(forward_input)
        forward_per_sample = 0.5 * ((predicted_next_encoded_state - encoded_next_state.detach()).pow(2)).mean(dim=-1)

        intrinsic_reward = forward_per_sample.detach()
        forward_loss = forward_per_sample.mean()

        return intrinsic_reward, inverse_loss, forward_loss

    # ---------------- RIDE intrinsic ----------------
    def compute_intrinsic_reward(self, states, next_states, is_terminals):
        """
        Batch RIDE intrinsic over a CHRONO rollout buffer.

        Args:
            states, next_states: Tensor [T, state_dim]
            is_terminals: 1-D Tensor or array-like of length T; entry t is truthy
                          if the transition at index t terminated an episode.
                          Episodic count is reset AFTER a terminal transition,
                          so the next state's count restarts at 1.

        Returns:
            intrinsic: Tensor [T] on the same device as `states`,
                       value = ||phi(s') - phi(s)||_2 / sqrt(N_episode(s')).
        """
        device = states.device

        with torch.no_grad():
            phi = self.encoder(states)
            phi_next = self.encoder(next_states)
        impact = (phi_next - phi).norm(dim=-1)  # [T]

        T = states.shape[0]
        next_states_np = next_states.detach().cpu().numpy()
        if isinstance(is_terminals, torch.Tensor):
            is_terminals_np = is_terminals.detach().cpu().numpy()
        else:
            is_terminals_np = np.asarray(is_terminals)

        counts = np.empty(T, dtype=np.float32)
        ep_count = {}  # LOCAL: born and dies in this method scope
        for t in range(T):
            h = self._hash_one(next_states_np[t].reshape(-1))
            ep_count[h] = ep_count.get(h, 0) + 1
            counts[t] = ep_count[h]
            if bool(is_terminals_np[t]):
                ep_count.clear()

        counts_t = torch.from_numpy(counts).to(device)
        intrinsic = impact / torch.sqrt(counts_t)
        return intrinsic
