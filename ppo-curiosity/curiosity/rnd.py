import torch
import torch.nn as nn
import numpy as np


class RunningMeanStd:
    """Welford-style running mean/var (per-dim). Used by RND §2.4 normalization."""
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0 or x.shape[0] == 0:
            return
        bm, bv, bc = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + np.square(delta) * self.count * bc / tot) / tot
        self.count = tot


class RewardForwardFilter:
    """Discounted forward filter on intrinsic rewards (single-env scalar)."""
    def __init__(self, gamma):
        self.gamma = float(gamma)
        self.rewems = None

    def update(self, r):
        r = float(r)
        self.rewems = r if self.rewems is None else self.rewems * self.gamma + r
        return self.rewems


class RND(nn.Module):
    """
    Random Network Distillation (Burda et al. 2018).
    - target: frozen random-init network
    - predictor: trainable network
    Intrinsic reward = per-sample MSE between predictor(s) and target(s) on next_state.

    Paper §2.4 normalization (added):
      * observation: per-dim (s - mean) / std, clipped to [-5, 5], same stats for
        both target and predictor.
      * intrinsic reward: divided by running std of *intrinsic returns* (discounted
        accumulator via RewardForwardFilter); no mean subtraction.
    """
    def __init__(self, state_dim, encoding_size=256, num_layers=2,
                 gamma_int=0.99, obs_clip=5.0):
        super(RND, self).__init__()
        self.state_dim = int(state_dim)
        self.obs_clip = float(obs_clip)
        self.target = self._build_net(state_dim, encoding_size, num_layers)
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = self._build_net(state_dim, encoding_size, num_layers)

        # §2.4 running stats
        self.obs_rms = RunningMeanStd(shape=(self.state_dim,))
        self.rff_int = RewardForwardFilter(gamma_int)
        self.rff_rms = RunningMeanStd(shape=())

    @staticmethod
    def _build_net(state_dim, encoding_size, num_layers):
        layers = []
        layers.append(nn.Linear(state_dim, encoding_size))
        nn.init.normal_(layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / state_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(encoding_size, encoding_size))
            nn.init.normal_(layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / encoding_size))
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    # ---------------- §2.4 normalization helpers ----------------
    def update_obs_rms(self, states):
        """Push a batch of observations into the running mean/std stats."""
        if isinstance(states, torch.Tensor):
            states = states.detach().cpu().numpy()
        states = np.asarray(states, dtype=np.float64).reshape(-1, self.state_dim)
        self.obs_rms.update(states)

    def _normalize_obs(self, x):
        mean = torch.as_tensor(self.obs_rms.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(np.sqrt(self.obs_rms.var), dtype=x.dtype, device=x.device)
        return torch.clamp((x - mean) / (std + 1e-8), -self.obs_clip, self.obs_clip)

    def normalize_intrinsic(self, intr_np):
        """
        Push raw per-step intrinsic rewards through the discounted forward filter,
        update the running variance of intrinsic returns, and return the rewards
        divided by sqrt(var).  No mean subtraction (paper §2.4).
        """
        intr_np = np.asarray(intr_np, dtype=np.float64).reshape(-1)
        rffs = np.empty_like(intr_np)
        for i, r in enumerate(intr_np):
            rffs[i] = self.rff_int.update(r)
        self.rff_rms.update(rffs)
        std = float(np.sqrt(max(self.rff_rms.var, 1e-12)))
        return (intr_np / (std + 1e-8)).astype(np.float32)

    def forward(self, next_state):
        """
        Args:
            next_state: [batch_size, state_dim] — state to compute novelty for
        Returns:
            intrinsic_reward: [batch_size] (detached, raw — pass through
                              `normalize_intrinsic` to get §2.4-scaled values)
            predictor_loss: scalar (with grad, for training predictor)
        """
        x = self._normalize_obs(next_state)
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        per_sample_error = 0.5 * ((pred_feat - target_feat) ** 2).mean(dim=-1)
        intrinsic_reward = per_sample_error.detach()
        predictor_loss = per_sample_error.mean()
        return intrinsic_reward, predictor_loss
