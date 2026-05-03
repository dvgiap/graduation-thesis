import math
import os
import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class BetaNetwork(nn.Module):
    """State-dependent β(s) ∈ [β_min, β_max] via log-space clamped output."""

    def __init__(self, state_dim, encoding_size=256, num_layers=2, head_hidden=128,
                 min_beta=1e-4, max_beta=1e-1,
                 beta_0=None):
        super().__init__()
        self.min = float(min_beta)
        self.max = float(max_beta)
        # Default β₀ = geometric mean of clamp range → init bias sits in the middle
        # of log-space, no asymmetric pull toward either bound.
        self.beta_0 = float(beta_0) if beta_0 is not None else math.sqrt(self.min * self.max)

        layers = []
        layers.append(nn.Linear(state_dim, encoding_size))
        nn.init.normal_(layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / max(1, state_dim)))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(encoding_size, encoding_size))
            nn.init.normal_(layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / max(1, encoding_size)))
            layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*layers)

        self.head_out = nn.Linear(head_hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(encoding_size, head_hidden),
            nn.Tanh(),
            self.head_out,
        )

        with torch.no_grad():
            if self.head_out.bias is not None:
                nn.init.constant_(self.head_out.bias, math.log(self.beta_0))
            nn.init.normal_(self.head_out.weight, mean=0.0, std=1e-3)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.encoder(state)
        log_beta = self.head(h).squeeze(-1)
        log_beta = torch.clamp(log_beta, math.log(self.min), math.log(self.max))
        return torch.exp(log_beta)


class CAREModule:
    """
    State-dependent β scaling module: BetaNetwork + correlation loss + Polyak target.

    Public API:
        update(states, intrinsic_plus, extrinsic_advantages) -> float  (meta_loss)
        combine(R_ext, I_plus, states) -> Tensor               (shaped rewards)
        get_beta_scalar(states) -> float                       (for logging)
        sample_and_log(states)                                 (viz helper)
        save(path) / load(path)
    """

    def __init__(self, state_dim,
                 beta_min=0.0001, beta_max=0.1, beta_0=None,
                 encoding_size=256, num_layers=2, head_hidden=128,
                 lr=5e-4, weight_decay=1e-6, grad_clip=1.0,
                 reg_weight=1e-3,
                 use_state_dependent=True, use_progress=True,
                 target_tau=0.01,
                 logger=None):

        self.use_state_dependent = use_state_dependent
        self.use_progress = use_progress
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        # β₀ defaults to geometric mean of clamp → log-space center, symmetric init.
        self.beta_0 = float(beta_0) if beta_0 is not None else math.sqrt(self.beta_min * self.beta_max)
        self.grad_clip = float(grad_clip)
        self.reg_weight = float(reg_weight)
        self.tau = float(target_tau)
        self.logger = logger

        if use_state_dependent:
            self.beta_net = BetaNetwork(
                state_dim, encoding_size, num_layers, head_hidden, beta_min, beta_max, beta_0
            ).to(device)

            # Polyak EMA target network: combine() uses this lagged copy so the β
            # shaping rewards is always one soft-update behind the β being trained.
            self.beta_net_target = BetaNetwork(
                state_dim, encoding_size, num_layers, head_hidden, beta_min, beta_max, beta_0
            ).to(device)
            self.beta_net_target.load_state_dict(self.beta_net.state_dict())
            for p in self.beta_net_target.parameters():
                p.requires_grad_(False)

            self.optimizer = torch.optim.Adam(
                self.beta_net.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            # Fixed-β path: scalar log-parameterized β, not updated when use_progress=False.
            self.beta_log = nn.Parameter(
                torch.tensor([math.log(self.beta_0)], dtype=torch.float32, device=device)
            )
            self.optimizer = torch.optim.Adam([self.beta_log], lr=lr)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _polyak_update(self):
        """Soft-update: β_target ← (1-τ)·β_target + τ·β_live."""
        with torch.no_grad():
            for p_live, p_tgt in zip(self.beta_net.parameters(),
                                     self.beta_net_target.parameters()):
                p_tgt.data.mul_(1.0 - self.tau).add_(self.tau * p_live.data)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(self, states, intrinsic_plus, extrinsic_advantages):
        """
        Correlation objective with extrinsic advantage as supervisory signal.

            L_β = -E[ Î_z · Â_z ]  +  λ_reg · L_reg

        where
            Î = β(s) · I⁺                         weighted intrinsic
            Â = Â^E_t  (extrinsic GAE advantage)  variance-reduced future-return proxy
            *_z = z-score within minibatch
            L_reg = (log β(s) - log β₀)²          log-space anchor toward prior

        Args:
            states:                tensor (B, state_dim)
            intrinsic_plus:        tensor (B,) — I⁺, already z-scored & rectified by caller
            extrinsic_advantages:  tensor (B,) — Â^E from caller's GAE
        Returns:
            float meta_loss (0.0 if use_progress=False)
        """
        if not self.use_progress:
            return 0.0

        if self.use_state_dependent:
            beta = self.beta_net(states).squeeze()
        else:
            beta = torch.exp(self.beta_log).expand(states.shape[0])

        intrinsic_plus = intrinsic_plus.detach().to(beta.device)
        adv = extrinsic_advantages.detach().to(beta.device)

        weighted_intr = beta * intrinsic_plus  # Î

        eps = 1e-8
        Iz = (weighted_intr - weighted_intr.mean()) / (weighted_intr.std(unbiased=False) + eps)
        Az = (adv - adv.mean()) / (adv.std(unbiased=False) + eps)
        loss_corr = -(Iz * Az).mean()

        if self.use_state_dependent:
            loss_reg = ((torch.log(beta + eps) - math.log(self.beta_0)) ** 2).mean()
        else:
            loss_reg = (self.beta_log - math.log(self.beta_0)) ** 2

        loss_meta = loss_corr + self.reg_weight * loss_reg

        self.optimizer.zero_grad()
        loss_meta.backward()

        total_norm = 0.0
        if self.use_state_dependent:
            for p in self.beta_net.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
        else:
            if self.beta_log.grad is not None:
                total_norm = float(self.beta_log.grad.data.norm(2).item() ** 2)
        total_norm = math.sqrt(total_norm) if total_norm > 0 else 0.0

        if self.use_state_dependent:
            torch.nn.utils.clip_grad_norm_(self.beta_net.parameters(), self.grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_([self.beta_log], self.grad_clip)
        self.optimizer.step()

        if self.use_state_dependent:
            self._polyak_update()

        meta_loss_val = float(loss_meta.detach().cpu().item())

        if self.logger is not None:
            self.logger.log_scalar('beta_gradient_norm', total_norm)
            self.logger.log_scalar('loss_corr', float(loss_corr.detach().cpu().item()))
            self.logger.log_scalar('loss_reg', float(loss_reg.detach().cpu().item()))
            self.logger.log_scalar('adv_ext_mean', float(adv.mean().item()))
            self.logger.log_scalar('adv_ext_std', float(adv.std(unbiased=False).item()))
            self.logger.log_scalar('intr_plus_mean', float(intrinsic_plus.mean().item()))
            self.logger.log_array('beta_per_state', beta.detach())

        return meta_loss_val

    def combine(self, R_ext, I_plus, states):
        """
        r̄ = R_ext + β_target(s) · I^+

        Uses the lagged TARGET network (not beta_net) so the β that shapes
        rewards is decoupled from the β currently being optimized.
        """
        with torch.no_grad():
            if self.use_state_dependent:
                beta = self.beta_net_target(states).squeeze().to(device)
            else:
                beta = float(torch.exp(self.beta_log).item())
        return R_ext + beta * I_plus

    def get_beta_scalar(self, states):
        """Return mean β(s) over states using the LIVE network (for logging)."""
        with torch.no_grad():
            try:
                if self.use_state_dependent:
                    return float(self.beta_net(states).mean().detach().cpu().item())
                else:
                    return float(torch.exp(self.beta_log).detach().cpu().item())
            except Exception:
                return 1.0

    def sample_and_log(self, states):
        """Log β values for sampled states via logger (live network)."""
        if self.logger is None:
            return
        with torch.no_grad():
            if self.use_state_dependent:
                beta_vals = self.beta_net(states).squeeze().detach()
            else:
                beta_vals = torch.full(
                    (states.shape[0],),
                    float(torch.exp(self.beta_log).item()),
                    device=device
                )
        self.logger.sample_and_log(states, beta_vals, name='beta')

    def save(self, path):
        ckpt = {'optimizer': self.optimizer.state_dict()}
        if self.use_state_dependent:
            ckpt['beta_net'] = self.beta_net.state_dict()
            ckpt['beta_net_target'] = self.beta_net_target.state_dict()
        else:
            ckpt['beta_log'] = self.beta_log.detach().cpu()
        torch.save(ckpt, path)

    def load(self, path):
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        if self.use_state_dependent:
            if 'beta_net' in ckpt:
                self.beta_net.load_state_dict(ckpt['beta_net'])
            if 'beta_net_target' in ckpt:
                self.beta_net_target.load_state_dict(ckpt['beta_net_target'])
            elif 'beta_net' in ckpt:
                # backward compat: old checkpoint has no target — init target = live
                self.beta_net_target.load_state_dict(ckpt['beta_net'])
        else:
            if 'beta_log' in ckpt:
                with torch.no_grad():
                    self.beta_log.data.copy_(ckpt['beta_log'].to(device))
        if 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except Exception:
                pass
