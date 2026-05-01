import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
from training_logger import TrainingLogger
from curiosity.icm import ICM

# ---------------- device ----------------
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    try:
        print("Device set to :", torch.cuda.get_device_name(0))
    except Exception:
        print("Device set to : cuda")
else:
    print("Device set to : cpu")


# ---------------- BetaNetwork  ----------------
class BetaNetwork(nn.Module):
    """
    Beta network that mirrors the ICM encoder design
    """
    def __init__(self, state_dim, encoding_size=256, num_layers=2, head_hidden=128,
                 min_beta=1e-3, max_beta=10.0):
        super(BetaNetwork, self).__init__()
        self.min = float(min_beta)
        self.max = float(max_beta)
        layers = []
        
        layers.append(nn.Linear(state_dim, encoding_size))
        nn.init.normal_(layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / max(1, state_dim)))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(encoding_size, encoding_size))
            nn.init.normal_(layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / max(1, encoding_size)))
            layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(encoding_size, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1)
        )
        
        with torch.no_grad():
            try:
                nn.init.constant_(self.head[-1].bias, math.log(0.01))
                nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)
            except Exception:
                pass

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.encoder(state)
        log_beta = self.head(h).squeeze(-1)
        log_beta = torch.clamp(log_beta, math.log(self.min), math.log(self.max))
        beta = torch.exp(log_beta)
        return beta


# ---------------- Rollout buffer ----------------
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.next_states = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.next_states[:]


# ---------------- ActorCritic ----------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init=0.6):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = int(action_dim)
            self.action_var = torch.full((self.action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, int(action_dim)),
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        single = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single = True

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(state.device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            logits = self.actor(state)
            dist = Categorical(logits=logits)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        if single:
            return action.squeeze(0), action_logprob.squeeze(0), state_val.squeeze(0)
        return action, action_logprob, state_val

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(state.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            if action.dim() == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            logits = self.actor(state)
            dist = Categorical(logits=logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


# ---------------- PPO ----------------
class PPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 gamma=0.99,
                 K_epochs=4,
                 eps_clip=0.2,
                 has_continuous_action_space=False,
                 action_std_init=0.6,
                 # ICM
                 use_icm=True,
                 icm_lr=1e-3,
                 icm_epochs=4,
                 icm_batch_size=64,
                 # GAE
                 gae_lambda=0.95,
                 # meta-beta
                 beta_lr=5e-4,
                 use_state_dependent_beta=True,
                 beta_init=1.0,
                 beta_encoding_size=256,
                 beta_num_layers=2,
                 beta_head_hidden=128,
                 beta_min=0.001,
                 beta_max=0.1,
                 # meta options (Learning-Progress)
                 meta_use_progress=True,
                 meta_progress_weight=1.0,
                 meta_reg_weight=1e-3,
                 meta_reg_beta_center=0.01,
                 # logging
                 sample_states_per_update=256,
                 sample_every_n_updates=1,
                 debug=False):
        
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init

        # basics
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        self.buffer = RolloutBuffer()

        # networks
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.MseLoss = nn.MSELoss()

        # ICM
        self.use_icm = use_icm and (ICM is not None)
        self.icm_epochs = icm_epochs
        self.icm_batch_size = icm_batch_size
        if self.use_icm:
            self.icm = ICM(state_dim, action_dim, encoding_size=beta_encoding_size, num_layers=beta_num_layers).to(device)
            self.optimizer_icm = torch.optim.Adam(self.icm.parameters(), lr=icm_lr)

        # meta-beta
        self.use_state_dependent_beta = use_state_dependent_beta
        if not use_state_dependent_beta:
            self.beta_log = nn.Parameter(torch.tensor([math.log(beta_init)], dtype=torch.float32, device=device))
            self.beta_optimizer = torch.optim.Adam([self.beta_log], lr=beta_lr)
        else:
            self.beta_net = BetaNetwork(state_dim,
                                        encoding_size=beta_encoding_size,
                                        num_layers=beta_num_layers,
                                        head_hidden=beta_head_hidden,
                                        min_beta=beta_min,
                                        max_beta=beta_max).to(device)
            self.beta_optimizer = torch.optim.Adam(self.beta_net.parameters(), lr=beta_lr, weight_decay=1e-6)

        # meta options (Learning-Progress)
        self.meta_use_progress = meta_use_progress
        self.meta_progress_weight = meta_progress_weight
        self.meta_reg_weight = meta_reg_weight
        self.meta_reg_beta_center = float(meta_reg_beta_center)
        self.beta_init = float(beta_init)

        # LOGGER 
        self.logger = TrainingLogger(
            sample_states_per_update=sample_states_per_update,
            sample_every_n_updates=sample_every_n_updates,
            auto_convert_to_numpy=True
        )

        # debug
        self.debug = debug

    # ---------------- action std helpers ----------------
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = max(self.action_std - action_std_decay_rate, min_action_std)
            self.set_action_std(self.action_std)

    # ---------------- select action ----------------
    def select_action(self, state):
        state_t = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state_t)
        
        self.buffer.states.append(state_t)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        
        if self.has_continuous_action_space:
            return action.cpu().numpy().flatten()
        else:
            return int(action.item())

    # ---------------- ICM update ----------------
    def icm_update(self, states, next_states, actions):
        total_forward_loss = 0.0
        total_inverse_loss = 0.0
        num_updates = 0

        dataset_size = states.shape[0]
        if dataset_size == 0 or (not self.use_icm):
            return 0.0, 0.0

        for _ in range(self.icm_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.icm_batch_size):
                num_updates += 1
                batch_idx = indices[start:start + self.icm_batch_size]
                batch_states = states[batch_idx]
                batch_next_states = next_states[batch_idx]
                batch_actions = actions[batch_idx]
                _, inverse_loss, forward_loss = self.icm(batch_states, batch_next_states, batch_actions)
                icm_loss = 0.2 * forward_loss + 0.8 * inverse_loss
                self.optimizer_icm.zero_grad()
                icm_loss.backward()
                self.optimizer_icm.step()
                total_forward_loss += forward_loss.item()
                total_inverse_loss += inverse_loss.item()

        avg_forward = total_forward_loss / num_updates if num_updates > 0 else 0.0
        avg_inverse = total_inverse_loss / num_updates if num_updates > 0 else 0.0
        return avg_forward, avg_inverse

    # ---------------- compute intrinsic rewards ----------------
    def compute_intrinsic(self, states, next_states, actions):
        if not self.use_icm:
            return torch.zeros(states.shape[0], device=device)
        with torch.no_grad():
            intr_raw, _, _ = self.icm(states, next_states, actions)
            intr = intr_raw.detach().squeeze().to(device)
            mean = intr.mean()
            std = intr.std(unbiased=False) + 1e-8
            intr_norm = (intr - mean) / std
            intr_pos = torch.relu(intr_norm)
        return intr_pos

    # ---------------- compute extrinsic advantages ----------------
    def compute_extrinsic_advantages(self, extrinsic_rewards, state_values, is_terminals):
        T = len(extrinsic_rewards)
        advantages = torch.zeros(T, device=device)
        deltas = torch.zeros(T, device=device)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = state_values[t + 1]
            delta = extrinsic_rewards[t] + self.gamma * next_value * (1 - is_terminals[t]) - state_values[t]
            deltas[t] = delta
            last_gae = delta + self.gamma * self.gae_lambda * (1 - is_terminals[t]) * last_gae
            advantages[t] = last_gae
        if advantages.std().item() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        return advantages, deltas

    # ---------------- discounted returns helper ----------------
    def _compute_discounted_returns_from(self, rewards, is_terminals):
        N = len(rewards)
        R = torch.zeros_like(rewards, device=device)
        Rt = 0.0
        for t in reversed(range(N)):
            Rt = rewards[t] + self.gamma * Rt * (1.0 - float(is_terminals[t].item() if isinstance(is_terminals[t], torch.Tensor) else is_terminals[t]))
            R[t] = Rt
        return R

    # ---------------- main update ----------------
    def update(self):
        if len(self.buffer.rewards) == 0:
            return

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        extrinsic_rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(device)

        N = len(extrinsic_rewards)

        # compute intrinsic
        if self.use_icm and len(self.buffer.next_states) > 0:
            old_next_states = torch.squeeze(torch.stack(self.buffer.next_states, dim=0)).detach().to(device)
            intrinsic_icm_pos = self.compute_intrinsic(old_states, old_next_states, old_actions)
            avg_fwd, avg_inv = self.icm_update(old_states, old_next_states, old_actions)
        else:
            intrinsic_icm_pos = torch.zeros(N, device=device)
            avg_fwd, avg_inv = 0.0, 0.0

        # ---------------- Post-rollout extrinsic statistics ----------------
        # Tính 1 lần / update, dùng buffer CHRONOLOGICAL (chưa shuffle).
        # Bất kỳ mini-batch loop nào sau này đều phải nằm SAU dòng này.
        advantages_ext, deltas_ext = self.compute_extrinsic_advantages(
            extrinsic_rewards, old_state_values, is_terminals
        )

        # ---------------- Meta-update (Learning-Progress) ----------------
        meta_loss_value = 0.0
        beta_value_scalar = 1.0

        if self.meta_use_progress:
            # 1. Đọc δ_ext đã tính sẵn ở post-rollout step (KHÔNG gọi lại GAE)
            delta_abs = deltas_ext.abs().detach()
            delta_max = delta_abs.max()

            # 2. β(s) hiện tại
            if self.use_state_dependent_beta:
                beta_for_loss = self.beta_net(old_states).squeeze()
            else:
                beta_for_loss = torch.exp(self.beta_log)

            # 3. Target từ |δ| (cold-start fallback dựa trên extrinsic reward signal)
            # Triết lý: nếu batch không có reward ngoại tại nào, không tồn tại
            # "extrinsic learning progress" — mọi biến thiên |δ| chỉ là nhiễu V_init.
            if extrinsic_rewards.max().item() <= 0.0:
                target = torch.full_like(beta_for_loss, self.beta_init)
                cold_start = True
            else:
                delta_norm = delta_abs / (delta_max + 1e-8)
                if self.use_state_dependent_beta:
                    b_min = self.beta_net.min
                    b_max = self.beta_net.max
                else:
                    b_min, b_max = 1e-3, 10.0
                target = b_min + (b_max - b_min) * delta_norm
                cold_start = False

            # 4. Progress loss (MSE regression)
            loss_progress = ((beta_for_loss - target.detach()) ** 2).mean()

            # 5. Regularization
            if self.use_state_dependent_beta:
                beta_log_vals = torch.log(beta_for_loss + 1e-8)
                reg_center = math.log(max(1e-8, self.meta_reg_beta_center))
                loss_reg = ((beta_log_vals - reg_center) ** 2).mean()
            else:
                reg_center = math.log(max(1e-8, self.meta_reg_beta_center))
                loss_reg = (self.beta_log - reg_center) ** 2

            loss_meta = self.meta_progress_weight * loss_progress + self.meta_reg_weight * loss_reg

            # 6. Optimize
            self.beta_optimizer.zero_grad()
            loss_meta.backward()

            # gradient norm logging
            total_norm = 0.0
            if self.use_state_dependent_beta:
                for p in self.beta_net.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
            else:
                if self.beta_log.grad is not None:
                    total_norm = float(self.beta_log.grad.data.norm(2).item() ** 2)
            total_norm = math.sqrt(total_norm) if total_norm > 0 else 0.0
            self.logger.log_scalar('beta_gradient_norm', total_norm)

            # clip and step
            if not self.use_state_dependent_beta:
                torch.nn.utils.clip_grad_norm_([self.beta_log], 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.beta_net.parameters(), 1.0)
            self.beta_optimizer.step()

            meta_loss_value = float(loss_meta.detach().cpu().item())

            # 7. Logging
            self.logger.log_scalar('delta_max', float(delta_max.item()))
            self.logger.log_scalar('delta_mean', float(delta_abs.mean().item()))
            self.logger.log_scalar('cold_start', float(cold_start))
            self.logger.log_scalar('loss_progress', float(loss_progress.detach().cpu().item()))
            self.logger.log_array('delta_abs', delta_abs)
            self.logger.log_array('beta_target', target.detach())

            # beta scalar
            if self.use_state_dependent_beta:
                with torch.no_grad():
                    try:
                        beta_value_scalar = float(self.beta_net(old_states).mean().detach().cpu().item())
                    except Exception:
                        beta_value_scalar = 1.0
            else:
                with torch.no_grad():
                    beta_value_scalar = float(torch.exp(self.beta_log).detach().cpu().item())

        else:
            # no meta update
            if self.use_state_dependent_beta:
                with torch.no_grad():
                    try:
                        beta_value_scalar = float(self.beta_net(old_states).mean().detach().cpu().item())
                    except Exception:
                        beta_value_scalar = 1.0
            else:
                with torch.no_grad():
                    try:
                        beta_value_scalar = float(torch.exp(self.beta_log).detach().cpu().item())
                    except Exception:
                        beta_value_scalar = 1.0

        # LOG scalar metrics
        self.logger.log_scalar('beta', beta_value_scalar)
        self.logger.log_scalar('meta_loss', meta_loss_value)

        # ---------------- sample state->beta for visualization ----------------
        if self.use_state_dependent_beta:
            self.logger.sample_and_log(old_states, 
                                      self.beta_net(old_states).squeeze().detach(),
                                      name='beta')
        else:
            # scalar beta
            beta_arr = torch.full((old_states.shape[0],), beta_value_scalar, device=device)
            self.logger.sample_and_log(old_states, beta_arr, name='beta')

        # ---------------- Combine rewards and PPO update ----------------
        if self.use_state_dependent_beta:
            with torch.no_grad():
                beta_full = self.beta_net(old_states).squeeze().to(device)
        else:
            with torch.no_grad():
                beta_full = torch.exp(self.beta_log).detach().cpu().item()

        if isinstance(beta_full, float) or isinstance(beta_full, int):
            combined_rewards = extrinsic_rewards + float(beta_full) * intrinsic_icm_pos
        else:
            combined_rewards = extrinsic_rewards + beta_full * intrinsic_icm_pos
        combined_rewards = combined_rewards.to(device)

        # GAE on combined rewards
        advantages = torch.zeros_like(combined_rewards).to(device)
        last_gae = 0.0
        for t in reversed(range(len(combined_rewards))):
            if t == len(combined_rewards) - 1:
                next_value = 0.0
            else:
                next_value = old_state_values[t + 1]
            delta = combined_rewards[t] + self.gamma * next_value * (1 - is_terminals[t]) - old_state_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - is_terminals[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + old_state_values

        if advantages.std().item() > 1e-7:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        else:
            advantages = advantages - advantages.mean()

        # PPO epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns.detach()) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # LOG averaged metrics
        try:
            avg_ext = float(extrinsic_rewards.mean().cpu().item())
            avg_int = float(intrinsic_icm_pos.mean().cpu().item())
        except Exception:
            avg_ext, avg_int = 0.0, 0.0

        self.logger.log_scalars({
            'icm_forward_loss': avg_fwd,
            'icm_inverse_loss': avg_inv,
            'avg_intrinsic_reward': avg_int,
            'avg_extrinsic_reward': avg_ext,
        })

        print(f"Update {self.logger.update_count}: AvgExt {avg_ext:.4f}, AvgInt {avg_int:.4f}, Beta {beta_value_scalar:.4f}, MetaLoss {meta_loss_value:.6f}")
        print(f"  ICM fwd: {avg_fwd:.4f}, inv: {avg_inv:.4f}")

        # copy weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # increment logger update count
        self.logger.step_update()
        
        # clear buffer
        self.buffer.clear()

    # ---------------- save / load ----------------
    def save(self, checkpoint_path):
        # main checkpoint
        save_dict = {
            'policy_state_dict': self.policy_old.state_dict(),
            'update_count': self.logger.update_count,
        }
        if self.use_icm:
            save_dict['icm_state_dict'] = self.icm.state_dict()
        if not self.use_state_dependent_beta:
            save_dict['beta_log'] = self.beta_log.detach().cpu()
            save_dict['beta_optimizer_state'] = self.beta_optimizer.state_dict()
        else:
            save_dict['beta_net_state_dict'] = self.beta_net.state_dict()
            save_dict['beta_optimizer_state'] = self.beta_optimizer.state_dict()
        
        torch.save(save_dict, checkpoint_path)
        
        # export logger data
        self.logger.export_logs(checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        if 'policy_state_dict' in checkpoint:
            self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if 'update_count' in checkpoint:
            self.logger.update_count = checkpoint['update_count']
        if self.use_icm and 'icm_state_dict' in checkpoint:
            try:
                self.icm.load_state_dict(checkpoint['icm_state_dict'])
            except Exception:
                pass
        if not self.use_state_dependent_beta and 'beta_log' in checkpoint:
            try:
                loaded_log = checkpoint['beta_log'].to(device)
                with torch.no_grad():
                    self.beta_log.data.copy_(loaded_log)
                if 'beta_optimizer_state' in checkpoint:
                    self.beta_optimizer.load_state_dict(checkpoint['beta_optimizer_state'])
            except Exception:
                pass
        elif self.use_state_dependent_beta and 'beta_net_state_dict' in checkpoint:
            try:
                self.beta_net.load_state_dict(checkpoint['beta_net_state_dict'])
                if 'beta_optimizer_state' in checkpoint:
                    self.beta_optimizer.load_state_dict(checkpoint['beta_optimizer_state'])
            except Exception:
                pass
        
        # load logger data
        self.logger.load_logs(checkpoint_path)

    # ---------------- convenience exports ----------------
    def export_logs(self, path_prefix):
        """Export logs separately (wrapper around logger.export_logs)"""
        self.logger.export_logs(path_prefix)
    
    def print_training_summary(self):
        """Print training summary (wrapper)"""
        self.logger.print_summary()