import math
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from curiosity.count_based import CountBasedExploration
from training_logger import TrainingLogger

################################## set device ##################################
print("============================================================================================")
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    try:
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    except:
        print("Device set to : cuda")
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## BetaNetwork ##################################
class BetaNetwork(nn.Module):
    """
    State-dependent beta network matching ICM encoder architecture
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

        self.head_out = nn.Linear(head_hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(encoding_size, head_hidden),
            nn.Tanh(),
            self.head_out,
        )

        # Initialize head so beta_psi ~= meta_reg_beta_center (= 0.01) at init
        with torch.no_grad():
            if self.head_out.bias is not None:
                nn.init.constant_(self.head_out.bias, math.log(0.01))
            nn.init.normal_(self.head_out.weight, mean=0.0, std=1e-3)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.encoder(state)
        log_beta = self.head(h).squeeze(-1)
        log_beta = torch.clamp(log_beta, math.log(self.min), math.log(self.max))
        beta = torch.exp(log_beta)
        return beta


################################## PPO Policy ##################################
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


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
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
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


################################## PPO ##################################
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6,
                 # Count-Based params
                 use_count_based=True,
                 hash_dim=32,
                 bonus_type='inverse_sqrt',
                 # GAE
                 gae_lambda=0.95,
                 # Adaptive Beta params
                 beta_lr=5e-4,
                 use_state_dependent_beta=True,
                 beta_init=0.01,
                 beta_encoding_size=256,
                 beta_num_layers=2,
                 beta_head_hidden=128,
                 beta_min=0.001,
                 beta_max=0.1,
                 # Meta options (Learning-Progress)
                 meta_use_progress=True,
                 meta_progress_weight=1.0,
                 meta_reg_weight=1e-3,
                 meta_reg_beta_center=0.01,
                 # Logging
                 sample_states_per_update=256,
                 sample_every_n_updates=1):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        self.buffer = RolloutBuffer()

        # Actor-Critic
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # Count-Based Exploration
        self.use_count_based = use_count_based

        if self.use_count_based:
            self.count_based = CountBasedExploration(
                state_dim=state_dim,
                hash_dim=hash_dim,
                bonus_type=bonus_type
            )

        # Adaptive Beta
        self.use_state_dependent_beta = use_state_dependent_beta
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        if not use_state_dependent_beta:
            # Scalar beta (log parameterization)
            self.beta_log = nn.Parameter(torch.tensor([math.log(beta_init)], dtype=torch.float32, device=device))
            self.beta_optimizer = torch.optim.Adam([self.beta_log], lr=beta_lr)
        else:
            # State-dependent beta network
            self.beta_net = BetaNetwork(
                state_dim,
                encoding_size=beta_encoding_size,
                num_layers=beta_num_layers,
                head_hidden=beta_head_hidden,
                min_beta=beta_min,
                max_beta=beta_max
            ).to(device)
            self.beta_optimizer = torch.optim.Adam(self.beta_net.parameters(), lr=beta_lr, weight_decay=1e-6)

        # Meta options (Learning-Progress)
        self.meta_use_progress = meta_use_progress
        self.meta_progress_weight = meta_progress_weight
        self.meta_reg_weight = meta_reg_weight
        self.meta_reg_beta_center = float(meta_reg_beta_center)
        self.beta_init = float(beta_init)

        # Logger
        self.logger = TrainingLogger(
            sample_states_per_update=sample_states_per_update,
            sample_every_n_updates=sample_every_n_updates,
            auto_convert_to_numpy=True
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = max(self.action_std - action_std_decay_rate, min_action_std)
            self.set_action_std(self.action_std)

    def select_action(self, state):
        state_t = torch.FloatTensor(state).to(device)
        
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state_t)

        self.buffer.states.append(state_t)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

    def _compute_discounted_returns_from(self, rewards, is_terminals):
        """Compute discounted future returns from each timestep"""
        N = len(rewards)
        R = torch.zeros_like(rewards, device=device)
        Rt = 0.0
        for t in reversed(range(N)):
            Rt = rewards[t] + self.gamma * Rt * (1.0 - float(is_terminals[t].item() if isinstance(is_terminals[t], torch.Tensor) else is_terminals[t]))
            R[t] = Rt
        return R

    def compute_extrinsic_advantages(self, extrinsic_rewards, state_values, is_terminals):
        """GAE on extrinsic rewards. Returns (normalized advantages, raw deltas)."""
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

    def update(self):
        if len(self.buffer.rewards) == 0:
            return

        # Convert buffer to tensors
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        extrinsic_rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(device)

        N = len(extrinsic_rewards)

        # Compute intrinsic rewards using count-based
        intrinsic_raw = np.zeros(N)
        
        if self.use_count_based:
            states_np = old_states.cpu().numpy()
            intrinsic_raw = self.count_based.update(states_np)
            
            # Normalize intrinsic rewards
            if len(intrinsic_raw) > 0:
                intr_mean = intrinsic_raw.mean()
                intr_std = intrinsic_raw.std() + 1e-8
                intrinsic_raw = (intrinsic_raw - intr_mean) / intr_std
                intrinsic_raw = np.maximum(intrinsic_raw, 0)  # ReLU
        
        intrinsic_rewards = torch.tensor(intrinsic_raw, dtype=torch.float32).to(device)

        # ============ Post-rollout extrinsic statistics ============
        # Tính 1 lần / update, dùng buffer CHRONOLOGICAL (chưa shuffle).
        # Bất kỳ mini-batch loop nào sau này đều phải nằm SAU dòng này.
        advantages_ext, deltas_ext = self.compute_extrinsic_advantages(
            extrinsic_rewards, old_state_values, is_terminals
        )

        # ============ Meta-Update (Learning-Progress) ============
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
                target = torch.full_like(beta_for_loss, self.meta_reg_beta_center)
                cold_start = True
            else:
                delta_norm = delta_abs / (delta_max + 1e-8)
                if self.use_state_dependent_beta:
                    b_min = self.beta_net.min
                    b_max = self.beta_net.max
                else:
                    b_min, b_max = self.beta_min, self.beta_max
                target = b_min + (b_max - b_min) * delta_norm
                cold_start = False

            # 4. Progress loss (MSE regression)
            loss_progress = ((beta_for_loss - target.detach()) ** 2).mean()

            # 5. Regularization: keep beta near center
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
            # No meta update - just get current beta
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

        # Log beta
        self.logger.log_scalar('beta', beta_value_scalar)
        self.logger.log_scalar('meta_loss', meta_loss_value)

        # Sample state->beta pairs for visualization
        if self.use_state_dependent_beta:
            self.logger.sample_and_log(
                old_states,
                self.beta_net(old_states).squeeze().detach(),
                name='beta'
            )
        else:
            beta_arr = torch.full((old_states.shape[0],), beta_value_scalar, device=device)
            self.logger.sample_and_log(old_states, beta_arr, name='beta')

        # ============ Combine Rewards ============
        # Get beta for all states
        if self.use_state_dependent_beta:
            with torch.no_grad():
                beta_full = self.beta_net(old_states).squeeze().to(device)
        else:
            with torch.no_grad():
                beta_full = torch.exp(self.beta_log).detach().cpu().item()

        # Combined rewards
        if isinstance(beta_full, float) or isinstance(beta_full, int):
            combined_rewards = extrinsic_rewards + float(beta_full) * intrinsic_rewards
        else:
            combined_rewards = extrinsic_rewards + beta_full * intrinsic_rewards
        combined_rewards = combined_rewards.to(device)

        # ============ GAE Computation ============
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

        # Normalize advantages
        if advantages.std().item() > 1e-7:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        else:
            advantages = advantages - advantages.mean()

        # ============ PPO Update ============
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

        # ============ Logging ============
        try:
            avg_ext = float(extrinsic_rewards.mean().cpu().item())
            avg_int = float(intrinsic_rewards.mean().cpu().item())
        except Exception:
            avg_ext, avg_int = 0.0, 0.0

        # Log metrics
        self.logger.log_scalars({
            'avg_extrinsic_reward': avg_ext,
            'avg_intrinsic_reward': avg_int,
        })

        # Count-based statistics
        if self.use_count_based:
            count_stats = self.count_based.get_statistics()
            self.logger.log_scalars({
                'count_unique_states': count_stats['unique_states'],
                'count_avg_count': count_stats['avg_count'],
                'count_max_count': count_stats['max_count'],
                'count_total_visits': count_stats['total_visits']
            })

        # Print summary
        print(f"Update {self.logger.update_count}: AvgExt {avg_ext:.4f}, AvgInt {avg_int:.4f}, Beta {beta_value_scalar:.4f}, MetaLoss {meta_loss_value:.6f}")
        if self.use_count_based:
            count_stats = self.count_based.get_statistics()
            print(f"  Count Stats - Unique: {count_stats['unique_states']}, Avg: {count_stats['avg_count']:.2f}, Max: {count_stats['max_count']}")

        # Copy weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Increment update counter
        self.logger.step_update()

        # Clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        save_dict = {
            'policy_state_dict': self.policy_old.state_dict(),
            'update_count': self.logger.update_count,
        }
        
        # Save beta
        if not self.use_state_dependent_beta:
            save_dict['beta_log'] = self.beta_log.detach().cpu()
            save_dict['beta_optimizer_state'] = self.beta_optimizer.state_dict()
        else:
            save_dict['beta_net_state_dict'] = self.beta_net.state_dict()
            save_dict['beta_optimizer_state'] = self.beta_optimizer.state_dict()
        
        # Save count-based statistics (optional)
        if self.use_count_based:
            save_dict['count_stats'] = self.count_based.get_statistics()
        
        torch.save(save_dict, checkpoint_path)
        
        # Export logger data
        self.logger.export_logs(checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
        if 'policy_state_dict' in checkpoint:
            self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        if 'update_count' in checkpoint:
            self.logger.update_count = checkpoint['update_count']
        
        # Load beta
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
        
        # Load logger data
        self.logger.load_logs(checkpoint_path)

    def export_logs(self, path_prefix):
        """Export logs separately"""
        self.logger.export_logs(path_prefix)

    def print_training_summary(self):
        """Print training summary"""
        self.logger.print_summary()