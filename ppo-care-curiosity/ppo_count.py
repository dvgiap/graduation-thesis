import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from curiosity.count_based import CountBasedExploration
from training_logger import TrainingLogger
from curiosity.care_module import CAREModule

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
                 beta_init=None,
                 beta_encoding_size=256,
                 beta_num_layers=2,
                 beta_head_hidden=128,
                 beta_min=5e-4,
                 beta_max=5e-2,
                 # Meta options (correlation-based β scaling)
                 meta_use_progress=True,
                 meta_reg_weight=1e-3,
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

        # Logger (created first so it can be passed to CAREModule)
        self.logger = TrainingLogger(
            sample_states_per_update=sample_states_per_update,
            sample_every_n_updates=sample_every_n_updates,
            auto_convert_to_numpy=True
        )

        # CARE module (BetaNetwork + meta-loss + target network)
        self.care = CAREModule(
            state_dim=state_dim,
            beta_min=beta_min,
            beta_max=beta_max,
            beta_0=beta_init,
            encoding_size=beta_encoding_size,
            num_layers=beta_num_layers,
            head_hidden=beta_head_hidden,
            lr=beta_lr,
            reg_weight=meta_reg_weight,
            use_state_dependent=use_state_dependent_beta,
            use_progress=meta_use_progress,
            target_tau=0.01,
            logger=self.logger,
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

    def compute_extrinsic_advantages(self, extrinsic_rewards, state_values, is_terminals, normalize=True):
        """GAE on extrinsic-only rewards.

        normalize=True  → z-score advantages (for PPO policy update path)
        normalize=False → raw advantages (for CARE: preserves signal-strength info)
        """
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
        if normalize:
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

        # ============ CARE: meta-update β + combine rewards ============
        if self.care.use_progress:
            adv_ext_raw, _ = self.compute_extrinsic_advantages(
                extrinsic_rewards, old_state_values, is_terminals, normalize=False
            )
            meta_loss_value = self.care.update(
                old_states, intrinsic_rewards.detach(), adv_ext_raw.detach()
            )
        else:
            meta_loss_value = 0.0

        beta_value_scalar = self.care.get_beta_scalar(old_states)
        self.logger.log_scalar('beta', beta_value_scalar)
        self.logger.log_scalar('meta_loss', meta_loss_value)
        self.care.sample_and_log(old_states)

        combined_rewards = self.care.combine(extrinsic_rewards, intrinsic_rewards, old_states)
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
        if self.use_count_based:
            save_dict['count_stats'] = self.count_based.get_statistics()
        torch.save(save_dict, checkpoint_path)

        care_path = checkpoint_path.replace('.pth', '') + '_care.pth'
        self.care.save(care_path)

        self.logger.export_logs(checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        if 'policy_state_dict' in checkpoint:
            self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if 'update_count' in checkpoint:
            self.logger.update_count = checkpoint['update_count']

        care_path = checkpoint_path.replace('.pth', '') + '_care.pth'
        self.care.load(care_path)

        self.logger.load_logs(checkpoint_path)

    def export_logs(self, path_prefix):
        """Export logs separately"""
        self.logger.export_logs(path_prefix)

    def print_training_summary(self):
        """Print training summary"""
        self.logger.print_summary()