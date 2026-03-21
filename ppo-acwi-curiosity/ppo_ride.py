import math
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from curiosity.ride import RIDE
from training_logger import TrainingLogger

################################## set device ##################################
print("============================================================================================")
device = torch.device('cpu')
if torch.cuda.is_available():
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

        self.head = nn.Sequential(
            nn.Linear(encoding_size, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1)
        )
        
        # Initialize head to output beta ~1.0 initially
        with torch.no_grad():
            try:
                nn.init.constant_(self.head[-1].bias, math.log(1.0))
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
            self.action_var = torch.full((self.action_var.shape[0],), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        state = state.to(device).float()
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach().squeeze(), action_logprob.detach().squeeze(), state_val.detach().squeeze()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            if action.dim() == 1:
                action = action.view(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state).squeeze(-1)

        return action_logprobs, state_values, dist_entropy


################################## PPO ##################################
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space=False, action_std_init=0.6,
                 # RIDE params
                 use_ride=True,
                 ride_lr=0.001,
                 ride_epochs=4,
                 ride_batch_size=64,
                 intr_reward_strength=0.02,
                 episodic_memory_size=1000,
                 global_memory_size=50000,
                 k_neighbors=10,
                 # GAE
                 gae_lambda=0.95,
                 # Adaptive Beta params
                 beta_lr=5e-4,
                 use_state_dependent_beta=True,
                 beta_init=1.0,
                 beta_encoding_size=256,
                 beta_num_layers=2,
                 beta_head_hidden=128,
                 beta_min=0.1,
                 beta_max=2.0,
                 # Meta options
                 meta_use_correlation=True,
                 meta_corr_weight=1.0,
                 meta_reg_weight=1e-3,
                 meta_corr_beta_center=1.0,
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

        # RIDE setup
        self.use_ride = use_ride
        self.intr_reward_strength = intr_reward_strength
        self.ride_epochs = ride_epochs
        self.ride_batch_size = ride_batch_size

        if self.use_ride:
            self.ride = RIDE(state_dim, action_dim,
                           episodic_memory_size=episodic_memory_size,
                           global_memory_size=global_memory_size,
                           k_neighbors=k_neighbors).to(device)
            self.optimizer_ride = torch.optim.Adam(self.ride.parameters(), lr=ride_lr)

        # Adaptive Beta
        self.use_state_dependent_beta = use_state_dependent_beta
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

        # Meta options
        self.meta_use_correlation = meta_use_correlation
        self.meta_corr_weight = meta_corr_weight
        self.meta_reg_weight = meta_reg_weight
        self.meta_corr_beta_center = float(meta_corr_beta_center)

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

    def select_action(self, state):
        state_t = torch.FloatTensor(state).to(device)
        action, action_logprob, state_val = self.policy_old.act(state_t)

        # Store in buffer as CPU tensors
        self.buffer.states.append(state_t.detach().cpu())
        self.buffer.actions.append(action.detach().cpu())
        self.buffer.logprobs.append(action_logprob.detach().cpu())
        self.buffer.state_values.append(state_val.detach().cpu())

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return int(action.detach().cpu().item())

    def reset_episodic_memory(self):
        """Clear episodic memory at the start of each episode"""
        if self.use_ride:
            self.ride.clear_episodic_memory()

    def update_global_memory(self):
        """Add episode states to global memory at episode end"""
        if self.use_ride and len(self.buffer.next_states) > 0:
            next_states = torch.stack(self.buffer.next_states, dim=0).cpu()
            with torch.no_grad():
                encoded = self.ride.encode(next_states.to(device))
                self.ride.add_to_global_memory(encoded.detach())

    def ride_update(self, states, next_states, actions):
        """Train RIDE on collected transitions"""
        total_forward_loss = 0.0
        total_inverse_loss = 0.0
        num_updates = 0
        dataset_size = states.shape[0]
        beta = 0.2  # forward weight

        for _ in range(self.ride_epochs):
            indices = np.random.permutation(dataset_size)
            for start_idx in range(0, dataset_size, self.ride_batch_size):
                num_updates += 1
                batch_idx = indices[start_idx:start_idx + self.ride_batch_size]
                batch_idx = torch.LongTensor(batch_idx).to(device)
                batch_states = states[batch_idx]
                batch_next_states = next_states[batch_idx]
                batch_actions = actions[batch_idx]

                _, inverse_loss, forward_loss = self.ride(batch_states, batch_next_states, batch_actions)
                ride_loss = beta * forward_loss + (1.0 - beta) * inverse_loss

                self.optimizer_ride.zero_grad()
                ride_loss.backward()
                self.optimizer_ride.step()

                total_forward_loss += forward_loss.item()
                total_inverse_loss += inverse_loss.item()

        avg_forward_loss = total_forward_loss / num_updates if num_updates > 0 else 0.0
        avg_inverse_loss = total_inverse_loss / num_updates if num_updates > 0 else 0.0
        return avg_forward_loss, avg_inverse_loss

    def _compute_discounted_returns_from(self, rewards, is_terminals):
        """Compute discounted future returns from each timestep"""
        N = len(rewards)
        R = torch.zeros_like(rewards, device=device)
        Rt = 0.0
        for t in reversed(range(N)):
            Rt = rewards[t] + self.gamma * Rt * (1.0 - float(is_terminals[t].item() if isinstance(is_terminals[t], torch.Tensor) else is_terminals[t]))
            R[t] = Rt
        return R

    def update(self):
        if len(self.buffer.states) == 0:
            return

        # Stack buffer to tensors
        old_states = torch.stack(self.buffer.states, dim=0).to(device)
        old_actions = torch.stack(self.buffer.actions, dim=0).to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).to(device)
        old_state_values = torch.stack(self.buffer.state_values, dim=0).to(device)
        old_state_values = old_state_values.view(-1)
        extrinsic_rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(device)

        N = len(extrinsic_rewards)

        # Compute intrinsic rewards using RIDE
        intrinsic_rewards = torch.zeros(N, device=device)

        if self.use_ride and len(self.buffer.next_states) > 0:
            old_next_states = torch.stack(self.buffer.next_states, dim=0).to(device)
            
            with torch.no_grad():
                intr_rewards, _, _ = self.ride(old_states, old_next_states, old_actions)
                intrinsic_ride = intr_rewards.detach().squeeze().to(device)
                
                # Normalize
                ride_mean = intrinsic_ride.mean()
                ride_std = intrinsic_ride.std(unbiased=False) + 1e-8
                intrinsic_ride_norm = (intrinsic_ride - ride_mean) / ride_std
                intrinsic_ride_pos = torch.relu(intrinsic_ride_norm)
                intrinsic_rewards = intrinsic_ride_pos.clone()

            # Train RIDE
            avg_forward_loss, avg_inverse_loss = self.ride_update(old_states, old_next_states, old_actions)
        else:
            avg_forward_loss = avg_inverse_loss = 0.0

        # ============ Meta-Update (Correlation-based Beta Adaptation) ============
        meta_loss_value = 0.0
        beta_value_scalar = 1.0

        if self.meta_use_correlation:
            # Compute future extrinsic returns
            R_ext_from_t = self._compute_discounted_returns_from(extrinsic_rewards, is_terminals)

            # Get beta outputs
            if self.use_state_dependent_beta:
                beta_for_loss = self.beta_net(old_states).squeeze()
                b_intr = beta_for_loss * intrinsic_rewards
            else:
                beta_for_loss = torch.exp(self.beta_log)
                b_intr = beta_for_loss * intrinsic_rewards

            # Normalize signals
            if b_intr.numel() > 1:
                b_mean = b_intr.mean()
                b_std = b_intr.std(unbiased=False) + 1e-8
                b_intr_norm = (b_intr - b_mean) / b_std
            else:
                b_intr_norm = b_intr - b_intr.mean()

            if R_ext_from_t.numel() > 1:
                r_mean = R_ext_from_t.mean()
                r_std = R_ext_from_t.std(unbiased=False) + 1e-8
                R_ext_norm = (R_ext_from_t - r_mean) / r_std
            else:
                R_ext_norm = R_ext_from_t - R_ext_from_t.mean()

            # Correlation loss (maximize positive correlation)
            loss_corr = -(b_intr_norm * R_ext_norm).mean()

            # Regularization: keep beta near center
            if self.use_state_dependent_beta:
                beta_vals = torch.log(beta_for_loss + 1e-8)
                reg_center = math.log(max(1e-8, self.meta_corr_beta_center))
                loss_reg = ((beta_vals - reg_center) ** 2).mean()
            else:
                reg_center = math.log(max(1e-8, self.meta_corr_beta_center))
                loss_reg = (self.beta_log - reg_center) ** 2

            # Total meta loss
            loss_meta = self.meta_corr_weight * loss_corr + self.meta_reg_weight * loss_reg

            # Optimize beta
            self.beta_optimizer.zero_grad()
            loss_meta.backward()

            # Compute gradient norm
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

            # Log gradient norm
            self.logger.log_scalar('beta_gradient_norm', total_norm)

            # Clip and step
            if not self.use_state_dependent_beta:
                torch.nn.utils.clip_grad_norm_([self.beta_log], 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.beta_net.parameters(), 1.0)
            self.beta_optimizer.step()

            meta_loss_value = float(loss_meta.detach().cpu().item())

            # Get beta scalar for logging
            if self.use_state_dependent_beta:
                with torch.no_grad():
                    try:
                        beta_value_scalar = float(self.beta_net(old_states).mean().detach().cpu().item())
                    except Exception:
                        beta_value_scalar = 1.0
            else:
                with torch.no_grad():
                    beta_value_scalar = float(torch.exp(self.beta_log).detach().cpu().item())

            # Log array metrics for correlation analysis
            self.logger.log_array('b_intr', b_intr.detach())
            self.logger.log_array('R_ext', R_ext_from_t.detach())

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
            combined_rewards = extrinsic_rewards + self.intr_reward_strength * float(beta_full) * intrinsic_rewards
        else:
            combined_rewards = extrinsic_rewards + self.intr_reward_strength * beta_full * intrinsic_rewards
        combined_rewards = combined_rewards.to(device)

        # ============ GAE Computation ============
        T = len(combined_rewards)
        advantages = torch.zeros(T, device=device)
        last_gae = torch.tensor(0.0, device=device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.tensor(0.0, device=device)
            else:
                next_value = old_state_values[t + 1]
            delta = combined_rewards[t] + self.gamma * next_value * (1 - is_terminals[t]) - old_state_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - is_terminals[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + old_state_values

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-7
        advantages = (advantages - adv_mean) / adv_std

        # ============ PPO Update ============
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

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
            'ride_forward_loss': avg_forward_loss,
            'ride_inverse_loss': avg_inverse_loss,
        })

        # RIDE memory statistics
        if self.use_ride:
            episodic_size = len(self.ride.episodic_memory) if hasattr(self.ride, 'episodic_memory') else 0
            global_size = len(self.ride.global_memory) if hasattr(self.ride, 'global_memory') else 0
            self.logger.log_scalars({
                'ride_episodic_memory_size': episodic_size,
                'ride_global_memory_size': global_size,
            })

        # Print summary
        print(f"Update {self.logger.update_count}: AvgExt {avg_ext:.4f}, AvgInt {avg_int:.4f}, Beta {beta_value_scalar:.4f}, MetaLoss {meta_loss_value:.6f}")
        if self.use_ride:
            print(f"  RIDE - Fwd: {avg_forward_loss:.4f}, Inv: {avg_inverse_loss:.4f}, EpiMem: {episodic_size}, GlobalMem: {global_size}")

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
        
        # Save RIDE
        if self.use_ride:
            save_dict['ride_state_dict'] = self.ride.state_dict()
        
        # Save beta
        if not self.use_state_dependent_beta:
            save_dict['beta_log'] = self.beta_log.detach().cpu()
            save_dict['beta_optimizer_state'] = self.beta_optimizer.state_dict()
        else:
            save_dict['beta_net_state_dict'] = self.beta_net.state_dict()
            save_dict['beta_optimizer_state'] = self.beta_optimizer.state_dict()
        
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
        
        # Load RIDE
        if self.use_ride and 'ride_state_dict' in checkpoint:
            try:
                self.ride.load_state_dict(checkpoint['ride_state_dict'])
            except Exception:
                pass
        
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