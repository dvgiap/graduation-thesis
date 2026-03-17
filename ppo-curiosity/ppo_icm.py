import os
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from curiosity.icm import ICM
from trajectory_logger import TrajectoryLogger

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
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
        if has_continuous_action_space :
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
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

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

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6,
                 use_icm=True, icm_lr=0.001, icm_epochs=4, icm_batch_size=64, intr_reward_strength=0.02,
                 gae_lambda=0.95,
                 # Trajectory logging
                 enable_trajectory_logging=True,
                 trajectory_grid_shape=None):  # (width, height) for grid-based envs

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # ICM setup
        self.use_icm = use_icm
        self.intr_reward_strength = intr_reward_strength

        if self.use_icm:
            self.icm = ICM(state_dim, action_dim).to(device)
            self.optimizer_icm = torch.optim.Adam(self.icm.parameters(), lr=icm_lr)
            self.icm_epochs = icm_epochs
            self.icm_batch_size = icm_batch_size

        self.enable_trajectory_logging = enable_trajectory_logging
        self.trajectory_grid_shape = trajectory_grid_shape
        if enable_trajectory_logging:
            self.trajectory_logger = TrajectoryLogger()
        else:
            self.trajectory_logger = None

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def record_position(self, pos, episode=None, timestep=None, as_state=False):
        """
        Record agent position for trajectory visualization.
        
        Args:
            pos: Position as (x, y) tuple, or state vector if as_state=True
            episode: Current episode number
            timestep: Current timestep within episode
            as_state: If True, extract position from state vector
        
        Examples:
            # Direct position recording
            ppo.record_position((agent_x, agent_y), episode=0, timestep=100)
            
            # Extract from state (for grid-based environments)
            ppo.record_position(state, episode=0, timestep=100, as_state=True)
        """
        if self.trajectory_logger is not None:
            self.trajectory_logger.record_position(
                pos, 
                episode=episode, 
                timestep=timestep,
                as_state=as_state,
                grid_shape=self.trajectory_grid_shape
            )

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def icm_update(self, states, next_states, actions):
        """Update ICM module"""
        total_forward_loss = 0
        total_inverse_loss = 0
        num_updates = 0

        dataset_size = states.shape[0]

        for _ in range(self.icm_epochs):
            # Shuffle indices
            indices = np.random.permutation(dataset_size)

            for start_idx in range(0, dataset_size, self.icm_batch_size):
                num_updates += 1

                # Get batch indices
                batch_indices = indices[start_idx:start_idx + self.icm_batch_size]

                # Get batch data
                batch_states = states[batch_indices]
                batch_next_states = next_states[batch_indices]
                batch_actions = actions[batch_indices]

                # Forward pass through ICM
                _, inverse_loss, forward_loss = self.icm(batch_states, batch_next_states, batch_actions)

                # Combined ICM loss (matching paper: β=0.2 for forward, 1-β=0.8 for inverse)
                icm_loss = 0.2 * forward_loss + 0.8 * inverse_loss

                # Backward pass
                self.optimizer_icm.zero_grad()
                icm_loss.backward()
                self.optimizer_icm.step()

                total_forward_loss += forward_loss.item()
                total_inverse_loss += inverse_loss.item()

        avg_forward_loss = total_forward_loss / num_updates if num_updates > 0 else 0
        avg_inverse_loss = total_inverse_loss / num_updates if num_updates > 0 else 0

        return avg_forward_loss, avg_inverse_loss

    def update(self):
        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # ICM: compute intrinsic rewards
        intrinsic_rewards = torch.zeros(len(self.buffer.rewards)).to(device)

        if self.use_icm and len(self.buffer.next_states) > 0:
            # Collect next states
            old_next_states = torch.squeeze(torch.stack(self.buffer.next_states, dim=0)).detach().to(device)

            # Compute intrinsic rewards from ICM
            with torch.no_grad():
                intr_rewards, _, _ = self.icm(old_states, old_next_states, old_actions)
                # intr_rewards assumed to be a tensor of shape (batch,) or (batch,1)
                intrinsic_icm = intr_rewards.detach().squeeze().to(device)

                # ---------- Normalization of ICM intrinsic (your snippet) ----------
                icm_mean = intrinsic_icm.mean()
                icm_std = intrinsic_icm.std(unbiased=False) + 1e-8
                intrinsic_icm_norm = (intrinsic_icm - icm_mean) / icm_std
                intrinsic_icm_pos = torch.relu(intrinsic_icm_norm)
                # -------------------------------------------------------------------

                # Make sure intrinsic_rewards has same length and device
                # If intrinsic_icm_pos is scalar for some reason, expand appropriately
                intrinsic_rewards = intrinsic_icm_pos.clone()

            # Update ICM parameters (training)
            avg_forward_loss, avg_inverse_loss = self.icm_update(old_states, old_next_states, old_actions)

        # Combine extrinsic and intrinsic rewards
        extrinsic_rewards = []
        intrinsic_rewards_list = []
        combined_rewards = []

        for idx, reward in enumerate(self.buffer.rewards):
            ext_reward = reward
            intr_reward = 0.0

            if self.use_icm and len(self.buffer.next_states) > 0:
                # intrinsic_rewards could be torch tensor; get scalar
                try:
                    intr_reward = float(intrinsic_rewards[idx].cpu().item())
                except:
                    # fallback if intrinsic_rewards is 0-d
                    intr_reward = float(intrinsic_rewards.cpu().item())

            combined_reward = ext_reward + self.intr_reward_strength * intr_reward

            extrinsic_rewards.append(ext_reward)
            intrinsic_rewards_list.append(intr_reward)
            combined_rewards.append(combined_reward)

        # Convert to tensors
        combined_rewards = torch.tensor(combined_rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(device)

        # Compute GAE advantages
        advantages = torch.zeros_like(combined_rewards).to(device)
        last_gae = 0

        for t in reversed(range(len(combined_rewards))):
            if t == len(combined_rewards) - 1:
                next_value = 0
            else:
                next_value = old_state_values[t + 1]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = combined_rewards[t] + self.gamma * next_value * (1 - is_terminals[t]) - old_state_values[t]

            # GAE: A_t = δ_t + γ * λ * A_{t+1}
            last_gae = delta + self.gamma * self.gae_lambda * (1 - is_terminals[t]) * last_gae
            advantages[t] = last_gae

        # Compute returns: R_t = A_t + V(s_t)
        returns = advantages + old_state_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with returns tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Print average rewards
        if len(extrinsic_rewards) > 0:
            avg_ext_reward = np.mean(extrinsic_rewards)
            avg_int_reward = np.mean(intrinsic_rewards_list) if len(intrinsic_rewards_list) > 0 else 0.0
            print(f"Update - Avg Extrinsic Reward: {avg_ext_reward:.4f}, Avg Intrinsic Reward: {avg_int_reward:.4f}")

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        """
        Save model checkpoint and trajectory data.
        
        Creates:
            - checkpoint_path: Main PyTorch checkpoint
            - checkpoint_path.trajectory.npz: Trajectory data (if enabled)
        """
        save_dict = {
            'policy_state_dict': self.policy_old.state_dict(),
        }
        if self.use_icm:
            save_dict['icm_state_dict'] = self.icm.state_dict()
        
        # Add trajectory data to checkpoint (optional, also saved separately)
        if self.trajectory_logger is not None:
            save_dict['trajectory_np'] = self.trajectory_logger.get_trajectory_array()
        
        torch.save(save_dict, checkpoint_path)
        
        # Export trajectory logger data
        if self.trajectory_logger is not None:
            traj_path = checkpoint_path + '.trajectory.npz'
            self.trajectory_logger.save_npz(traj_path)

    def load(self, checkpoint_path):
        """Load model checkpoint and trajectory data."""
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if self.use_icm and 'icm_state_dict' in checkpoint:
            self.icm.load_state_dict(checkpoint['icm_state_dict'])
        
        # Load trajectory logger data
        if self.trajectory_logger is not None:
            # Try loading from checkpoint first
            if 'trajectory_np' in checkpoint:
                try:
                    traj_np = checkpoint['trajectory_np']
                    if isinstance(traj_np, np.ndarray) and traj_np.size > 0:
                        self.trajectory_logger.trajectory = [
                            tuple(map(int, row.tolist())) for row in traj_np
                        ]
                except Exception:
                    pass
            
            # Also try loading from separate NPZ file
            traj_path = checkpoint_path + '.trajectory.npz'
            if os.path.exists(traj_path):
                try:
                    self.trajectory_logger.load_npz(traj_path)
                except Exception:
                    pass

    # ---------------- Trajectory utilities (wrappers) ----------------
    def save_trajectory_csv(self, csv_path):
        """Save trajectory data to CSV file."""
        if self.trajectory_logger is not None:
            self.trajectory_logger.save_csv(csv_path)
        else:
            print("Warning: Trajectory logging is disabled")

    def save_trajectory_heatmap(self, out_path, **kwargs):
        """
        Generate and save trajectory heatmap.
        
        Args:
            out_path: Output PNG file path
            **kwargs: Additional arguments passed to save_heatmap_png
                     (start_idx, end_idx, grid_shape, start_episode, 
                      end_episode, normalize, cmap, annotate_max, etc.)
        
        Examples:
            # Basic heatmap
            ppo.save_trajectory_heatmap("heatmap.png")
            
            # Last 1000 positions
            ppo.save_trajectory_heatmap("recent.png", start_idx=-1000)
            
            # Episodes 10-50 with max annotation
            ppo.save_trajectory_heatmap("early.png", 
                                       start_episode=10, 
                                       end_episode=50,
                                       annotate_max=True)
        """
        if self.trajectory_logger is not None:
            # Use grid_shape from PPO config if not provided
            if 'grid_shape' not in kwargs and self.trajectory_grid_shape is not None:
                kwargs['grid_shape'] = self.trajectory_grid_shape
            self.trajectory_logger.save_heatmap_png(out_path, **kwargs)
        else:
            print("Warning: Trajectory logging is disabled")

    def get_trajectory_statistics(self, **kwargs):
        """
        Get trajectory statistics.
        
        Returns dict with keys: total_positions, num_episodes, unique_positions,
                                most_visited_position, max_visit_count, x_range, y_range
        """
        if self.trajectory_logger is not None:
            return self.trajectory_logger.get_statistics(**kwargs)
        else:
            return None

    def print_trajectory_statistics(self, **kwargs):
        """Print trajectory statistics to console."""
        if self.trajectory_logger is not None:
            self.trajectory_logger.print_statistics(**kwargs)
        else:
            print("Warning: Trajectory logging is disabled")