import os
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from curiosity.ride import RIDE
from collections import deque
from trajectory_logger import TrajectoryLogger

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
        else:
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        state: tensor [state_dim] or [1, state_dim]
        returns action (tensor), action_logprob (tensor), state_val (tensor)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, state_dim]

        state = state.to(device).float()
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()  # shape [1] for single sample
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        # return detached (single-sample shaped)
        return action.detach().squeeze(), action_logprob.detach().squeeze(), state_val.detach().squeeze()

    def evaluate(self, state, action):
        """
        state: [T, state_dim]
        action: [T] (discrete) or [T, action_dim] (continuous)
        returns: action_logprobs [T], state_values [T], dist_entropy [T]
        """
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


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space=False, action_std_init=0.6,
                 use_ride=True, ride_lr=0.001, ride_epochs=4, ride_batch_size=64,
                 intr_reward_strength=0.02, gae_lambda=0.95,
                 episodic_memory_size=1000, global_memory_size=50000, k_neighbors=10,
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

        # RIDE setup
        self.use_ride = use_ride
        self.intr_reward_strength = intr_reward_strength

        if self.use_ride:
            self.ride = RIDE(state_dim, action_dim,
                             episodic_memory_size=episodic_memory_size,
                             global_memory_size=global_memory_size,
                             k_neighbors=k_neighbors).to(device)
            self.optimizer_ride = torch.optim.Adam(self.ride.parameters(), lr=ride_lr)
            self.ride_epochs = ride_epochs
            self.ride_batch_size = ride_batch_size

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
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")

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

    # ---------------- trajectory recording (NEW) ----------------
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
        """
        state: numpy array/list or tensor (single observation)
        returns: action (python scalar or numpy), and appends tensors to buffer
        """
        # ensure tensor
        state_t = torch.FloatTensor(state).to(device)
        action, action_logprob, state_val = self.policy_old.act(state_t)

        # store in buffer as tensors
        self.buffer.states.append(state_t.detach().cpu())  # store CPU tensors for stacking later
        self.buffer.actions.append(action.detach().cpu())
        self.buffer.logprobs.append(action_logprob.detach().cpu())
        self.buffer.state_values.append(state_val.detach().cpu())

        # return for env step
        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return int(action.detach().cpu().item())

    def reset_episodic_memory(self):
        if self.use_ride:
            self.ride.clear_episodic_memory()

    def update_global_memory(self):
        """Call at the end of an episode to add episode states to global memory."""
        if self.use_ride and len(self.buffer.next_states) > 0:
            next_states = torch.stack(self.buffer.next_states, dim=0).cpu()  # CPU
            with torch.no_grad():
                encoded = self.ride.encode(next_states.to(device))
                self.ride.add_to_global_memory(encoded.detach())

    def ride_update(self, states, next_states, actions):
        """
        Train RIDE on batches.
        states, next_states: [N, state_dim] tensors on device
        actions: [N] LongTensor on device
        """
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

    def update(self):
        # stack buffer lists into tensors
        if len(self.buffer.states) == 0:
            return

        old_states = torch.stack(self.buffer.states, dim=0).to(device)          # [T, state_dim]
        old_actions = torch.stack(self.buffer.actions, dim=0).to(device)        # [T]
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).to(device)      # [T]
        old_state_values = torch.stack(self.buffer.state_values, dim=0).to(device)  # [T]
        old_state_values = old_state_values.view(-1)

        # compute intrinsic rewards
        intrinsic_rewards = torch.zeros(len(self.buffer.rewards), device=device)

        if self.use_ride and len(self.buffer.next_states) > 0:
            old_next_states = torch.stack(self.buffer.next_states, dim=0).to(device)  # [T, state_dim]
            with torch.no_grad():
                intr_rewards, _, _ = self.ride(old_states, old_next_states, old_actions)
                intrinsic_ride = intr_rewards.detach().squeeze().to(device)  # [T]
                # normalize
                ride_mean = intrinsic_ride.mean()
                ride_std = intrinsic_ride.std(unbiased=False) + 1e-8
                intrinsic_ride_norm = (intrinsic_ride - ride_mean) / ride_std
                intrinsic_ride_pos = torch.relu(intrinsic_ride_norm)
                intrinsic_rewards = intrinsic_ride_pos.clone()

            # train RIDE on collected transitions
            avg_forward_loss, avg_inverse_loss = self.ride_update(old_states, old_next_states, old_actions)
        else:
            avg_forward_loss = avg_inverse_loss = 0.0

        # combine extrinsic + intrinsic into combined_rewards list
        extrinsic_rewards = []
        intrinsic_rewards_list = []
        combined_rewards = []

        for idx, reward in enumerate(self.buffer.rewards):
            ext_reward = reward
            intr_reward = 0.0
            if self.use_ride and len(self.buffer.next_states) > 0:
                intr_reward = float(intrinsic_rewards[idx].cpu().item())
            combined_reward = ext_reward + self.intr_reward_strength * intr_reward

            extrinsic_rewards.append(ext_reward)
            intrinsic_rewards_list.append(intr_reward)
            combined_rewards.append(combined_reward)

        combined_rewards = torch.tensor(combined_rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(device)

        # GAE advantage calculation (use tensor ops only)
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
        # normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-7
        advantages = (advantages - adv_mean) / adv_std

        # PPO update K epochs
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

        # log
        if len(extrinsic_rewards) > 0:
            avg_ext_reward = np.mean(extrinsic_rewards)
            avg_int_reward = np.mean(intrinsic_rewards_list) if len(intrinsic_rewards_list) > 0 else 0.0
            print(f"Update - Avg Extrinsic Reward: {avg_ext_reward:.4f}, Avg Intrinsic Reward: {avg_int_reward:.6f}")

        # copy weights
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
        if self.use_ride:
            save_dict['ride_state_dict'] = self.ride.state_dict()
        
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
        if self.use_ride and 'ride_state_dict' in checkpoint:
            self.ride.load_state_dict(checkpoint['ride_state_dict'])
        
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