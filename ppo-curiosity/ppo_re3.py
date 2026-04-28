import os
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from curiosity.re3 import RE3
from trajectory_logger import TrajectoryLogger

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

        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, action_dim), nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
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


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6,
                 use_re3=True, re3_encoding_size=64, re3_num_layers=2,
                 re3_k=3, re3_buffer_size=10000,
                 intr_reward_strength=0.02,
                 gae_lambda=0.95,
                 # Trajectory logging
                 enable_trajectory_logging=True,
                 trajectory_grid_shape=None):

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

        # RE3 (encoder is fixed and random; no parameters to train)
        self.use_re3 = use_re3
        self.intr_reward_strength = intr_reward_strength

        if self.use_re3:
            self.re3 = RE3(state_dim,
                           encoding_size=re3_encoding_size,
                           num_layers=re3_num_layers,
                           k=re3_k,
                           buffer_size=re3_buffer_size).to(device)

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

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
            self.set_action_std(self.action_std)

    def record_position(self, pos, episode=None, timestep=None, as_state=False):
        if self.trajectory_logger is not None:
            self.trajectory_logger.record_position(
                pos,
                episode=episode,
                timestep=timestep,
                as_state=as_state,
                grid_shape=self.trajectory_grid_shape
            )

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

    def update(self):
        if len(self.buffer.rewards) == 0:
            return

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # RE3: compute intrinsic rewards (no model update -- encoder is fixed)
        intrinsic_rewards = torch.zeros(len(self.buffer.rewards)).to(device)

        if self.use_re3:
            intrinsic_rewards = self.re3.compute_intrinsic_reward(old_states, update_buffer=True)

        # Combine extrinsic and intrinsic rewards
        extrinsic_rewards = []
        intrinsic_rewards_list = []
        combined_rewards = []

        for idx, reward in enumerate(self.buffer.rewards):
            ext_reward = reward
            intr_reward = 0.0

            if self.use_re3:
                try:
                    intr_reward = float(intrinsic_rewards[idx].cpu().item())
                except Exception:
                    intr_reward = float(intrinsic_rewards.cpu().item())

            combined_reward = ext_reward + self.intr_reward_strength * intr_reward

            extrinsic_rewards.append(ext_reward)
            intrinsic_rewards_list.append(intr_reward)
            combined_rewards.append(combined_reward)

        combined_rewards = torch.tensor(combined_rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(device)

        # GAE advantages
        advantages = torch.zeros_like(combined_rewards).to(device)
        last_gae = 0

        for t in reversed(range(len(combined_rewards))):
            if t == len(combined_rewards) - 1:
                next_value = 0
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
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        if len(extrinsic_rewards) > 0:
            avg_ext_reward = np.mean(extrinsic_rewards)
            avg_int_reward = np.mean(intrinsic_rewards_list) if len(intrinsic_rewards_list) > 0 else 0.0
            buf_size = len(self.re3.embedding_buffer) if self.use_re3 else 0
            print(f"Update - Avg Ext: {avg_ext_reward:.4f}, Avg Int: {avg_int_reward:.4f}, RE3 buffer: {buf_size}")

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        save_dict = {
            'policy_state_dict': self.policy_old.state_dict(),
        }
        if self.use_re3:
            save_dict['re3_encoder_state_dict'] = self.re3.encoder.state_dict()

        if self.trajectory_logger is not None:
            save_dict['trajectory_np'] = self.trajectory_logger.get_trajectory_array()

        torch.save(save_dict, checkpoint_path)

        if self.trajectory_logger is not None:
            traj_path = checkpoint_path + '.trajectory.npz'
            self.trajectory_logger.save_npz(traj_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if self.use_re3 and 're3_encoder_state_dict' in checkpoint:
            try:
                self.re3.encoder.load_state_dict(checkpoint['re3_encoder_state_dict'])
            except Exception:
                pass

        if self.trajectory_logger is not None:
            if 'trajectory_np' in checkpoint:
                try:
                    traj_np = checkpoint['trajectory_np']
                    if isinstance(traj_np, np.ndarray) and traj_np.size > 0:
                        self.trajectory_logger.trajectory = [
                            tuple(map(int, row.tolist())) for row in traj_np
                        ]
                except Exception:
                    pass

            traj_path = checkpoint_path + '.trajectory.npz'
            if os.path.exists(traj_path):
                try:
                    self.trajectory_logger.load_npz(traj_path)
                except Exception:
                    pass

    # ---------------- Trajectory utilities (wrappers) ----------------
    def save_trajectory_csv(self, csv_path):
        if self.trajectory_logger is not None:
            self.trajectory_logger.save_csv(csv_path)

    def save_trajectory_heatmap(self, out_path, **kwargs):
        if self.trajectory_logger is not None:
            if 'grid_shape' not in kwargs and self.trajectory_grid_shape is not None:
                kwargs['grid_shape'] = self.trajectory_grid_shape
            self.trajectory_logger.save_heatmap_png(out_path, **kwargs)

    def get_trajectory_statistics(self, **kwargs):
        if self.trajectory_logger is not None:
            return self.trajectory_logger.get_statistics(**kwargs)
        return None

    def print_trajectory_statistics(self, **kwargs):
        if self.trajectory_logger is not None:
            self.trajectory_logger.print_statistics(**kwargs)
