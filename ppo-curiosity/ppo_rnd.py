import os
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from curiosity.rnd import RND
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
                 use_rnd=True, rnd_lr=0.001, rnd_epochs=4, rnd_batch_size=64, intr_reward_strength=0.02,
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

        # RND setup
        self.use_rnd = use_rnd
        self.intr_reward_strength = intr_reward_strength

        if self.use_rnd:
            self.rnd = RND(state_dim, gamma_int=gamma).to(device)
            self.optimizer_rnd = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
            self.rnd_epochs = rnd_epochs
            self.rnd_batch_size = rnd_batch_size

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

    def rnd_update(self, next_states):
        """Update RND predictor on next_states"""
        total_predictor_loss = 0
        num_updates = 0

        dataset_size = next_states.shape[0]

        for _ in range(self.rnd_epochs):
            indices = np.random.permutation(dataset_size)

            for start_idx in range(0, dataset_size, self.rnd_batch_size):
                num_updates += 1
                batch_indices = indices[start_idx:start_idx + self.rnd_batch_size]
                batch_next_states = next_states[batch_indices]

                _, predictor_loss = self.rnd(batch_next_states)

                self.optimizer_rnd.zero_grad()
                predictor_loss.backward()
                self.optimizer_rnd.step()

                total_predictor_loss += predictor_loss.item()

        avg_predictor_loss = total_predictor_loss / num_updates if num_updates > 0 else 0
        return avg_predictor_loss

    def update(self):
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # RND: compute intrinsic rewards
        intrinsic_rewards = torch.zeros(len(self.buffer.rewards)).to(device)

        if self.use_rnd and len(self.buffer.next_states) > 0:
            old_next_states = torch.squeeze(torch.stack(self.buffer.next_states, dim=0)).detach().to(device)

            # Paper §2.4: update obs running mean/std before computing intrinsic.
            self.rnd.update_obs_rms(old_next_states)

            with torch.no_grad():
                intr_rewards, _ = self.rnd(old_next_states)

            # Paper §2.4: divide raw intrinsic by running std of intrinsic returns.
            intr_norm_np = self.rnd.normalize_intrinsic(intr_rewards.detach().cpu().numpy())
            intrinsic_rewards = torch.as_tensor(intr_norm_np, dtype=torch.float32, device=device)

            avg_predictor_loss = self.rnd_update(old_next_states)

        # Combine extrinsic and intrinsic rewards
        extrinsic_rewards = []
        intrinsic_rewards_list = []
        combined_rewards = []

        for idx, reward in enumerate(self.buffer.rewards):
            ext_reward = reward
            intr_reward = 0.0

            if self.use_rnd and len(self.buffer.next_states) > 0:
                try:
                    intr_reward = float(intrinsic_rewards[idx].cpu().item())
                except:
                    intr_reward = float(intrinsic_rewards.cpu().item())

            combined_reward = ext_reward + self.intr_reward_strength * intr_reward

            extrinsic_rewards.append(ext_reward)
            intrinsic_rewards_list.append(intr_reward)
            combined_rewards.append(combined_reward)

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

            delta = combined_rewards[t] + self.gamma * next_value * (1 - is_terminals[t]) - old_state_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - is_terminals[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Optimize policy for K epochs
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
            print(f"Update - Avg Extrinsic Reward: {avg_ext_reward:.4f}, Avg Intrinsic Reward: {avg_int_reward:.4f}")

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        save_dict = {
            'policy_state_dict': self.policy_old.state_dict(),
        }
        if self.use_rnd:
            save_dict['rnd_state_dict'] = self.rnd.state_dict()

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
        if self.use_rnd and 'rnd_state_dict' in checkpoint:
            self.rnd.load_state_dict(checkpoint['rnd_state_dict'])

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
        else:
            print("Warning: Trajectory logging is disabled")

    def save_trajectory_heatmap(self, out_path, **kwargs):
        if self.trajectory_logger is not None:
            if 'grid_shape' not in kwargs and self.trajectory_grid_shape is not None:
                kwargs['grid_shape'] = self.trajectory_grid_shape
            self.trajectory_logger.save_heatmap_png(out_path, **kwargs)
        else:
            print("Warning: Trajectory logging is disabled")

    def get_trajectory_statistics(self, **kwargs):
        if self.trajectory_logger is not None:
            return self.trajectory_logger.get_statistics(**kwargs)
        else:
            return None

    def print_trajectory_statistics(self, **kwargs):
        if self.trajectory_logger is not None:
            self.trajectory_logger.print_statistics(**kwargs)
        else:
            print("Warning: Trajectory logging is disabled")
