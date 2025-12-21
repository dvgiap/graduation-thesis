import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import math
import os
import numpy as np
from collections import deque
import hashlib

################################## set device ##################################
print("============================================================================================")
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## rnd networks ##################################
class RNDTarget(nn.Module):
    """ random fixed target network (frozen) """
    def __init__(self, state_dim, embed_dim=32):
        super(RNDTarget, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU()
        )
        # freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, s):
        return self.net(s)


class RNDPredictor(nn.Module):
    """ predictor trying to match target """
    def __init__(self, state_dim, embed_dim=32):
        super(RNDPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU()
        )

    def forward(self, s):
        return self.net(s)


################################## ppo policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


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

            # for single action environments.
            if self.action_dim == 1:
                try:
                    action = action.reshape(-1, self.action_dim)
                except Exception:
                    pass
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, use_rnd=True, lambda_rnd=0.1, lr_rnd=0.0001,
                 rnd_train_epochs=5):
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # rnd parameters
        self.use_rnd = use_rnd
        self.lambda_rnd = lambda_rnd
        self.rnd_train_epochs = rnd_train_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # initialize rnd networks
        if self.use_rnd:
            self.rnd_target = RNDTarget(state_dim).to(device)
            self.rnd_predictor = RNDPredictor(state_dim).to(device)
            self.rnd_optimizer = torch.optim.Adam(self.rnd_predictor.parameters(), lr=lr_rnd)

            # for tracking intrinsic rewards statistics (optional)
            self.intrinsic_reward_mean = 0.0
            self.intrinsic_reward_std = 1.0
            self.intrinsic_reward_count = 0

        # metrics from https://arxiv.org/pdf/2501.11533
        # total frames (interactions) seen so far
        self.total_frames = 0

        # unique observations (hash set)
        self.unique_obs = set()

        # visited grid positions (set of (x,y))
        self.visited_positions = set()
        # if user provides n_total_positions via env_info in record_reward_and_info, we'll store it
        self.n_total_positions = None

        # policy entropy running stats (sum and count)
        self.entropy_sum = 0.0
        self.entropy_count = 0

        # episodic returns: keep a deque of last N episodes
        self.last_episode_returns = deque(maxlen=100)
        self.current_episode_return = 0.0

        # reward discovery frames: list of discovered frame indices (we capture first up to 3 discoveries)
        self.reward_discovery_frames = []

        # store last computed metrics snapshot 
        self.metrics_snapshot = {}

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

    def _get_dist(self, policy, states):
        """
        helper: build distribution object for a given policy and states (batched)
        states: (N, state_dim)
        """
        if self.has_continuous_action_space:
            action_mean = policy.actor(states)
            action_var = policy.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            return MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = policy.actor(states)
            return Categorical(action_probs)

    def compute_intrinsic_reward(self, state):
        """compute r_int = ||target(s) - predictor(s)||^2"""
        if not self.use_rnd:
            return 0.0

        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.FloatTensor(state).to(device)
            else:
                state_tensor = state.to(device)
            target_feat = self.rnd_target(state_tensor)
            pred_feat = self.rnd_predictor(state_tensor)
            intrinsic_reward = torch.mean((target_feat - pred_feat) ** 2).item()

        return intrinsic_reward

    def select_action(self, state):
        """
        Select action with policy_old, append to buffer, and update metrics:
          - unique observation hashing
          - policy entropy running average
        NOTE: total_frames is incremented in record_reward_and_info() when env.step completes.
        """
        # convert to torch tensor (1D)
        if not isinstance(state, torch.Tensor):
            state_t = torch.FloatTensor(state).to(device)
        else:
            state_t = state.to(device)

        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state_t)

        # record into buffer (same fields as before)
        self.buffer.states.append(state_t)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        # observation hashing: quantize floats to reduce noise and compute bytes hash
        try:
            s_np = state_t.cpu().numpy()
            s_q = np.round(s_np, 3)
            obs_hash = hashlib.sha1(s_q.tobytes()).hexdigest()
        except Exception:
            try:
                b = torch.tensor(state_t).cpu().numpy().tobytes()
                obs_hash = hashlib.sha1(b).hexdigest()
            except Exception:
                obs_hash = None

        if obs_hash is not None:
            self.unique_obs.add(obs_hash)

        # policy entropy: compute current policy entropy at this state (using policy_old)
        try:
            dist = self._get_dist(self.policy_old, state_t.unsqueeze(0))  # batched (1, ...)
            ent = dist.entropy()  # torch.Tensor, maybe shape (1,) or scalar
            if isinstance(ent, torch.Tensor):
                entropy = float(ent.mean().cpu().item())
            else:
                entropy = float(ent)
        except Exception:
            entropy = 0.0
        self.entropy_sum += entropy
        self.entropy_count += 1

        # Return action in the same type as before
        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

    def record_reward_and_info(self, reward, is_terminal, env_info=None):
        """
        Call this after env.step to record reward/is_terminal and optional env_info dict.
        env_info can contain:
            - 'agent_pos': (x,y) tuple or list
            - 'n_total_positions': int (total possible grid positions)
            - any other info
        This method appends reward and is_terminal to buffer (so it's equivalent to
        self.buffer.rewards.append(...) and self.buffer.is_terminals.append(...)),
        AND updates metrics like position coverage and reward discovery frames and episodic returns.

        """
        # increment total frames
        self.total_frames += 1

        # append to buffer as before
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(is_terminal)

        # episodic return tracking
        try:
            self.current_episode_return += float(reward)
        except Exception:
            try:
                self.current_episode_return += float(torch.tensor(reward).item())
            except Exception:
                pass

        if is_terminal:
            self.last_episode_returns.append(self.current_episode_return)
            self.current_episode_return = 0.0

        # record position if provided
        if env_info is not None:
            if 'agent_pos' in env_info and env_info['agent_pos'] is not None:
                try:
                    pos = tuple(env_info['agent_pos'])
                    self.visited_positions.add(pos)
                except Exception:
                    pass
            # record total positions if provided (used for normalizing)
            if 'n_total_positions' in env_info:
                try:
                    self.n_total_positions = int(env_info['n_total_positions'])
                except Exception:
                    pass

        # check reward discovery times (we record frame index when reward>0)
        if reward is not None:
            try:
                rval = float(reward)
            except Exception:
                try:
                    rval = float(torch.tensor(reward).item())
                except Exception:
                    rval = 0.0
            if rval > 0:
                # ensure we only add up to 3 discovery times and avoid duplicates
                if len(self.reward_discovery_frames) < 3:
                    # avoid adding same frame twice
                    if len(self.reward_discovery_frames) == 0 or self.reward_discovery_frames[-1] != int(self.total_frames):
                        self.reward_discovery_frames.append(int(self.total_frames))

    def update(self):
        # compute intrinsic rewards and proxy rewards
        intrinsic_rewards = []
        proxy_rewards = []

        if self.use_rnd:
            for i, state in enumerate(self.buffer.states):
                with torch.no_grad():
                    target_feat = self.rnd_target(state)
                    pred_feat = self.rnd_predictor(state)
                    r_int = torch.mean((target_feat - pred_feat) ** 2).item()
                    intrinsic_rewards.append(r_int)

                    # r_proxy = r_ext + lambda * r_int
                    r_ext = self.buffer.rewards[i]
                    r_proxy = r_ext + self.lambda_rnd * r_int
                    proxy_rewards.append(r_proxy)

            # update rnd predictor
            self._update_rnd()

            # replace rewards with proxy rewards for ppo training
            original_rewards = self.buffer.rewards.copy()
            self.buffer.rewards = proxy_rewards
        else:
            original_rewards = self.buffer.rewards.copy()

        # monte carlo estimate of returns using proxy rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        if len(rewards) == 0:
            return

        # normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # optimize policy for k epochs
        for _ in range(self.K_epochs):
            # evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # finding surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective ppo
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # print statistics
        if self.use_rnd:
            avg_r_ext = sum(original_rewards) / len(original_rewards)
            avg_r_proxy = sum(proxy_rewards) / len(proxy_rewards)
            avg_r_int = sum(intrinsic_rewards) / len(intrinsic_rewards)
            print(f"Avg r_ext: {avg_r_ext:.4f} | Avg r_int: {avg_r_int:.4f} | Avg r_proxy: {avg_r_proxy:.4f}")
        else:
            avg_r_ext = sum(original_rewards) / len(original_rewards)
            print(f"Avg r_ext: {avg_r_ext:.4f}")

        # snapshot metrics before clearing buffer (useful for logging)
        self.metrics_snapshot = self.compute_metrics_snapshot()

        # clear buffer
        self.buffer.clear()

    def _update_rnd(self):
        """update rnd predictor network"""
        if not self.use_rnd:
            return

        # stack all states
        if len(self.buffer.states) == 0:
            return

        states = torch.stack(self.buffer.states).to(device)

        # train rnd predictor
        for _ in range(self.rnd_train_epochs):
            target_feat = self.rnd_target(states)
            pred_feat = self.rnd_predictor(states)

            # mse loss between target and predictor
            rnd_loss = torch.mean((target_feat - pred_feat) ** 2)

            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            self.rnd_optimizer.step()

    def save(self, checkpoint_path):
        save_dict = {
            'policy_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        if self.use_rnd:
            save_dict['rnd_predictor_state_dict'] = self.rnd_predictor.state_dict()
            save_dict['rnd_optimizer_state_dict'] = self.rnd_optimizer.state_dict()

        torch.save(save_dict, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        if 'policy_state_dict' in checkpoint:
            self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        # try to load full policy too if present
        try:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
        except Exception:
            pass

        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass

        if self.use_rnd and 'rnd_predictor_state_dict' in checkpoint:
            self.rnd_predictor.load_state_dict(checkpoint['rnd_predictor_state_dict'])
            if 'rnd_optimizer_state_dict' in checkpoint:
                try:
                    self.rnd_optimizer.load_state_dict(checkpoint['rnd_optimizer_state_dict'])
                except Exception:
                    pass


    def compute_metrics_snapshot(self, normalize_obs_by=None):
        """
        Return a snapshot dictionary of current metrics (raw values and some normalized)
        Keys:
          - avg_recent_return: average episodic return over last_episode_returns (or None)
          - observation_coverage_count: number of unique observations visited
          - position_coverage_fraction: fraction visited positions if n_total_positions known else None
          - avg_policy_entropy: average entropy over recorded frames
          - reward_discovery_frames: list of discovery frame indices (first up to 3)
          - total_frames: total_frames

        normalize_obs_by: optional int (e.g. max coverage across methods) to compute normalized observation coverage.
        """
        if len(self.last_episode_returns) > 0:
            avg_recent_return = float(np.mean(self.last_episode_returns))
        else:
            avg_recent_return = None

        obs_cov = len(self.unique_obs)
        if normalize_obs_by is not None and normalize_obs_by > 0:
            obs_cov_norm = float(obs_cov) / float(normalize_obs_by)
        else:
            obs_cov_norm = None

        if self.n_total_positions is not None and self.n_total_positions > 0:
            pos_cov = len(self.visited_positions) / float(self.n_total_positions)
        else:
            pos_cov = None

        if self.entropy_count > 0:
            avg_entropy = self.entropy_sum / float(self.entropy_count)
        else:
            avg_entropy = None

        snapshot = {
            'avg_recent_return': avg_recent_return,
            'observation_coverage_count': obs_cov,
            'observation_coverage_normalized': obs_cov_norm,
            'position_coverage_fraction': pos_cov,
            'avg_policy_entropy': avg_entropy,
            'reward_discovery_frames': list(self.reward_discovery_frames),
            'total_frames': int(self.total_frames)
        }
        return snapshot

    def print_metrics(self):
        snap = self.compute_metrics_snapshot()
        print("===== Exploration metrics snapshot =====")
        print(f"total frames: {snap['total_frames']}")
        if snap['avg_recent_return'] is not None:
            print(f"avg episodic return (last up to 100): {snap['avg_recent_return']:.4f}")
        print(f"unique observations visited: {snap['observation_coverage_count']}")
        if snap['position_coverage_fraction'] is not None:
            print(f"position coverage fraction: {snap['position_coverage_fraction']:.4f}")
        if snap['avg_policy_entropy'] is not None:
            print(f"average policy entropy: {snap['avg_policy_entropy']:.6f}")
        if len(snap['reward_discovery_frames']) > 0:
            print(f"reward discovery frames (first up to 3): {snap['reward_discovery_frames']}")
        print("========================================")

    def compute_metrics(self):
        """
        Public convenience function returning same snapshot.
        """
        return self.compute_metrics_snapshot()
