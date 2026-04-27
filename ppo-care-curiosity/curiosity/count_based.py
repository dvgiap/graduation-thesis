import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class CountBasedExploration:
    def __init__(self, state_dim, hash_dim=32, bonus_type='inverse_sqrt'):
        self.state_dim = state_dim
        self.hash_dim = hash_dim
        self.bonus_type = bonus_type
        self.projection_matrix = np.random.randn(state_dim, hash_dim)
        self.count_table = defaultdict(int)
        
        # Statistics
        self.total_visits = 0
        
    def hash_state(self, state):
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        state = state.flatten()
        
        # Random projection
        projected = np.dot(state, self.projection_matrix)
        binned = np.round(projected / 0.1).astype(int)
        hash_key = ','.join(map(str, binned))
        
        return hash_key
    
    def get_count(self, state):
        hash_key = self.hash_state(state)
        return self.count_table[hash_key]
    
    def increment_count(self, state):
        hash_key = self.hash_state(state)
        self.count_table[hash_key] += 1
        self.total_visits += 1
    
    def compute_intrinsic_reward(self, state):
        count = self.get_count(state)
        if self.bonus_type == 'inverse':
            # r = 1 / count
            bonus = 1.0 / max(count, 1)
        elif self.bonus_type == 'inverse_sqrt':
            # r = 1 / sqrt(count)
            bonus = 1.0 / np.sqrt(max(count, 1))
        elif self.bonus_type == 'exponential':
            # r = exp(-count)
            bonus = np.exp(-count * 0.1)
        else:
            bonus = 1.0 / max(count, 1)
        
        return bonus
    
    def update(self, states):
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        
        intrinsic_rewards = []
        
        for i in range(states.shape[0]):
            state = states[i]
            
            reward = self.compute_intrinsic_reward(state)
            intrinsic_rewards.append(reward)
            
            # Increment count
            self.increment_count(state)
        
        return np.array(intrinsic_rewards)
    
    def get_statistics(self):
        unique_states = len(self.count_table)
        avg_count = self.total_visits / max(unique_states, 1)
        
        counts = list(self.count_table.values())
        max_count = max(counts) if counts else 0
        min_count = min(counts) if counts else 0
        
        return {
            'unique_states': unique_states,
            'total_visits': self.total_visits,
            'avg_count': avg_count,
            'max_count': max_count,
            'min_count': min_count
        }
    
    def reset(self):
        """Reset count table"""
        self.count_table.clear()
        self.total_visits = 0


class CountBasedNN(nn.Module):
    def __init__(self, state_dim, encoding_size=64, num_layers=2):
        super(CountBasedNN, self).__init__()
        
        # Encoder network
        encoder_layers = []
        encoder_layers.append(nn.Linear(state_dim, encoding_size))
        encoder_layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            encoder_layers.append(nn.Linear(encoding_size, encoding_size))
            encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Density predictor (predicts log probability)
        self.density_head = nn.Sequential(
            nn.Linear(encoding_size, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, 1)
        )
        
    def forward(self, state):
        encoded = self.encoder(state)
        log_density = self.density_head(encoded)
        return log_density
    
    def compute_intrinsic_reward(self, state):
        with torch.no_grad():
            log_density = self.forward(state)
            intrinsic_reward = -log_density.squeeze()
        return intrinsic_reward

def create_count_based_module(state_dim, method='hash', **kwargs):
    if method == 'hash':
        return CountBasedExploration(
            state_dim, 
            hash_dim=kwargs.get('hash_dim', 32),
            bonus_type=kwargs.get('bonus_type', 'inverse_sqrt')
        )
    elif method == 'nn':
        return CountBasedNN(
            state_dim,
            encoding_size=kwargs.get('encoding_size', 64),
            num_layers=kwargs.get('num_layers', 2)
        )
    else:
        raise ValueError(f"Unknown method: {method}")