import torch
import torch.nn as nn
import numpy as np


class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, encoding_size=256, num_layers=2):
        super(ICM, self).__init__()
        self.action_dim = action_dim
        
        # Encoder network
        encoder_layers = []
        encoder_layers.append(nn.Linear(state_dim, encoding_size))
        nn.init.normal_(encoder_layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / state_dim))
        encoder_layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            encoder_layers.append(nn.Linear(encoding_size, encoding_size))
            nn.init.normal_(encoder_layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / encoding_size))
            encoder_layers.append(nn.Tanh())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Inverse model: predicts action from state encodings
        self.inverse_model = nn.Sequential(
            nn.Linear(encoding_size * 2, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        
        # Forward model: predicts next state encoding from current encoding and action
        self.forward_model = nn.Sequential(
            nn.Linear(encoding_size + action_dim, 256),
            nn.Tanh(),
            nn.Linear(256, encoding_size)
        )
    
    def encode(self, state):
        """Encode state to feature space"""
        return self.encoder(state)
    
    def forward(self, state, next_state, action):
        """
        Args:
            state: current state [batch_size, state_dim]
            next_state: next state [batch_size, state_dim]
            action: action taken [batch_size] (discrete)
        
        Returns:
            intrinsic_reward: intrinsic reward [batch_size]
            inverse_loss: inverse model loss (scalar)
            forward_loss: forward model loss (scalar)
        """
        # Encode states
        encoded_state = self.encode(state)
        encoded_next_state = self.encode(next_state)
        
        # Inverse model: predict action
        inverse_input = torch.cat([encoded_state, encoded_next_state], dim=-1)
        predicted_action_logits = self.inverse_model(inverse_input)
        
        # Inverse loss (cross-entropy)
        inverse_loss = nn.CrossEntropyLoss()(predicted_action_logits, action)
        
        # Forward model: predict next state encoding
        action_one_hot = nn.functional.one_hot(action, num_classes=self.action_dim).float()
        forward_input = torch.cat([encoded_state, action_one_hot], dim=-1)
        predicted_next_encoded_state = self.forward_model(forward_input)
        
        # Forward loss (MSE) - this is also the intrinsic reward
        forward_loss = 0.5 * ((predicted_next_encoded_state - encoded_next_state.detach()).pow(2)).mean(dim=-1)
        
        # Intrinsic reward is the prediction error
        intrinsic_reward = forward_loss.detach()
        
        # Average forward loss for optimization
        forward_loss = forward_loss.mean()
        
        return intrinsic_reward, inverse_loss, forward_loss