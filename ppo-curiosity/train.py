import os
import time
import random
from datetime import datetime
import argparse

import torch
import numpy as np

import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper


def train(
    exploration_method='none',
    random_seed=1,
    env_name="MiniGrid-DoorKey-8x8-v0",
    max_training_timesteps=int(1e6),
    seeds_range=(1, 5)
):
    """
    Unified training function for PPO with different exploration methods.
    
    Args:
        exploration_method: 'none', 'icm', 'count', or 'rnd'
        random_seed: Starting random seed
        env_name: Gymnasium environment name
        max_training_timesteps: Maximum timesteps per seed
        seeds_range: Tuple of (start_seed, end_seed) inclusive
    """

    print("============================================================================================")

    # Import the appropriate PPO implementation
    if exploration_method == 'none':
        from ppo import PPO
    elif exploration_method == 'icm':
        from ppo_icm import PPO
    elif exploration_method == 'count':
        from ppo_count import PPO
    elif exploration_method == 're3':
        from ppo_re3 import PPO
    else:
        raise ValueError(f"Unknown exploration method: {exploration_method}")

    ####### Environment hyperparameters ######
    has_continuous_action_space = False
    max_ep_len = 1000
    print_freq = max_ep_len * 10
    log_freq = max_ep_len * 2
    save_model_freq = int(1e5)

    ####### PPO hyperparameters ######
    update_timestep = max_ep_len * 4
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    gae_lambda = 0.95
    lr_actor = 0.0003
    lr_critic = 0.001

    ####### Exploration-specific hyperparameters ######
    # ICM
    icm_lr = 0.001
    icm_epochs = 4
    icm_batch_size = 64
    icm_intr_strength = 0.001
    
    # Count-based
    hash_dim = 32
    bonus_type = 'inverse_sqrt'
    count_intr_strength = 0.001
    
    # RE3 (random encoder is frozen; no learning rate / epochs needed)
    re3_encoding_size = 64
    re3_num_layers = 2
    re3_k = 3
    re3_buffer_size = 10000
    re3_intr_strength = 0.001

    # Create environment
    env = gym.make(env_name)
    env = FlatObsWrapper(env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if not has_continuous_action_space else env.action_space.shape[0]

    # Get grid shape for trajectory logging (if available)
    try:
        grid_width = getattr(env.unwrapped, 'width', None)
        grid_height = getattr(env.unwrapped, 'height', None)
        trajectory_grid_shape = (grid_width, grid_height) if (grid_width and grid_height) else None
    except Exception:
        trajectory_grid_shape = None

    # Setup logging and checkpointing
    suffix_map = {
        'none': '',
        'icm': '_ICM',
        'count': '_COUNT',
        're3': '_RE3'
    }
    suffix = suffix_map[exploration_method]
    
    log_dir = os.path.join("logs", env_name)
    os.makedirs(log_dir, exist_ok=True)
    
    model_dir = os.path.join("models", env_name)
    os.makedirs(model_dir, exist_ok=True)

    # Print hyperparameters
    print(f"Training environment: {env_name}")
    print(f"Exploration method: {exploration_method.upper()}")
    print("--------------------------------------------------------------------------------------------")
    print(f"Max training timesteps: {max_training_timesteps}")
    print(f"Max timesteps per episode: {max_ep_len}")
    print(f"Model saving frequency: {save_model_freq} timesteps")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    if trajectory_grid_shape:
        print(f"Grid shape for trajectory: {trajectory_grid_shape}")
    print("--------------------------------------------------------------------------------------------")
    print(f"PPO update frequency: {update_timestep} timesteps")
    print(f"K epochs: {K_epochs}, eps_clip: {eps_clip}, gamma: {gamma}, GAE lambda: {gae_lambda}")
    print(f"LR actor: {lr_actor}, LR critic: {lr_critic}")
    
    if exploration_method == 'icm':
        print("--------------------------------------------------------------------------------------------")
        print(f"ICM - lr: {icm_lr}, epochs: {icm_epochs}, batch: {icm_batch_size}, strength: {icm_intr_strength}")
    elif exploration_method == 'count':
        print("--------------------------------------------------------------------------------------------")
        print(f"Count-Based - hash_dim: {hash_dim}, bonus_type: {bonus_type}, strength: {count_intr_strength}")
    elif exploration_method == 're3':
        print("--------------------------------------------------------------------------------------------")
        print(f"RE3 - encoding: {re3_encoding_size}, layers: {re3_num_layers}, k: {re3_k}, buffer: {re3_buffer_size}, strength: {re3_intr_strength}")
    
    print("============================================================================================")

    # Training loop over seeds
    start_time_all = datetime.now().replace(microsecond=0)
    
    for seed in range(seeds_range[0], seeds_range[1] + 1):
        print(f"\n\n########## STARTING RUN FOR SEED {seed} ##########\n")
        
        # Setup paths for this seed
        log_f_name = os.path.join(log_dir, f'PPO{suffix}_{env_name}_seed_{seed}.csv')
        checkpoint_path = os.path.join(model_dir, f'PPO{suffix}_{env_name}_seed_{seed}.pth')
        
        print(f"Logging at: {log_f_name}")
        print(f"Checkpoint path: {checkpoint_path}")
        
        # Set random seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        except Exception:
            pass
        
        # Initialize PPO agent based on exploration method
        # All agents now have trajectory logging enabled
        if exploration_method == 'none':
            ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, 
                          K_epochs, eps_clip, has_continuous_action_space,
                          enable_trajectory_logging=True,
                          trajectory_grid_shape=trajectory_grid_shape)
        elif exploration_method == 'icm':
            ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma,
                          K_epochs, eps_clip, has_continuous_action_space,
                          use_icm=True, icm_lr=icm_lr, icm_epochs=icm_epochs,
                          icm_batch_size=icm_batch_size, intr_reward_strength=icm_intr_strength,
                          gae_lambda=gae_lambda,
                          enable_trajectory_logging=True,
                          trajectory_grid_shape=trajectory_grid_shape)
        elif exploration_method == 'count':
            ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma,
                          K_epochs, eps_clip, has_continuous_action_space,
                          use_count_based=True, hash_dim=hash_dim, bonus_type=bonus_type,
                          intr_reward_strength=count_intr_strength, gae_lambda=gae_lambda,
                          enable_trajectory_logging=True,
                          trajectory_grid_shape=trajectory_grid_shape)
        elif exploration_method == 're3':
            ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma,
                          K_epochs, eps_clip, has_continuous_action_space,
                          use_re3=True, re3_encoding_size=re3_encoding_size,
                          re3_num_layers=re3_num_layers, re3_k=re3_k,
                          re3_buffer_size=re3_buffer_size,
                          intr_reward_strength=re3_intr_strength,
                          gae_lambda=gae_lambda,
                          enable_trajectory_logging=True,
                          trajectory_grid_shape=trajectory_grid_shape)
        
        # Training variables
        start_time = datetime.now().replace(microsecond=0)
        print(f"Started training at: {start_time}")
        
        log_f = open(log_f_name, "w+")
        log_f.write('episode,timestep,reward\n')
        
        print_running_reward = 0
        print_running_episodes = 0
        log_running_reward = 0
        log_running_episodes = 0
        time_step = 0
        i_episode = 0
        
        # Episode loop
        while time_step <= max_training_timesteps:
            state, _ = env.reset()
            current_ep_reward = 0
            
            # Record initial agent position
            try:
                agent_pos = env.unwrapped.agent_pos
                ppo_agent.record_position(agent_pos, episode=i_episode, timestep=time_step)
            except Exception:
                pass
            
            for t in range(1, max_ep_len + 1):
                # Select and execute action
                action = ppo_agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store transition
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)
                
                # Store next_state for exploration methods that need it
                if exploration_method in ['icm', 'count', 'rnd']:
                    device = next(ppo_agent.policy.actor.parameters()).device
                    ppo_agent.buffer.next_states.append(torch.FloatTensor(next_state).to(device))
                
                time_step += 1
                current_ep_reward += reward
                
                # Record agent position after step
                try:
                    agent_pos = env.unwrapped.agent_pos
                    ppo_agent.record_position(agent_pos, episode=i_episode, timestep=time_step)
                except Exception:
                    pass
                
                state = next_state
                
                # Update policy
                if time_step % update_timestep == 0:
                    ppo_agent.update()
                
                # Logging
                if time_step % log_freq == 0:
                    log_avg_reward = log_running_reward / log_running_episodes if log_running_episodes > 0 else 0
                    log_f.write(f'{i_episode},{time_step},{round(log_avg_reward, 4)}\n')
                    log_f.flush()
                    log_running_reward = 0
                    log_running_episodes = 0
                
                # Printing
                if time_step % print_freq == 0:
                    print_avg_reward = print_running_reward / print_running_episodes if print_running_episodes > 0 else 0
                    print(f"Episode: {i_episode} \t Timestep: {time_step} \t Avg Reward: {round(print_avg_reward, 2)}")
                    print_running_reward = 0
                    print_running_episodes = 0
                
                # Save model + trajectory + heatmap
                if time_step % save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print(f"Saving model at: {checkpoint_path}")
                    try:
                        # Save checkpoint and trajectory
                        ppo_agent.save(checkpoint_path)
                        
                        # Export trajectory CSV
                        try:
                            csv_out = checkpoint_path + ".trajectory.csv"
                            ppo_agent.save_trajectory_csv(csv_out)
                        except Exception as e:
                            print(f"Warning: failed to save trajectory CSV: {e}")
                        
                        # Export heatmap PNG
                        try:
                            heat_out = checkpoint_path + ".trajectory.heatmap.png"
                            ppo_agent.save_trajectory_heatmap(
                                heat_out,
                                grid_shape=trajectory_grid_shape,
                                normalize=False,
                                annotate_max=True
                            )
                        except Exception as e:
                            print(f"Warning: failed to save heatmap: {e}")
                        
                        print("Model & trajectory saved (checkpoint + .npz + csv + heatmap)")
                    except Exception as e:
                        print(f"Error saving model/trajectory: {e}")
                    
                    print(f"Elapsed time: {datetime.now().replace(microsecond=0) - start_time}")
                    print("--------------------------------------------------------------------------------------------")
                
                if done:
                    break
            
            print_running_reward += current_ep_reward
            print_running_episodes += 1
            log_running_reward += current_ep_reward
            log_running_episodes += 1
            i_episode += 1
        
        # Final save when training finishes
        try:
            print("Final saving model & trajectory...")
            ppo_agent.save(checkpoint_path)
            
            # Final CSV export
            csv_out = checkpoint_path + ".trajectory.csv"
            ppo_agent.save_trajectory_csv(csv_out)
            
            # Final heatmap export
            heat_out = checkpoint_path + ".trajectory.heatmap.final.png"
            ppo_agent.save_trajectory_heatmap(
                heat_out,
                grid_shape=trajectory_grid_shape,
                normalize=False,
                annotate_max=True
            )
            print("Final save done.")
        except Exception as e:
            print(f"Error on final save: {e}")
        
        # Seed finished
        log_f.close()
        end_time = datetime.now().replace(microsecond=0)
        print("============================================================================================")
        print(f"Seed {seed} - Started: {start_time}, Finished: {end_time}")
        print(f"Training time: {end_time - start_time}")
        print("============================================================================================")
        time.sleep(2)  # Brief pause between seeds
    
    env.close()
    print(f"\nALL SEEDS COMPLETED. Total time: {datetime.now().replace(microsecond=0) - start_time_all}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified PPO training with exploration methods")
    parser.add_argument("--method", type=str, default="none", 
                       choices=['none', 'icm', 'count', 're3'],
                       help="Exploration method to use")
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-8x8-v0",
                       help="Gymnasium environment name")
    parser.add_argument("--seed_start", type=int, default=1,
                       help="Starting seed (inclusive)")
    parser.add_argument("--seed_end", type=int, default=5,
                       help="Ending seed (inclusive)")
    parser.add_argument("--max_steps", type=int, default=int(1e6),
                       help="Max training timesteps per seed")
    
    args = parser.parse_args()
    
    train(
        exploration_method=args.method,
        random_seed=args.seed_start,
        env_name=args.env,
        max_training_timesteps=args.max_steps,
        seeds_range=(args.seed_start, args.seed_end)
    )