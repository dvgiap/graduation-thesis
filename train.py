# train.py
import os
import glob
import time
import random
import json
from datetime import datetime

import torch
import numpy as np

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, FlatObsWrapper

from rnd import PPO

################################### helpers ###################################
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_json_serializable(obj):
    """
    Recursively convert obj into JSON-serializable Python types.
    Handles: numpy scalars, numpy arrays, dicts, lists, tuples, datetime.
    """
    # None, bool, str, int, float are fine
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    # datetime -> isoformat string
    if isinstance(obj, datetime):
        return obj.isoformat()
    # numpy scalar (np.int64, np.float64, np.bool_)
    if isinstance(obj, (np.generic,)):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    # numpy arrays -> list
    if isinstance(obj, np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            # if object dtype, convert each element
            return [make_json_serializable(x) for x in obj.flatten().tolist()]
    # dict -> recurse
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    # list or tuple -> recurse
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    # other numeric types (python)
    if isinstance(obj, (int, float)):
        return obj
    # fallback: try to convert to int/float, else str
    try:
        return int(obj)
    except Exception:
        try:
            return float(obj)
        except Exception:
            return str(obj)

def safe_list_to_json_serializable(lst):
    # convert items to basic python types using make_json_serializable
    return [make_json_serializable(x) for x in lst]

def save_metrics_npz(snapshot, filepath):
    """
    Save snapshot dict to .npz.
    We convert non-array objects into numpy arrays (object dtype) if needed.
    """
    np_save_dict = {}
    for k, v in snapshot.items():
        try:
            # If v is list/dict, attempt conversion to array; object dtype otherwise
            if isinstance(v, dict):
                np_save_dict[k] = np.array(make_json_serializable(v), dtype=object)
            else:
                np_save_dict[k] = np.array(v)
        except Exception:
            # fallback to object array with string representation
            np_save_dict[k] = np.array([str(v)], dtype=object)
    # use savez_compressed for smaller files
    np.savez_compressed(filepath, **np_save_dict)

################################### training ###################################
def train(random_seed=1):
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "MiniGrid-DoorKey-5x5-v0"
    # you can try:
    # MiniGrid-DoorKey-6x6-v0
    # MiniGrid-DoorKey-8x8-v0
    # MiniGrid-LavaCrossingS9N1-v0
    # MiniGrid-LavaCrossingS9N2-v0
    # MiniGrid-LavaCrossingS9N3-v0
    # MiniGrid-KeyCorridorS3R1-v0
    # MiniGrid-KeyCorridorS3R2-v0
    ########################################################
    # more: MiniGrid-FourRooms-v0, ...
    has_continuous_action_space = False  # minigrid uses discrete action space

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(1e6)   # break training loop if timesteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    # no need for action_std for discrete action space
    #####################################################

    ## note : print/log frequencies should be > than max_ep_len

    ################ ppo hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one ppo update

    eps_clip = 0.2          # clip parameter for ppo
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    #####################################################

    print(f"training environment name : {env_name} (seed={random_seed})")

    # create minigrid environment (Gymnasium)
    env = gym.make(env_name)

    # wrap environment to use flat observation instead of image
    env = FlatObsWrapper(env)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging / results directories ######################
    # results/run directory with timestamp so runs are separate
    base_results_dir = "results"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_results_dir, env_name, f"seed_{random_seed}_{run_timestamp}")
    ensure_dir(run_dir)

    # main CSV log (time-series) and model/checkpoint directories
    log_f_name = os.path.join(run_dir, f'PPO_{env_name}_seed_{random_seed}.csv')
    models_dir = os.path.join("models", env_name)
    ensure_dir(models_dir)
    checkpoint_path = os.path.join(models_dir, f"PPO_{env_name}_seed_{random_seed}.pth")

    print("logging at : " + log_f_name)
    print("save checkpoint path : " + checkpoint_path)
    print("results directory : " + run_dir)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("initializing a continuous action space policy")
    else:
        print("initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("ppo update frequency : " + str(update_timestep) + " timesteps")
    print("ppo k epochs : ", K_epochs)
    print("ppo epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    print("--------------------------------------------------------------------------------------------")

    # set random seeds
    if random_seed:
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        # seed environment (gymnasium)
        try:
            env.reset(seed=random_seed)
        except TypeError:
            # older gym may not accept seed in reset; fallback to seeding spaces
            pass
        try:
            env.action_space.seed(random_seed)
        except Exception:
            pass
    print("============================================================================================")

    ################# training procedure ################

    # initialize a ppo agent
    if has_continuous_action_space:
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std=None)
    else:
        # discrete action space - no action_std needed
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("started training at : ", start_time)

    print("============================================================================================")

    # logging file -- extend columns with exploration metrics
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,episode_reward,avg_recent_return,obs_coverage,pos_coverage,avg_entropy,reward_discovery_frames,total_frames\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # store time-series arrays in memory for faster final save (also saved periodically)
    timeseries = {
        'episode': [],
        'timestep': [],
        'episode_reward': [],
        'avg_recent_return': [],
        'obs_coverage': [],
        'pos_coverage': [],
        'avg_entropy': [],
        'reward_discovery_frames': [],
        'total_frames': []
    }

    # training loop
    while time_step <= max_training_timesteps:

        state, _ = env.reset()  # gymnasium returns (state, info)
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)

            # execute in environment
            state_next, reward, terminated, truncated, info = env.step(action)  # gymnasium returns 5 values
            done = terminated or truncated

            # prepare env_info to pass to PPO.metrics (agent position, n_total_positions)
            env_info = {}
            try:
                # try to get agent_pos and grid dims from unwrapped env (works for minigrid)
                unwrapped = env.unwrapped
                try:
                    agent_pos = getattr(unwrapped, 'agent_pos', None)
                except Exception:
                    agent_pos = None
                # width/height in many minigrid envs:
                try:
                    n_total_positions = int(unwrapped.width * unwrapped.height)
                except Exception:
                    n_total_positions = None
            except Exception:
                agent_pos = None
                n_total_positions = None

            env_info['agent_pos'] = agent_pos
            env_info['n_total_positions'] = n_total_positions

            # record reward and is_terminals; prefer using record_reward_and_info if available

            ppo_agent.record_reward_and_info(reward, done, env_info=env_info)

            time_step += 1
            current_ep_reward += reward

            # update ppo agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

                # optional: print current metrics snapshot after each update
                ppo_agent.print_metrics()

            # log in logging file + save snapshot to disk periodically
            if time_step % log_freq == 0:
                # log average reward till last episodes in the logging interval
                log_avg_reward = log_running_reward / log_running_episodes if log_running_episodes > 0 else 0
                log_avg_reward = round(log_avg_reward, 4)

                # get metrics snapshot (if available)
                metrics = ppo_agent.compute_metrics()
                # prepare serializable snapshot
                reward_discovery_str = '|'.join([str(x) for x in metrics.get('reward_discovery_frames', [])]) if metrics.get('reward_discovery_frames') else ''
                avg_recent = metrics['avg_recent_return'] if metrics['avg_recent_return'] is not None else ''
                pos_cov = metrics['position_coverage_fraction'] if metrics['position_coverage_fraction'] is not None else ''
                avg_ent = metrics['avg_policy_entropy'] if metrics['avg_policy_entropy'] is not None else ''

                # write CSV line
                log_line = '{},{},{},{},{},{},{},{},{}\n'.format(
                    i_episode,
                    time_step,
                    log_avg_reward,
                    avg_recent,
                    metrics['observation_coverage_count'],
                    pos_cov,
                    avg_ent,
                    reward_discovery_str,
                    metrics['total_frames']
                )
                log_f.write(log_line)
                log_f.flush()

                # append to in-memory time series
                timeseries['episode'].append(i_episode)
                timeseries['timestep'].append(time_step)
                timeseries['episode_reward'].append(log_avg_reward)
                timeseries['avg_recent_return'].append(avg_recent)
                timeseries['obs_coverage'].append(metrics['observation_coverage_count'])
                timeseries['pos_coverage'].append(pos_cov)
                timeseries['avg_entropy'].append(avg_ent)
                timeseries['reward_discovery_frames'].append(reward_discovery_str)
                timeseries['total_frames'].append(metrics['total_frames'])

                # Save snapshot .npz (for later visualization). This includes:
                # - metrics dict (as arrays)
                # - visited_positions (if present)
                snapshot = {
                    'timestamp': datetime.now(),
                    'episode': i_episode,
                    'timestep': time_step,
                    'metrics': metrics
                }

                # collect visited positions if available
                vispos = None
                try:
                    vispos = list(getattr(ppo_agent, 'visited_positions'))
                except Exception:
                    vispos = None
                snapshot['visited_positions'] = vispos

                # Save JSON metadata and NPZ numeric data
                # 1) JSON summary (human readable) - use make_json_serializable
                json_path = os.path.join(run_dir, f"metrics_summary_t{time_step}.json")
                try:
                    serializable_snapshot = {
                        'timestamp': make_json_serializable(snapshot['timestamp']),
                        'episode': make_json_serializable(snapshot['episode']),
                        'timestep': make_json_serializable(snapshot['timestep']),
                        'metrics': {
                            'avg_recent_return': make_json_serializable(metrics['avg_recent_return']),
                            'observation_coverage_count': make_json_serializable(metrics['observation_coverage_count']),
                            'position_coverage_fraction': make_json_serializable(metrics['position_coverage_fraction']),
                            'avg_policy_entropy': make_json_serializable(metrics['avg_policy_entropy']),
                            'reward_discovery_frames': make_json_serializable(metrics['reward_discovery_frames']),
                            'total_frames': make_json_serializable(metrics['total_frames'])
                        },
                        'visited_positions': make_json_serializable(vispos) if vispos is not None else None
                    }
                    with open(json_path, 'w') as jf:
                        json.dump(serializable_snapshot, jf, indent=2)
                except Exception as e:
                    print("[WARN] failed saving json snapshot:", e)

                # 2) NPZ (for numeric / programmatic loading)
                npz_path = os.path.join(run_dir, f"metrics_snapshot_t{time_step}.npz")
                try:
                    # flatten metrics for npz
                    flat_snapshot = {
                        'timestamp': make_json_serializable(snapshot['timestamp']),
                        'episode': make_json_serializable(snapshot['episode']),
                        'timestep': make_json_serializable(snapshot['timestep']),
                        'avg_recent_return': make_json_serializable(metrics['avg_recent_return']) if metrics['avg_recent_return'] is not None else np.nan,
                        'observation_coverage_count': make_json_serializable(metrics['observation_coverage_count']),
                        'position_coverage_fraction': make_json_serializable(metrics['position_coverage_fraction']) if metrics['position_coverage_fraction'] is not None else np.nan,
                        'avg_policy_entropy': make_json_serializable(metrics['avg_policy_entropy']) if metrics['avg_policy_entropy'] is not None else np.nan,
                        'reward_discovery_frames': np.array(make_json_serializable(metrics['reward_discovery_frames']), dtype=object) if metrics.get('reward_discovery_frames') else np.array([], dtype=object),
                        'total_frames': make_json_serializable(metrics['total_frames']),
                        'visited_positions': np.array(make_json_serializable(vispos), dtype=object) if vispos is not None else np.array([], dtype=object)
                    }
                    save_metrics_npz(flat_snapshot, npz_path)
                except Exception as e:
                    print("[WARN] failed saving npz snapshot:", e)

                # reset running logs
                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes if print_running_episodes > 0 else 0
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                # also print metrics snapshot
            
                print("---- metrics snapshot ----")
                print(ppo_agent.compute_metrics())
                print("--------------------------")

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                try:
                    ppo_agent.save(checkpoint_path)
                except Exception as e:
                    print("[WARN] save model failed:", e)
                print("model saved")
                print("elapsed time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

            # move to next state
            state = state_next

        # end episode bookkeeping
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    # training finished: save final metrics timeseries and a final snapshot
    print("Training finished: saving final metrics to results directory...")

    # Save timeseries arrays as npz
    timeseries_path = os.path.join(run_dir, "timeseries.npz")
    try:
        # convert lists to numpy arrays safely (object arrays for strings)
        ts_to_save = {}
        for k, v in timeseries.items():
            if len(v) == 0:
                ts_to_save[k] = np.array([])
                continue
            try:
                ts_to_save[k] = np.array(v)
            except Exception:
                ts_to_save[k] = np.array(v, dtype=object)
        np.savez_compressed(timeseries_path, **ts_to_save)
        print("Saved timeseries ->", timeseries_path)
    except Exception as e:
        print("[WARN] failed saving timeseries:", e)

    # final snapshot JSON + NPZ
    final_metrics = ppo_agent.compute_metrics()
    final_snapshot = {
        'timestamp': datetime.now(),
        'episode': i_episode,
        'timestep': time_step,
        'metrics': final_metrics,
        'visited_positions': list(getattr(ppo_agent, 'visited_positions', []))
    }
    try:
        json_final = {
            'timestamp': make_json_serializable(final_snapshot['timestamp']),
            'episode': make_json_serializable(final_snapshot['episode']),
            'timestep': make_json_serializable(final_snapshot['timestep']),
            'metrics': {
                'avg_recent_return': make_json_serializable(final_metrics.get('avg_recent_return')),
                'observation_coverage_count': make_json_serializable(final_metrics.get('observation_coverage_count')),
                'position_coverage_fraction': make_json_serializable(final_metrics.get('position_coverage_fraction')),
                'avg_policy_entropy': make_json_serializable(final_metrics.get('avg_policy_entropy')),
                'reward_discovery_frames': make_json_serializable(final_metrics.get('reward_discovery_frames', [])),
                'total_frames': make_json_serializable(final_metrics.get('total_frames', time_step))
            },
            'visited_positions': make_json_serializable(final_snapshot['visited_positions'])
        }
        with open(os.path.join(run_dir, "metrics_final.json"), "w") as jf:
            json.dump(json_final, jf, indent=2)
        save_metrics_npz({
            'timestamp': make_json_serializable(final_snapshot['timestamp']),
            'episode': make_json_serializable(final_snapshot['episode']),
            'timestep': make_json_serializable(final_snapshot['timestep']),
            'avg_recent_return': make_json_serializable(final_metrics.get('avg_recent_return', np.nan)),
            'observation_coverage_count': make_json_serializable(final_metrics.get('observation_coverage_count', 0)),
            'position_coverage_fraction': make_json_serializable(final_metrics.get('position_coverage_fraction', np.nan)),
            'avg_policy_entropy': make_json_serializable(final_metrics.get('avg_policy_entropy', np.nan)),
            'reward_discovery_frames': np.array(make_json_serializable(final_metrics.get('reward_discovery_frames', [])), dtype=object),
            'total_frames': make_json_serializable(final_metrics.get('total_frames', time_step)),
            'visited_positions': np.array(make_json_serializable(final_snapshot['visited_positions']), dtype=object) if final_snapshot['visited_positions'] is not None else np.array([], dtype=object)
        }, os.path.join(run_dir, "metrics_final.npz"))
        print("Saved final metrics JSON + NPZ in", run_dir)
    except Exception as e:
        print("[WARN] failed saving final snapshot:", e)

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("started training at : ", start_time)
    print("finished training at : ", end_time)
    print("total training time  : ", end_time - start_time)
    print("results saved in : ", run_dir)
    print("============================================================================================")


if __name__ == '__main__':
    # run seeds 1..5 sequentially
    for seed in range(1, 6):
        print(f"\n\n\n########## STARTING RUN FOR SEED {seed} ##########\n")
        train(random_seed=seed)
        # small pause to help free resources (optional)
        time.sleep(2)
