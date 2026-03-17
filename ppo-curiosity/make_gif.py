import os
import glob
import time
from datetime import datetime

import numpy as np
from PIL import Image

import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper

from ppo import PPO


# ---------------- helper utilities ----------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def img_to_uint8_rgb(img):
    """
    ensure image is numpy uint8 and has 3 rgb channels.
    if img is float in [0,1], scale to 0..255.
    """
    if isinstance(img, Image.Image):
        img = np.asarray(img)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0.0, 1.0)
        img = (255 * img).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # if image is 2d -> convert to 3 channels
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    # if has alpha channel, drop alpha
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


# ---------------- save frames from running env ----------------
def save_gif_images(env_name, has_continuous_action_space, max_ep_len, action_std,
                    total_test_episodes=1, models_dir="models"):
    """
    run env with ppo (load weights from models/<env_name>/...) and save each frame to folder:
      PPO_gif_images/<env_name>/000001.png ...
    returns: path to image folder and total number of frames saved
    """
    print("=== save_gif_images:", env_name)
    # ------ create env with render_mode rgb_array ------
    env = gym.make(env_name, render_mode="rgb_array")
    env = FlatObsWrapper(env)

    # state & action dim
    state_dim = int(np.prod(env.observation_space.shape))
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # create ppo agent (keep your constructor)
    lr_actor = 0.0003
    lr_critic = 0.001
    gamma = 0.99
    K_epochs = 80
    eps_clip = 0.2
    if has_continuous_action_space:
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space, action_std)
    else:
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space)

    # load pretrained checkpoint if exists
    random_seed = 0
    run_num_pretrained = 0
    directory = os.path.join(models_dir, env_name)
    checkpoint_path = os.path.join(directory, "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained))
    if os.path.exists(checkpoint_path):
        print("Loading weights from:", checkpoint_path)
        ppo_agent.load(checkpoint_path)
    else:
        print("Warning: checkpoint not found at:", checkpoint_path)
        print("Proceeding without loading weights (agent may be random).")

    # directory to save images
    images_root = "PPO_gif_images"
    images_env_dir = os.path.join(images_root, env_name)
    ensure_dir(images_env_dir)

    frame_count = 0
    test_running_reward = 0.0

    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0.0
        state, _ = env.reset()
        # if state is dict / not a vector, ppo.select_action may need handling;
        # leave as is because you have a corresponding ppo.
        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)

            # render frame; ensure uint8 rgb type
            frame = env.render()
            frame = img_to_uint8_rgb(frame)
            pil_img = Image.fromarray(frame).convert("RGB")

            frame_count += 1
            filename = os.path.join(images_env_dir, str(frame_count).zfill(6) + ".png")
            pil_img.save(filename)  # save as png for safety

            if done:
                break

        # clear buffer if exists
        if hasattr(ppo_agent, "buffer") and hasattr(ppo_agent.buffer, "clear"):
            ppo_agent.buffer.clear()
        test_running_reward += ep_reward
        print(f"episode {ep}  reward: {round(ep_reward,2)}  frames so far: {frame_count}")

    env.close()
    avg_test_reward = round(test_running_reward / total_test_episodes, 2)
    print("done. total frames saved:", frame_count, " average reward:", avg_test_reward)
    return images_env_dir, frame_count


# ---------------- create gif from saved frames ----------------
def save_gif(env_name, images_dir=None, gif_num=0,
             total_timesteps=300, step=1, frame_duration=100):
    """
    create gif from folder images PPO_gif_images/<env_name>/*.png
    - total_timesteps: max number of images to take from folder (in case many frames)
    - step: take 1 image per step frames -> degree of downsampling
    - frame_duration: duration of each frame (ms)
    """
    print("=== save_gif:", env_name)
    if images_dir is None:
        images_dir = os.path.join("PPO_gif_images", env_name)

    pattern = os.path.join(images_dir, "*.png")
    img_paths = sorted(glob.glob(pattern))
    if len(img_paths) == 0:
        print("no frames found in", images_dir)
        return None

    # limit total frames, and downsample by step
    img_paths = img_paths[:total_timesteps]
    img_paths = img_paths[::step]

    print("frames used for gif:", len(img_paths))
    total_duration_s = round(len(img_paths) * frame_duration / 1000.0, 2)
    print("total gif duration (s):", total_duration_s)

    # create folder gifs/<env_name>/
    gif_dir = os.path.join("gifs", env_name)
    ensure_dir(gif_dir)
    gif_path = os.path.join(gif_dir, f"PPO_{env_name}_gif_{gif_num}.gif")

    # open all images, make size/mode consistent
    frames = []
    base_size = None
    for p in img_paths:
        try:
            im = Image.open(p).convert("RGBA")
            if base_size is None:
                base_size = im.size
            else:
                # if frame has different size, resize to base_size
                if im.size != base_size:
                    im = im.resize(base_size, resample=Image.BILINEAR)
            frames.append(im.convert("RGBA"))
        except Exception as e:
            print("warning: failed to open", p, ":", e)

    if len(frames) == 0:
        print("no valid frames after loading.")
        return None

    # pillow requires palette-based 'P' for gif - pillow will convert when saving; we use rgba first
    first = frames[0].convert("RGBA")
    rest = [f.convert("RGBA") for f in frames[1:]]

    # save gif. duration is in ms. loop=0 => infinite loop.
    first.save(fp=gif_path, format="GIF", append_images=rest, save_all=True,
               duration=frame_duration, loop=0, optimize=False)
    print("saved gif at:", gif_path)
    return gif_path


# ---------------- utility list gif sizes ----------------
def list_gif_size(env_name):
    gif_dir = os.path.join("gifs", env_name)
    pattern = os.path.join(gif_dir, "*.gif")
    gif_paths = sorted(glob.glob(pattern))
    if len(gif_paths) == 0:
        print("no gifs found for", env_name)
        return
    for p in gif_paths:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"{p}\t{size_mb:.2f} mb")


# ---------------- main ----------------
if __name__ == "__main__":
    env_name = "MiniGrid-DoorKey-8x8-v0"
    has_continuous_action_space = False
    max_ep_len = 1000
    action_std = None

    # 1) save images (if you want to overwrite folder, you can manually remove PPO_gif_images/<env_name>)
    images_dir, n_frames = save_gif_images(env_name, has_continuous_action_space, max_ep_len, action_std,
                                           total_test_episodes=1, models_dir="models")

    # 2) create gif from saved images
    # note: step=1 uses every frame (large gif), step=2 drops 1 frame every 2 to make it lighter
    gif_path = save_gif(env_name, images_dir=images_dir, gif_num=0,
                        total_timesteps=1000, step=1, frame_duration=120)

    # 3) print gif sizes
    list_gif_size(env_name)
