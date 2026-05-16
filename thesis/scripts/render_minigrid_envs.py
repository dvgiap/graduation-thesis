"""Render top-down screenshots of the 6 MiniGrid benchmark environments.

Outputs to thesis/figures/env_<name>.png.
Run from project root:
    python thesis/scripts/render_minigrid_envs.py
"""

from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  (registers minigrid envs with gymnasium)
from PIL import Image

ENVS = [
    ("MiniGrid-DoorKey-8x8-v0", "doorkey"),
    ("MiniGrid-Empty-16x16-v0", "empty"),
    ("MiniGrid-KeyCorridorS3R3-v0", "keycorridor"),
    ("MiniGrid-LavaCrossingS9N3-v0", "lavacrossing"),
    ("MiniGrid-RedBlueDoors-8x8-v0", "redbluedoors"),
    ("MiniGrid-UnlockPickup-v0", "unlockpickup"),
]

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def render_env(env_id: str, slug: str, seed: int = 0) -> Path:
    env = gym.make(env_id, render_mode="rgb_array", highlight=False)
    env.reset(seed=seed)
    frame = env.unwrapped.get_frame(highlight=False, tile_size=32)
    env.close()
    out_path = FIG_DIR / f"env_{slug}.png"
    Image.fromarray(frame).save(out_path)
    return out_path


def main() -> None:
    for env_id, slug in ENVS:
        out = render_env(env_id, slug)
        print(f"Saved {out.name}  ({env_id})")


if __name__ == "__main__":
    main()
