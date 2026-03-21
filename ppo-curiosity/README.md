# PPO for MiniGrid

This repository provides a clean and simple implementation of **Proximal Policy Optimization (PPO)** for the **MiniGrid** environments.

## Performance
Average over 5 seeds (1→5):

Doorkey:
<p float="left">
  <img src="assets\performance\MiniGrid-DoorKey-8x8-v0.png" width="100%" />
</p>

Empty:
<p float="left">
  <img src="assets\performance\MiniGrid-Empty-16x16-v0.png" width="100%" />
</p>

RedBlueDoors:
<p float="left">
  <img src="assets\performance\MiniGrid-RedBlueDoors-8x8-v0.png" width="100%" />
</p>

UnlockPickup:
<p float="left">
  <img src="assets\performance\MiniGrid-UnlockPickup-v0.png" width="100%" />
</p>

## Training

You can start training with the [PIPELINE](https://colab.research.google.com/drive/1oBcsHoY81DZ_x9poIW-CsWmBY5-Q-HmA)

## TODO

run intrinsic_strength = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
env: MiniGrid-LavaCrossingS9N3-v0, MiniGrid-KeyCorridorS3R3-v0


## References

[1] *Proximal Policy Optimization Algorithms* — Schulman et al., 2017

[2] [PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
