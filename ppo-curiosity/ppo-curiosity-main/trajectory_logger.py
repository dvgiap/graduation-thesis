"""
Trajectory Logger Module
-------------------------
Standalone module for recording and visualizing agent trajectories.
Can be used with any RL algorithm (PPO, DQN, SAC, etc.)

Features:
- Record agent positions over time
- Export to CSV/NPZ formats
- Generate heatmaps for any time window
- Support for grid-based and continuous environments
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TrajectoryLogger:
    """
    Records and visualizes agent trajectories across episodes.
    
    Usage:
        logger = TrajectoryLogger()
        
        # During training
        logger.record_position((x, y), episode=0, timestep=10)
        
        # After training
        logger.save_csv("trajectory.csv")
        logger.save_heatmap_png("heatmap.png", grid_shape=(10, 10))
    """
    
    def __init__(self):
        """Initialize empty trajectory storage."""
        self.trajectory = []  # List of (episode, timestep, x, y) tuples
    
    def record_position(self, pos, episode=None, timestep=None, 
                       as_state=False, grid_shape=None):
        """
        Record agent position at current timestep.
        
        Args:
            pos: Position as (x, y) tuple/list, or raw state vector if as_state=True
            episode: Episode number (int or None)
            timestep: Timestep within episode (int or None)
            as_state: If True, attempt to extract position from state vector
            grid_shape: (width, height) tuple - required when as_state=True
        
        Examples:
            # Direct position
            logger.record_position((5, 3), episode=0, timestep=100)
            
            # Extract from state vector (grid-based env)
            logger.record_position(state, episode=0, timestep=100, 
                                 as_state=True, grid_shape=(10, 10))
        """
        if as_state:
            x, y = self._extract_position_from_state(pos, grid_shape)
        else:
            x, y = self._parse_position(pos)
        
        self.trajectory.append((
            int(episode) if episode is not None else -1,
            int(timestep) if timestep is not None else -1,
            int(x),
            int(y)
        ))
    
    def _parse_position(self, pos):
        """Parse position from various input formats."""
        try:
            x, y = int(pos[0]), int(pos[1])
            return x, y
        except (TypeError, IndexError, ValueError):
            # Fallback for invalid input
            return 0, 0
    
    def _extract_position_from_state(self, state, grid_shape):
        """
        Best-effort extraction of position from state vector.
        
        Heuristics:
        1. If state matches grid_shape * 3 (RGB image), find max-sum pixel
        2. If state has 2 elements, treat as (x, y)
        3. Otherwise, take first 2 elements as (x, y)
        """
        arr = np.array(state)
        
        if grid_shape is not None:
            w, h = grid_shape
            
            # Check if state is flattened RGB grid
            if arr.size == w * h * 3:
                img = arr.reshape((w, h, 3))
                sums = img.sum(axis=2)
                idx = np.unravel_index(np.argmax(sums), (w, h))
                return int(idx[0]), int(idx[1])
            
            # Check if state is (x, y) pair
            elif arr.size == 2:
                return int(arr[0]), int(arr[1])
        
        # Fallback: assume first 2 elements are (x, y)
        if arr.size >= 2:
            return int(arr[0]), int(arr[1])
        
        return 0, 0
    
    def get_trajectory_array(self):
        """
        Get trajectory as numpy array.
        
        Returns:
            np.ndarray: Shape [N, 4] with columns [episode, timestep, x, y]
        """
        if len(self.trajectory) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return np.array(self.trajectory, dtype=np.int32)
    
    def clear(self):
        """Clear all recorded trajectory data."""
        self.trajectory.clear()
    
    # -------------------- Export Methods --------------------
    
    def save_csv(self, csv_path):
        """
        Save trajectory to CSV file.
        
        Args:
            csv_path: Output file path (e.g., "trajectory.csv")
        """
        arr = self.get_trajectory_array()
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'timestep', 'x', 'y'])
            for row in arr:
                writer.writerow(row.tolist())
        
        print(f"Trajectory saved to: {csv_path} ({len(arr)} positions)")
    
    def save_npz(self, npz_path):
        """
        Save trajectory to compressed NPZ file.
        
        Args:
            npz_path: Output file path (e.g., "trajectory.npz")
        """
        arr = self.get_trajectory_array()
        os.makedirs(os.path.dirname(npz_path) or '.', exist_ok=True)
        np.savez_compressed(npz_path, trajectory=arr)
        print(f"Trajectory saved to: {npz_path} ({len(arr)} positions)")
    
    def load_npz(self, npz_path):
        """
        Load trajectory from NPZ file.
        
        Args:
            npz_path: Input file path
        """
        data = np.load(npz_path)
        if 'trajectory' in data:
            traj_np = data['trajectory']
            self.trajectory = [tuple(map(int, row.tolist())) for row in traj_np]
            print(f"Loaded {len(self.trajectory)} positions from {npz_path}")
    
    # -------------------- Heatmap Methods --------------------
    
    def compute_heatmap(self, start_idx=None, end_idx=None, 
                       grid_shape=None, start_episode=None, end_episode=None):
        """
        Compute visit count heatmap for a time window.
        
        Args:
            start_idx: Start index in trajectory list (inclusive)
            end_idx: End index in trajectory list (exclusive)
            grid_shape: (width, height) for heatmap bins, or None to infer
            start_episode: Alternative: filter by episode number (inclusive)
            end_episode: Alternative: filter by episode number (exclusive)
        
        Returns:
            np.ndarray: Heatmap of shape (width, height) with visit counts
        
        Examples:
            # Last 1000 positions
            heatmap = logger.compute_heatmap(start_idx=-1000)
            
            # Specific episodes
            heatmap = logger.compute_heatmap(start_episode=10, end_episode=20)
            
            # Force grid size
            heatmap = logger.compute_heatmap(grid_shape=(20, 20))
        """
        arr = self.get_trajectory_array()
        
        if arr.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32)
        
        # Filter by episode if specified
        if start_episode is not None or end_episode is not None:
            mask = np.ones(arr.shape[0], dtype=bool)
            if start_episode is not None:
                mask &= (arr[:, 0] >= start_episode)
            if end_episode is not None:
                mask &= (arr[:, 0] < end_episode)
            arr = arr[mask]
            
            if arr.shape[0] == 0:
                return np.zeros((0, 0), dtype=np.float32)
        
        # Filter by index
        s = 0 if start_idx is None else max(0, int(start_idx))
        e = arr.shape[0] if end_idx is None else min(arr.shape[0], int(end_idx))
        sel = arr[s:e, 2:4]  # Extract x, y columns
        
        xs = sel[:, 0]
        ys = sel[:, 1]
        
        # Determine grid dimensions
        if grid_shape is None:
            max_x = int(xs.max()) if xs.size > 0 else 0
            max_y = int(ys.max()) if ys.size > 0 else 0
            width = max_x + 1
            height = max_y + 1
        else:
            width, height = grid_shape
        
        # Compute 2D histogram
        heat, _, _ = np.histogram2d(
            xs, ys, 
            bins=[width, height], 
            range=[[0, width], [0, height]]
        )
        
        return heat  # Shape: (width, height)
    
    def save_heatmap_png(self, out_path, start_idx=None, end_idx=None,
                        grid_shape=None, start_episode=None, end_episode=None,
                        normalize=False, cmap='hot', dpi=150, 
                        annotate_max=False, title=None):
        """
        Generate and save heatmap visualization as PNG.
        
        Args:
            out_path: Output file path (e.g., "heatmap.png")
            start_idx: Start index in trajectory
            end_idx: End index in trajectory
            grid_shape: (width, height) for bins
            start_episode: Filter by start episode
            end_episode: Filter by end episode
            normalize: If True, scale heatmap to [0, 1]
            cmap: Matplotlib colormap name
            dpi: Image resolution
            annotate_max: If True, mark position with max visits
            title: Custom plot title
        
        Examples:
            # Basic heatmap
            logger.save_heatmap_png("heatmap.png", grid_shape=(10, 10))
            
            # Episodes 0-50 with annotation
            logger.save_heatmap_png("early_training.png", 
                                   start_episode=0, end_episode=50,
                                   annotate_max=True)
        """
        heat = self.compute_heatmap(start_idx, end_idx, grid_shape, 
                                    start_episode, end_episode)
        
        if heat.size == 0:
            print("Warning: No trajectory data to plot")
            self._save_empty_heatmap(out_path, dpi)
            return
        
        # Normalize if requested
        if normalize:
            max_val = heat.max()
            if max_val > 0:
                heat = heat / float(max_val)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Transpose so x is horizontal, y is vertical
        # origin='lower' makes (0,0) bottom-left
        im = ax.imshow(heat.T, origin='lower', 
                      interpolation='nearest', 
                      cmap=cmap, aspect='equal')
        
        ax.set_xlabel('X position', fontsize=12)
        ax.set_ylabel('Y position', fontsize=12)
        
        # Set title
        if title is None:
            title = 'Agent Trajectory Heatmap'
            if start_episode is not None or end_episode is not None:
                title += f'\nEpisodes {start_episode or 0}-{end_episode or "end"}'
        ax.set_title(title, fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Visit Count', fontsize=11)
        
        # Annotate maximum if requested
        if annotate_max:
            idx = np.unravel_index(np.argmax(heat), heat.shape)
            ax.plot(idx[0], idx[1], marker='x', markersize=12, 
                   markeredgewidth=3, color='cyan')
            ax.text(idx[0] + 0.3, idx[1] + 0.3, 
                   f'Max: ({idx[0]}, {idx[1]})\n{int(heat[idx])} visits',
                   color='cyan', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Save figure
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        
        print(f"Heatmap saved to: {out_path}")
    
    def _save_empty_heatmap(self, out_path, dpi):
        """Save placeholder image when no data available."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, 'No trajectory data', 
               ha='center', va='center', fontsize=16)
        ax.axis('off')
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
    
    # -------------------- Analysis Methods --------------------
    
    def get_statistics(self, start_episode=None, end_episode=None):
        """
        Compute trajectory statistics.
        
        Returns:
            dict: Statistics including total positions, episodes, 
                  unique positions, most visited position, etc.
        """
        arr = self.get_trajectory_array()
        
        if arr.shape[0] == 0:
            return {
                'total_positions': 0,
                'num_episodes': 0,
                'unique_positions': 0
            }
        
        # Filter by episode
        if start_episode is not None or end_episode is not None:
            mask = np.ones(arr.shape[0], dtype=bool)
            if start_episode is not None:
                mask &= (arr[:, 0] >= start_episode)
            if end_episode is not None:
                mask &= (arr[:, 0] < end_episode)
            arr = arr[mask]
        
        episodes = arr[:, 0]
        positions = arr[:, 2:4]
        
        # Count unique positions
        unique_pos = np.unique(positions, axis=0)
        
        # Find most visited position
        heat = self.compute_heatmap(start_episode=start_episode, 
                                    end_episode=end_episode)
        if heat.size > 0:
            max_idx = np.unravel_index(np.argmax(heat), heat.shape)
            most_visited = (int(max_idx[0]), int(max_idx[1]))
            max_visits = int(heat[max_idx])
        else:
            most_visited = None
            max_visits = 0
        
        return {
            'total_positions': len(arr),
            'num_episodes': len(np.unique(episodes[episodes >= 0])),
            'unique_positions': len(unique_pos),
            'most_visited_position': most_visited,
            'max_visit_count': max_visits,
            'x_range': (int(positions[:, 0].min()), int(positions[:, 0].max())),
            'y_range': (int(positions[:, 1].min()), int(positions[:, 1].max()))
        }
    
    def print_statistics(self, start_episode=None, end_episode=None):
        """Print trajectory statistics to console."""
        stats = self.get_statistics(start_episode, end_episode)
        
        print("\n" + "="*50)
        print("TRAJECTORY STATISTICS")
        print("="*50)
        print(f"Total positions recorded: {stats['total_positions']}")
        print(f"Number of episodes: {stats['num_episodes']}")
        print(f"Unique positions visited: {stats['unique_positions']}")
        if stats['most_visited_position']:
            print(f"Most visited position: {stats['most_visited_position']} "
                  f"({stats['max_visit_count']} visits)")
        print(f"X range: {stats['x_range']}")
        print(f"Y range: {stats['y_range']}")
        print("="*50 + "\n")


# -------------------- Integration Helper --------------------

def integrate_with_checkpoint(checkpoint_dict, logger):
    """
    Helper function to save/load trajectory with model checkpoints.
    
    Usage in training script:
        # Saving
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            ...
        }
        integrate_with_checkpoint(checkpoint, trajectory_logger)
        torch.save(checkpoint, 'model.pt')
        
        # Loading
        checkpoint = torch.load('model.pt')
        new_logger = TrajectoryLogger()
        integrate_with_checkpoint(checkpoint, new_logger)
    """
    if isinstance(logger, TrajectoryLogger):
        # Saving: add trajectory to checkpoint
        checkpoint_dict['trajectory_np'] = logger.get_trajectory_array()
    elif 'trajectory_np' in checkpoint_dict:
        # Loading: restore trajectory from checkpoint
        traj_np = checkpoint_dict['trajectory_np']
        if isinstance(traj_np, np.ndarray) and traj_np.size > 0:
            logger.trajectory = [tuple(map(int, row.tolist())) 
                                for row in traj_np]