import numpy as np
import os


class TrainingLogger:
    def __init__(self, 
                 sample_states_per_update=256,
                 sample_every_n_updates=1,
                 auto_convert_to_numpy=True):
        self.scalar_metrics = {
            'beta': [],
            'meta_loss': [],
            'icm_forward_loss': [],
            'icm_inverse_loss': [],
            'avg_intrinsic_reward': [],
            'avg_extrinsic_reward': [],
            'beta_gradient_norm': [],
            'episode_reward': [],
            'episode_length': [],
        }
        
        self.custom_metrics = {}
        
        # Update tracking
        self.update_timesteps = []
        self.update_count = 0
        
        # State-dependent samples (e.g., beta network outputs)
        self.state_samples = []  # List of dicts: {'update', 'states', 'values'}
        
        # Proxy/meta analysis data (per update arrays)
        self.array_metrics = {
            'b_intr': [],  # beta * intrinsic per timestep
            'R_ext': [],   # future extrinsic returns per timestep
        }
        
        # Sampling config
        self.sample_states_per_update = int(sample_states_per_update)
        self.sample_every_n_updates = int(sample_every_n_updates)
        self.auto_convert = auto_convert_to_numpy
        
    def _to_numpy(self, value):
        """Convert torch tensor to numpy if needed"""
        if not self.auto_convert:
            return value
        try:
            import torch
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
        except ImportError:
            pass
        return value
    
    # ============ Scalar Metrics ============
    
    def log_scalar(self, name, value, update_idx=None):
        value = self._to_numpy(value)
        if isinstance(value, np.ndarray):
            value = float(value.item()) if value.size == 1 else float(value.mean())
        else:
            value = float(value)
            
        if name in self.scalar_metrics:
            self.scalar_metrics[name].append(value)
        elif name in self.custom_metrics:
            self.custom_metrics[name].append(value)
        else:
            # Auto-create new metric
            self.custom_metrics[name] = [value]
    
    def log_scalars(self, metrics_dict, update_idx=None):
        for name, value in metrics_dict.items():
            self.log_scalar(name, value, update_idx)
    
    # ============ Array Metrics (per-update arrays) ============
    
    def log_array(self, name, array):
        array = self._to_numpy(array)
        if isinstance(array, np.ndarray):
            array = array.astype(np.float32)
        
        if name in self.array_metrics:
            self.array_metrics[name].append(array)
        else:
            self.array_metrics[name] = [array]
    
    # ============ State Sampling ============
    
    def should_sample(self):
        return (self.update_count % max(1, self.sample_every_n_updates) == 0)
    
    def log_state_samples(self, states, values, name='default'):
        states = self._to_numpy(states)
        values = self._to_numpy(values)
        
        if isinstance(states, np.ndarray):
            states = states.astype(np.float32)
        if isinstance(values, np.ndarray):
            values = values.astype(np.float32)
        
        sample_dict = {
            'update': self.update_count,
            'states': states,
            'values': values,
            'name': name
        }
        self.state_samples.append(sample_dict)
    
    def sample_and_log(self, states, values, name='default', max_samples=None):
        if not self.should_sample():
            return
        
        states = self._to_numpy(states)
        values = self._to_numpy(values)
        
        if max_samples is None:
            max_samples = self.sample_states_per_update
        
        num_available = len(states) if hasattr(states, '__len__') else states.shape[0]
        num_samples = min(max_samples, num_available)
        
        if num_samples < num_available:
            indices = np.random.permutation(num_available)[:num_samples]
            states = states[indices]
            values = values[indices]
        
        self.log_state_samples(states, values, name)
    
    # ============ Update Tracking ============
    
    def step_update(self):
        self.update_timesteps.append(self.update_count)
        self.update_count += 1
    
    # ============ Export/Import ============
    
    def export_logs(self, path_prefix):
        meta_path = path_prefix + '.meta.npz'
        meta_dict = {}
        
        # Add scalar metrics
        for name, values in self.scalar_metrics.items():
            if len(values) > 0:
                meta_dict[name] = np.array(values, dtype=np.float32)
        
        for name, values in self.custom_metrics.items():
            if len(values) > 0:
                meta_dict[f'custom_{name}'] = np.array(values, dtype=np.float32)
        
        # Add update timesteps
        meta_dict['update_timesteps'] = np.array(self.update_timesteps, dtype=np.int32)
        
        # Add array metrics (as object array since varying lengths)
        for name, arrays in self.array_metrics.items():
            if len(arrays) > 0:
                meta_dict[f'array_{name}'] = np.array(arrays, dtype=object)
        
        try:
            np.savez_compressed(meta_path, **meta_dict)
            print(f"Exported meta logs to {meta_path}")
        except Exception as e:
            print(f"Error exporting meta logs: {e}")
        
        # Samples file
        if len(self.state_samples) > 0:
            samples_path = path_prefix + '.samples.npz'
            try:
                updates = [s['update'] for s in self.state_samples]
                states_list = [s['states'] for s in self.state_samples]
                values_list = [s['values'] for s in self.state_samples]
                names_list = [s.get('name', 'default') for s in self.state_samples]
                
                np.savez_compressed(
                    samples_path,
                    updates=np.array(updates, dtype=np.int32),
                    states=np.array(states_list, dtype=object),
                    values=np.array(values_list, dtype=object),
                    names=np.array(names_list, dtype=object)
                )
                print(f"Exported sample logs to {samples_path}")
            except Exception as e:
                print(f"Error exporting sample logs: {e}")
    
    def load_logs(self, path_prefix):
        meta_path = path_prefix + '.meta.npz'
        if os.path.exists(meta_path):
            try:
                data = np.load(meta_path, allow_pickle=True)
                
                # Load scalar metrics
                for key in data.keys():
                    if key.startswith('custom_'):
                        name = key[7:]  # Remove 'custom_' prefix
                        self.custom_metrics[name] = data[key].tolist()
                    elif key.startswith('array_'):
                        name = key[6:]  # Remove 'array_' prefix
                        self.array_metrics[name] = list(data[key])
                    elif key == 'update_timesteps':
                        self.update_timesteps = data[key].tolist()
                    elif key in self.scalar_metrics:
                        self.scalar_metrics[key] = data[key].tolist()
                
                if len(self.update_timesteps) > 0:
                    self.update_count = self.update_timesteps[-1] + 1
                
                print(f"Loaded meta logs from {meta_path}")
            except Exception as e:
                print(f"Error loading meta logs: {e}")
        samples_path = path_prefix + '.samples.npz'
        if os.path.exists(samples_path):
            try:
                data = np.load(samples_path, allow_pickle=True)
                updates = data['updates']
                states = data['states']
                values = data['values']
                names = data.get('names', ['default'] * len(updates))
                
                self.state_samples = []
                for i in range(len(updates)):
                    self.state_samples.append({
                        'update': int(updates[i]),
                        'states': states[i],
                        'values': values[i],
                        'name': str(names[i])
                    })
                
                print(f"Loaded sample logs from {samples_path}")
            except Exception as e:
                print(f"Error loading sample logs: {e}")
    
    # ============ Utilities ============
    
    def get_metric(self, name):
        if name in self.scalar_metrics:
            return np.array(self.scalar_metrics[name])
        elif name in self.custom_metrics:
            return np.array(self.custom_metrics[name])
        elif name in self.array_metrics:
            return self.array_metrics[name]
        else:
            return None
    
    def get_last_n_values(self, name, n=10):
        metric = self.get_metric(name)
        if metric is not None and len(metric) > 0:
            return metric[-n:]
        return None
    
    def print_summary(self):
        print("\n=== Training Logger Summary ===")
        print(f"Total updates: {self.update_count}")
        print(f"\nScalar Metrics:")
        for name, values in self.scalar_metrics.items():
            if len(values) > 0:
                arr = np.array(values)
                print(f"  {name}: mean={arr.mean():.4f}, last={arr[-1]:.4f}, n={len(arr)}")
        
        if len(self.custom_metrics) > 0:
            print(f"\nCustom Metrics:")
            for name, values in self.custom_metrics.items():
                if len(values) > 0:
                    arr = np.array(values)
                    print(f"  {name}: mean={arr.mean():.4f}, last={arr[-1]:.4f}, n={len(arr)}")
        
        if len(self.state_samples) > 0:
            print(f"\nState Samples: {len(self.state_samples)} snapshots")
        
        print("=" * 40)
    
    def reset(self):
        for key in self.scalar_metrics:
            self.scalar_metrics[key] = []
        self.custom_metrics = {}
        self.array_metrics = {
            'b_intr': [],
            'R_ext': [],
        }
        self.update_timesteps = []
        self.update_count = 0
        self.state_samples = []