import cupy as cp
import numpy as np
from minisom import MiniSom
from tqdm import tqdm

class DTSOM_GPU(MiniSom):
    def __init__(self, x, y, input_len, sigma=1.0, tau=0.5, learning_rate=0.5, density_threshold=0.1):
        super().__init__(x, y, input_len, sigma, tau, learning_rate)
        
        # Initialize weight and density map on the GPU
        self._weights = cp.zeros((x, y, input_len))
        self.density_map = cp.zeros((x, y))
        self.density_threshold = density_threshold

    def random_weights_init(self, data):
        """Initialize weights with random data points."""
        data_gpu = cp.asarray(data) # Convert to a CuPy array
        for it in np.ndindex(self._weights.shape[:2]):
            rand_i = np.random.randint(0, len(data))
            self._weights[it] = data_gpu[rand_i]

    def update_density(self, data_point):
        """Update the density map based on a new data point."""
        bmu = self.winner(data_point)
        self.density_map[bmu[0], bmu[1]] += 1

    def decay_density(self):
        """Apply decay to density values to simulate decreasing memory over time."""
        self.density_map *= 0.99

    def update_weights_parallel(self, bmu_x, bmu_y, data_point, learning_rate, sigma):
        """Optimized weight update (using GPU with vectorized operations)"""
        weights = self._weights
        density_map = self.density_map
        
        # Calculating the necessary values
        rows, cols = weights.shape[0], weights.shape[1]        
        data_point_gpu = cp.asarray(data_point)        
        max_density = cp.max(density_map)        
        i_coords, j_coords = cp.meshgrid(cp.arange(rows), cp.arange(cols), indexing='ij')        
        dist_sq = (bmu_x - i_coords) ** 2 + (bmu_y - j_coords) ** 2
        
        # Computing factors for influencing weights
        influence = cp.exp(-dist_sq / (2 * sigma ** 2))        
        density_factors = cp.maximum(1.0, density_map / (max_density + 1e-5))
        adjusted_learning_rate = learning_rate / density_factors
        
        # Expand the influence and adjusted learning rate to match the shape of weights
        influence_expanded = influence[..., cp.newaxis]
        adjusted_learning_rate_expanded = adjusted_learning_rate[..., cp.newaxis]
        influence_broadcasted = influence_expanded * adjusted_learning_rate_expanded
        data_point_expanded = data_point_gpu[cp.newaxis, cp.newaxis, :]
        weight_updates = influence_broadcasted * (data_point_expanded - weights)
        
        weights += weight_updates

    def train(self, data, num_iterations, verbose=False):
        """Train the map with density tracking and show progress."""
        if verbose:
            pbar = tqdm(range(num_iterations), desc="Training Density SOM", unit="iteration")
        else:
            pbar = range(num_iterations)

        data = cp.asarray(data)  # Convert data to a CuPy array

        for _ in pbar:
            # Shuffle data and iterate through each data point
            cp.random.shuffle(data)
            for data_point in data:
                self.update_density(data_point)
                bmu_idx = self.winner(data_point)
                self.update_weights_parallel(bmu_idx[0], bmu_idx[1], data_point, self._learning_rate, self._sigma)

            # Apply density decay after each iteration
            self.decay_density()

class DTSOM_CPU(MiniSom):
    def __init__(self, x, y, input_len, sigma=1.0, tau=0.5, learning_rate=0.5, density_threshold=0.1):
        super().__init__(x, y, input_len, sigma, tau, learning_rate)
        
        # Density Tracking
        self.density_map = np.zeros((x, y))
        self.density_threshold = density_threshold
    
    def update_density(self, data_point):
        """Update the density map based on a new data point."""
        bmu = self.winner(data_point)
        self.density_map[bmu[0], bmu[1]] += 1

    def decay_density(self):
        """Apply decay to density values to simulate decreasing memory over time."""
        self.density_map *= 0.99  # Decay factor for density map

    def _update_weights(self, bmu_idx, data_point, learning_rate, sigma):
        """Override the weight update to incorporate density tracking."""
        bmu_x, bmu_y = bmu_idx
        # Update the weights of the neurons around the BMU
        for i in range(self._weights.shape[0]):
            for j in range(self._weights.shape[1]):
                # Calculate the distance of the neuron from the BMU
                dist = np.linalg.norm([bmu_x - i, bmu_y - j])
                influence = np.exp(-dist / (2 * sigma ** 2))
                
                # Adjust the learning rate based on the density of the current neuron
                density_factor = max(1.0, self.density_map[i, j] / (np.max(self.density_map) + 1e-5))
                adjusted_learning_rate = learning_rate / density_factor
                
                # Update the weights with the adjusted learning rate
                self._weights[i, j] += adjusted_learning_rate * influence * (data_point - self._weights[i, j])

    def train(self, data, num_iterations, verbose=False):
        """Train the map with density tracking and show progress."""
        if verbose:
            pbar = tqdm(range(num_iterations), desc="Training Density SOM", unit="iteration")
        else:
            pbar = range(num_iterations)

        for _ in pbar:
            # Shuffle data and iterate through each data point
            np.random.shuffle(data)
            for data_point in data:
                self.update_density(data_point)
                bmu_idx = self.winner(data_point)
                self._update_weights(bmu_idx, data_point, self._learning_rate, self._sigma)
            
            # Apply density decay after each iteration
            self.decay_density()
