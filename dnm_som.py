import numpy as np
import soundfile as sf
import minisom
from minisom import MiniSom
import scipy.spatial.distance as ds

class DNM_SOM(MiniSom):
    # finding the nearest neighbors for the BMU
    def _nearest_neighbors(self, bmu_index, radius, msize):
        neighbors = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if (dx, dy) != (0, 0):
                    neighbor_x = bmu_index[0] + dx
                    neighbor_y = bmu_index[1] + dy
                    if 0 <= neighbor_x < msize[0] and 0 <= neighbor_y < msize[1]:
                        neighbors.append((neighbor_x, neighbor_y))
        
        return neighbors    
    
    def _get_feature_neighborhood_vector(self, bmu_index, radius, msize):
        x_bmu, y_bmu = bmu_index
        neighborhood_weights = []
        
        # iterating over the neighborhood
        for x in range(max(0, x_bmu - radius), min(msize[0], x_bmu + radius + 1)):
            for y in range(max(0, y_bmu - radius), min(msize[1], y_bmu + radius + 1)):
                neighborhood_weights.append(self.get_weights()[x, y])
        
        feature_neighborhood_vector = np.hstack(neighborhood_weights)
        
        return feature_neighborhood_vector

    def update(self, x, win, t, max_iteration):
            """Updates the weights of the neurons.

            Parameters
            ----------
            x : np.array
                Current pattern to learn.
            win : tuple
                Position of the winning neuron for x (array or tuple).
            t : int
                rate of decay for sigma and learning rate
            max_iteration : int
                If use_epochs is True:
                    Number of epochs the SOM will be trained for
                If use_epochs is False:
                    Maximum number of iterations (one iteration per sample).
            """
            eta = self._decay_function(self._learning_rate, t, max_iteration)
            # sigma and learning rate decrease with the same rule
            sig = self._decay_function(self._sigma, t, max_iteration)
            # improves the performances
            g = self.neighborhood(win, sig)*eta
            # w_new = eta * neighborhood_function * (x-w)
            
            feature_nh = np.dstack(np.unravel_index(np.argsort(-g.ravel()), (len(g), len(g[0]))))[0]
            map_nh = np.moveaxis(np.indices((len(g), len(g[0]))), 0, -1)
            map_nh = map_nh.reshape(len(g) * len(g[0]), 2)
            map_nh = map_nh[np.argsort(ds.cdist([win], map_nh)[0])]
            
            count = 0
            for i, j in feature_nh:
               if (g[i][j] != 0) and ([i ,j] != win):
                   replace_crd = map_nh[count]
                   node_wts = self._weights[i][j]
                   replace_wts = self._weights[replace_crd[0]][replace_crd[1]]
                   self._weights[replace_crd[0]][replace_crd[1]] = node_wts
                   self._weights[i][j] = replace_wts
                   count += 1

            self._weights += np.einsum('ij, ijk->ijk', g, x-self._weights)