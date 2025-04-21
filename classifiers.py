import numpy as np
import dnm_som
import minisom

def classify_BMU(som, X_test, X_train, y_train):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = som.labels_map(X_train, y_train)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in X_test:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

def classify_minsum(som, X_test, X_train, y_train):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = som.labels_map(X_train, y_train)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    dist_map = som._distance_from_weights(X_test)
    #winmap_flat = winmap.reshape(-1, winmap.shape[2])
    #weights_flat = np.dstack(np.unravel_index(np.argsort(-som._weights.ravel()), (len(som._weights), len(som._weights[0]))))[0]
    weights_flat = som._weights.reshape(-1, som._weights.shape[2])
    for i in range(len(X_test)):
        counts = {}
        for j in range(len(weights_flat)):
            print(np.unravel_index(j, som._weights.shape))
            counts[winmap[np.unravel_index(j, som._weights.shape)]] = counts.get(winmap[np.unravel_index(j, som._weights.shape)], 0) + dist_map[i,j]
        result.append(min(counts))
    return result