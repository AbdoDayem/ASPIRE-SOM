from collections import Counter
import numpy as np

# Monkey-patch the labels_map function
def patched_labels_map(self, data, labels):
    """Monkey-patched version of labels_map to use tuples as keys."""
    winmap = {}
    for x, l in zip(data, labels):
        # Ensure winner(x) returns a hashable tuple
        win_position = tuple(map(int, self.winner(x)))  # Explicitly cast elements to int
        if win_position not in winmap:
            winmap[win_position] = []
        winmap[win_position].append(l)
    return winmap

def classify_BMU(som, X_test, X_train, y_train):
    # Apply the patch
    from minisom import MiniSom
    MiniSom.labels_map = patched_labels_map

    winmap = som.labels_map(X_train, y_train)
    
    # Compute the default class
    all_labels = [label for labels in winmap.values() for label in labels]
    default_class = Counter(all_labels).most_common(1)[0][0]
    
    result = []
    for d in X_test:
        win_position = tuple(map(int, som.winner(d)))  # Force conversion to tuple
        if win_position in winmap:
            # Find the most common label for this position
            result.append(Counter(winmap[win_position]).most_common(1)[0][0])
        else:
            # Use the default class if the position is not in winmap
            result.append(default_class)
    return result
