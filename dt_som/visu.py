import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import pickle
import cupy as cp

INFILE = "dt_test2"
DATASET = "16000_pcm_speeches"
PCA_COMPONENTS = 80
AUDIO_TIME = 0.9

for i in range(2):
    with open('../data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_" + str(PCA_COMPONENTS) + "PCA_data.p", 'rb') as infile:
        data = pickle.load(infile)
    with open('../data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_labels.p", 'rb') as infile:
        labels = pickle.load(infile)
    
    if i != 0:
        def patched_labels_map(self, data, labels):
            """Monkey-patched version of labels_map to use tuples as keys."""
            winmap = {}
            for x, l in zip(data, labels):
                x = cp.asarray(x)  # Convert x to a CuPy array
                win_position = tuple(map(int, self.winner(x)))  # Explicitly cast elements to int
                if win_position not in winmap:
                    winmap[win_position] = []
                winmap[win_position].append(l)
            return winmap
        
        # Apply the patch
        from minisom import MiniSom
        MiniSom.labels_map = patched_labels_map
        
        with open('../som_files/' + INFILE + '/dt_som.p', 'rb') as infile:
            som = pickle.load(infile)
    else:
        with open('../som_files/' + INFILE + '/som.p', 'rb') as infile:
            som = pickle.load(infile)

    ## Visu
    labels_map = som.labels_map(data, labels)
    label_names = list(set(labels))

    fig = plt.figure(figsize=(9, 9))
    # Access the grid size from the SOM's weights (x and y dimensions)
    grid_size = som._weights.shape  # shape is (x, y, n_features)
    x, y = grid_size[0], grid_size[1]  # x is number of rows, y is number of columns

    # Create the grid specification for plotting
    the_grid = gridspec.GridSpec(x, y, fig)

    from collections import Counter

    for position in labels_map.keys():
        # Count the frequency of each label at the current position
        label_counts = Counter(labels_map[position])
        
        # Create a list of label fractions based on the counts
        label_fracs = [label_counts.get(l, 0) for l in label_names]
        
        # Create the pie chart
        plt.subplot(the_grid[x - 1 - position[1], position[0]], aspect=1)
        patches, texts = plt.pie(label_fracs)

    # Add legend and show the plot
    plt.legend(patches, label_names, bbox_to_anchor=(x, y / 4 * 3))
    plt.show()
