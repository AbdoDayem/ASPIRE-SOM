from matplotlib import pyplot as plt
from minisom import MiniSom
import matplotlib.gridspec as gridspec
import pickle

INFILE = "dnm_test1"
DATASET = "16000_pcm_speeches"
PCA_COMPONENTS = 80
AUDIO_TIME = 0.99 #sec

with open('./data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_" + str(PCA_COMPONENTS) + "PCA_data.p", 'rb') as infile:
    data = pickle.load(infile)
with open('./data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_labels.p", 'rb') as infile:
    labels = pickle.load(infile)
with open('./som_files/' + INFILE + '/dnm_som.p', 'rb') as infile:
    som = pickle.load(infile)

## Visu
labels_map = som.labels_map(data, labels)
label_names = list(set(labels))

fig = plt.figure(figsize=(9, 9))
the_grid = gridspec.GridSpec(som.x, som.y, fig)
for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in label_names]
    plt.subplot(the_grid[som.x-1-position[1], position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)

plt.legend(patches, label_names,  bbox_to_anchor=(som.x, som.y/4 * 3))
plt.show()