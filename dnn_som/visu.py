from matplotlib import pyplot as plt
from minisom import MiniSom
import matplotlib.gridspec as gridspec
import pickle

import numpy as np

INFILE = "SNR_TIME_BN"
DATASET = "16000_pcm_speeches"
PCA_COMPONENTS = 80
AUDIO_TIME = 0.99 #sec

with open('./data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_" + str(PCA_COMPONENTS) + "PCA_data.p", 'rb') as infile:
#with open('./data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_" + "data.p", 'rb') as infile:    
    data = pickle.load(infile)
with open('./data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_labels.p", 'rb') as infile:
    labels = pickle.load(infile)
with open('./som_files/' + INFILE + '/som.p', 'rb') as infile:
    som = pickle.load(infile)

## Visu
label_names = list(set(labels))
labels_map = som.labels_map(data, labels)

fig = plt.figure(figsize=(9, 9))
the_grid = gridspec.GridSpec(som.x, som.y, fig)
for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in label_names]
    plt.subplot(the_grid[som.x-1-position[0], position[1]], aspect=1)
    #plt.subplot(the_grid[som.x-1-position[1], position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)

plt.legend(patches, label_names)
plt.show()