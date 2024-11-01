import glob
import os
import numpy as np
import soundfile as sf
from minisom import MiniSom
from tqdm import tqdm
import classifiers
import pickle
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

MSIZE = [32, 32]
DATASET = "16000_pcm_speeches"
NRADIUS = 2
ITERATIONS = 2000
LEARN_RATE = 0.7
SOM_NAME = "test2"
PCA_COMPONENTS = 80
AUDIO_TIME = 1  # sec

## File Import
audio_files = glob.glob("./soundfiles/" + DATASET + "/**/*.wav")

## Data Processing
print("Processing Data")
input_len = len(audio_files)
data = [None] * input_len
labels = [None] * input_len
for i in tqdm(range(input_len)):
    samples, Fs = librosa.load(audio_files[i], sr=16000)
    labels[i] = os.path.basename(os.path.dirname(audio_files[i]))
    data[i] = librosa.feature.mfcc(y=samples[0:Fs * AUDIO_TIME], sr=Fs).flatten()

print(np.array(data).shape)

print("Splitting Data")
if PCA_COMPONENTS != 0:
    pca = PCA(n_components=PCA_COMPONENTS)
    data = pca.fit_transform(data)

print(np.array(data).shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, train_size=0.9, stratify=labels)

class CustomSOM(MiniSom):
    # override init 
    def __init__(self, x, y, input_len, sigma, learning_rate, random_seed=None):
        super().__init__(x, y, input_len, sigma, learning_rate, random_seed)
        
        # initialize the map neighborhood funcion
        print('Hello World!')

    # finding the nearest neighbors for the BMU
    def _nearest_neighbors(self, bmu_index):
        neighbors = []
        
        for dx in range(-NRADIUS, NRADIUS + 1):
            for dy in range(-NRADIUS, NRADIUS + 1):
                
                if (dx, dy) != (0, 0):
                    neighbor_x = bmu_index[0] + dx
                    neighbor_y = bmu_index[1] + dy
                    
                    if 0 <= neighbor_x < MSIZE[0] and 0 <= neighbor_y < MSIZE[1]:
                        neighbors.append((neighbor_x, neighbor_y))
        
        return neighbors    
    
    def _get_feature_neighborhood_vector(self, som, bmu_index, radius):
        x_bmu, y_bmu = bmu_index
        neighborhood_weights = []
        
        # iterating over the neighborhood
        for x in range(max(0, x_bmu - radius), min(MSIZE[0], x_bmu + radius + 1)):
            for y in range(max(0, y_bmu - radius), min(MSIZE[1], y_bmu + radius + 1)):
                neighborhood_weights.append(som.get_weights()[x, y])
        
        feature_neighborhood_vector = np.hstack(neighborhood_weights)
        
        return feature_neighborhood_vector

    def train_random(self, X_train, ITERATIONS, verbose):
        super().train_random(X_train, ITERATIONS, verbose)
        
        # update the map neighborhood vector based on the feature vector
        print('Hello World!')

## CustomSOM
# Init SOM
print("Initializing SOM")
som = CustomSOM(
    x=MSIZE[0],
    y=MSIZE[1],
    input_len=data[0].shape[0],
    sigma=NRADIUS,
    learning_rate=LEARN_RATE,
    random_seed=None
)

som.pca_weights_init(X_train)

# Train SOM
print("Training SOM")
som.train_random(X_train, ITERATIONS, verbose=True)

# Classification
print(classification_report(y_test, classifiers.classify_BMU(som, X_test, X_train, y_train)))

# Save SOM
if not os.path.exists('./som_files/' + SOM_NAME):
    os.makedirs('./som_files/' + SOM_NAME)
with open('./som_files/' + SOM_NAME + "/som.p", 'wb') as outfile:
    pickle.dump(som, outfile)
