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

from dnm_som import DNM_SOM

SOM_NAME = "dnm_test1"
MSIZE = [32, 32]
NRADIUS = 3
ITERATIONS = 3000
LEARN_RATE = 0.7
DATASET = "16000_pcm_speeches"
PCA_COMPONENTS = 80
AUDIO_TIME = 0.99 #sec

with open('./data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_" + str(PCA_COMPONENTS) + "PCA_data.p", 'rb') as infile:
    data = pickle.load(infile)
with open('./data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_labels.p", 'rb') as infile:
    labels = pickle.load(infile)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, train_size = 0.9, stratify=labels)

print("Initializing SOM")
dnmsom = DNM_SOM(
    x             = MSIZE[0], 
    y             = MSIZE[1], 
    input_len     = data[0].shape[0],
    sigma         = NRADIUS,
    learning_rate = LEARN_RATE,
    random_seed   = None)

print("Initializing Weights")
dnmsom.random_weights_init(X_train)

# Train SOM
print("Training SOM")
dnmsom.train_random(X_train, ITERATIONS, verbose = True)

# Classification
print(classification_report(y_test, classifiers.classify_BMU(dnmsom, X_test, X_train, y_train)))


print("Initializing SOM")
som = MiniSom(
    x             = MSIZE[0], 
    y             = MSIZE[1], 
    input_len     = data[0].shape[0],
    sigma         = NRADIUS,
    learning_rate = LEARN_RATE,
    random_seed   = None)

print("Initializing Weights")
som.random_weights_init(X_train)

# Train SOM
print("Training SOM")
som.train_random(X_train, ITERATIONS, verbose = True)

# Classification
print(classification_report(y_test, classifiers.classify_BMU(som, X_test, X_train, y_train)))

som.win_map
# Save SOM
if not os.path.exists('./som_files/' + SOM_NAME):
    os.makedirs('./som_files/' + SOM_NAME)
with open('./som_files/' + SOM_NAME + "/som.p", 'wb') as outfile:
    pickle.dump(som, outfile)
with open('./som_files/' + SOM_NAME + "/dnm_som.p", 'wb') as outfile:
    pickle.dump(dnmsom, outfile)








