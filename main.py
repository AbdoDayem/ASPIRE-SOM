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
SOM_NAME = "test1"
PCA_COMPONENTS = 80
AUDIO_TIME = 1 #sec

## File Import
audio_files = glob.glob("./soundfiles/" + DATASET + "/**/*.wav")

## Data Processing
print("Processing Data")
input_len = len(audio_files)
data = [None] * input_len
labels = [None] * input_len
for i in tqdm(range(input_len)):
    samples, Fs = librosa.load(audio_files[i], sr = 16000)
    labels[i] = os.path.basename(os.path.dirname(audio_files[i]))
    data[i] = librosa.feature.mfcc(y = samples[0 : Fs * AUDIO_TIME], sr = Fs).flatten()

print(np.array(data).shape)

print("Splitting Data")
if PCA_COMPONENTS != 0:
    pca = PCA(n_components = PCA_COMPONENTS)
    data = pca.fit_transform(data)

print(np.array(data).shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, train_size = 0.9, stratify=labels)

## SOM
# Init SOM
print("Initializing SOM")
som = MiniSom(
    x             = MSIZE[0], 
    y             = MSIZE[1], 
    input_len     = data[0].shape[0],
    sigma         = NRADIUS,
    learning_rate = LEARN_RATE,
    random_seed   = None)

som.pca_weights_init(X_train)

# Train SOM
print("Training SOM")
som.train_random(X_train, ITERATIONS, verbose = True)

# Classification
print(classification_report(y_test, classifiers.classify_BMU(som, X_test, X_train, y_train)))

# Save SOM
if not os.path.exists('./som_files/' + SOM_NAME):
    os.makedirs('./som_files/' + SOM_NAME)
with open('./som_files/' + SOM_NAME + "/som.p", 'wb') as outfile:
    pickle.dump(som, outfile)
