import glob
import math
import os
from matplotlib import gridspec, pyplot as plt
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
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from collections import defaultdict
from itertools import count
from functools import partial

torch.device("cuda")

SOM_NAME = "NN_SOM"
MSIZE = [16, 16]
NRADIUS = 0.1
SOM_EPOCHS = 5000
LEARN_RATE = 0.5
DATASET = "SpeakerBN"
PCA_COMPONENTS = 80
AUDIO_TIME = 0.99 #sec
NN_EPOCHS = 100
NN_BATCH = 20
CLASSES = 8

with open('./data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_" + str(PCA_COMPONENTS) + "PCA_data.p", 'rb') as infile:
#with open('./data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_" + "data.p", 'rb') as infile:
    data = pickle.load(infile)
with open('./data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_labels.p", 'rb') as infile:
    labels = pickle.load(infile)

label_to_number = defaultdict(partial(next, count(1)))
y = np.zeros((len(labels), len(np.unique(labels))))
for i in range(len(labels)):
    y[i][label_to_number[labels[i]] - 1] = 1
 
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.2, train_size = 0.8, stratify=y)
X_train = torch.tensor(X_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32)
X_test = torch.tensor(X_test, dtype = torch.float32)
y_test = torch.tensor(y_test, dtype = torch.float32)

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(X_train.shape[1], 64)
        self.a1 = nn.ReLU()
        self.h2 = nn.Linear(64, 32)
        self.a2 = nn.ReLU()
        self.out = nn.Linear(32, CLASSES)
        self.a_out = nn.Sigmoid()
    def forward(self, x):
        x = self.a1(self.h1(x))
        x = self.a2(self.h2(x))
        x_som = x
        x = self.a_out(self.out(x))
        return x, x_som
model = DNN()

print(model)

loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
 
print("Training DNN")
epoch_x = []
loss_y = []
acc_y = []
for epoch in range(NN_EPOCHS):
    epoch_x.append(epoch)
    for i in range(0, len(X_train), NN_BATCH):
        Xbatch = X_train[i:i+NN_BATCH]
        y_pred, _ = model(Xbatch)
        ybatch = y_train[i:i+NN_BATCH]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        nn_y_pred, _ = model(X_train)
    acc_y.append((nn_y_pred.round() == y_train).float().mean())
    loss_y.append(loss)
    if epoch % 10 == 0: print(f'Epoch: {epoch}| Loss: {loss}')

with torch.no_grad():
    nn_y_pred, SOM_X_train = model(X_train)
    nn_y_pred_test, SOM_X_test = model(X_test)

accuracy = (nn_y_pred_test.round() == y_test).float().mean()
print(f"NN Test Accuracy: {accuracy}")

print("Initializing SOM")
dnn_som = MiniSom(
    x             = MSIZE[0], 
    y             = MSIZE[1], 
    input_len     = SOM_X_train[0].shape[0],
    sigma         = NRADIUS,
    learning_rate = LEARN_RATE,
    random_seed   = None)

som = MiniSom(
    x             = MSIZE[0], 
    y             = MSIZE[1], 
    input_len     = X_train[0].shape[0],
    sigma         = NRADIUS,
    learning_rate = LEARN_RATE,
    random_seed   = None)


print("Initializing Weights")
dnn_som.random_weights_init(SOM_X_train)
som.random_weights_init(X_train)

# Train SOM
print("Training SOM")
dnn_som.train_random(SOM_X_train, SOM_EPOCHS, verbose = True)
som.train_random(X_train, SOM_EPOCHS, verbose = True)

# Save SOM
if 0:
    if not os.path.exists('./som_files/' + SOM_NAME):
        os.makedirs('./som_files/' + SOM_NAME)
    with open('./som_files/' + SOM_NAME + "/som.p", 'wb') as outfile:
        pickle.dump(dnn_som, outfile)

# Classification
dnn_som_y_pred = torch.stack(classifiers.classify_BMU(dnn_som, SOM_X_test, SOM_X_train, y_train), dim = 0)
print(classification_report(y_test, dnn_som_y_pred))
accuracy = (dnn_som_y_pred == y_test).float().mean()
print(f"DNN-SOM Test Accuracy: {accuracy}")

som_y_pred = torch.stack(classifiers.classify_BMU(som, X_test, X_train, y_train), dim = 0)
print(classification_report(y_test, som_y_pred))
accuracy = (som_y_pred == y_test).float().mean()
print(f"SOM Test Accuracy: {accuracy}")

## Visu
plt.figure(0)
loss_y = [i.tolist() for i in loss_y]
plt.plot(epoch_x, loss_y)
plt.title("NN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.figure(1)
plt.plot(epoch_x, acc_y)
plt.title("NN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Acccuracy")
plt.show()

if 0:
    plt.figure(0)
    label_names = np.identity(len(np.unique(labels))).tolist()
    label_names = [torch.tensor(i) for i in label_names]
    labels_map = dnn_som.labels_map(SOM_X_train, y_train)

    fig = plt.figure(figsize=(9, 9))
    the_grid = gridspec.GridSpec(MSIZE[0], MSIZE[1], fig)
    for position in labels_map.keys():
        label_fracs = [labels_map[position][l] for l in label_names]
        plt.subplot(the_grid[MSIZE[0]-1-position[0], position[1]], aspect=1)
        #plt.subplot(the_grid[som.x-1-position[1], position[0]], aspect=1)
        if 1 in label_fracs: patches, texts = plt.pie(label_fracs)

    #plt.legend(patches, label_names)
    plt.show()


