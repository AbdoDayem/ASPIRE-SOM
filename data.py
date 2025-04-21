import glob
import os
import numpy as np
from tqdm import tqdm
import pickle
import librosa
from sklearn.decomposition import PCA

DATASET = "16000_pcm_speeches"
PCA_COMPONENTS = 80
AUDIO_TIME = 0.99 #sec


def mfcc_arr(dataset, audio_time):
    audio_files = glob.glob("./soundfiles/" + dataset + "/**/*.wav")
    ## Data Processing
    print("Processing Data")
    input_len = len(audio_files)
    data = []
    labels = []
    for i in tqdm(range(input_len)):
        samples, Fs = librosa.load(audio_files[i], sr = 16000)
        file_label = os.path.basename(os.path.dirname(audio_files[i]))
        sample_len = int((len(samples)/Fs)/AUDIO_TIME)
        for j in range(sample_len):
            if (audio_time*(j+1)) < (len(samples)/Fs):
                labels.append(os.path.basename(file_label))
                data.append(librosa.feature.mfcc(y = samples[int(Fs * j * audio_time): int(Fs * (j+1) * audio_time)], sr = Fs).flatten())

    print(np.array(data).shape)
    if not os.path.exists('./data_files/' + dataset):
        os.makedirs('./data_files/' + dataset)
    with open('./data_files/' + dataset + "/" + str(audio_time) + "sec_data.p", 'wb') as outfile:
        pickle.dump(data, outfile)
    with open('./data_files/' + dataset + "/" + str(audio_time) + "sec_labels.p", 'wb') as outfile:
        pickle.dump(labels, outfile)

def pca_matrix(dataset, audio_time, components):
    # MUST HAVE A STORED PROCESSED DATA FILE FIRST
    # ^ USE mfcc_arr

    pca = PCA(n_components = components)
    with open('./data_files/' + dataset + "/" + str(audio_time) + "sec_" + "data.p", 'rb') as infile:
        data = pickle.load(infile)
    data = pca.fit_transform(data)
    with open('./data_files/' + DATASET + "/" + str(AUDIO_TIME) + "sec_" + str(PCA_COMPONENTS) + "PCA_data.p", 'wb') as outfile:
        pickle.dump(data, outfile)
