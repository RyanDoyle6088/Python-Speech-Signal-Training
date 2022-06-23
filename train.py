import numpy as np
from scipy.io.wavfile import read
from mel_coefficients import mfcc
import matplotlib.pyplot as plt
import os
from scipy import signal


def train(filterbank):
    n_speaker = 8
    n_centroid = 16
    codebooks_mfcc = np.empty((n_speaker, filterbank, n_centroid))
    directory = os.path.dirname(os.getcwd()) + '/train'
    filename = str()
    for i in range(n_speaker):
        filename = '/s' + str(i+1) + '.wav'
        print('Current training file is ', str(i+1))
        (fs, s) = read(directory + filename)
        mel_coefs = mfcc(s, fs, filterbank)
        codebooks_mfcc[i, :, :] = lbg(mel_coefs, n_centroid)
        for j in range(n_centroid):
            plt.figure(i)
            plt.title('VQ Codebook for speaker ' + str(i + 1))
            markerline, stemlines, baseline = plt.stem(codebooks_mfcc[i, :, j])
            plt.ylabel('MFCC')
            plt.axis(ymin=-3, ymax=3)
            plt.xlabel('Number of features')
            plt.setp(markerline, 'markerfacecolor', 'r')
            plt.setp(baseline, 'color', 'k')
    plt.show()
    # training completed here

    # plotting 5th and 6th dimension MFCC features on a 2D plane
    codebooks = np.empty((2, filterbank, n_centroid))
    mel_coefs = np.empty((2, filterbank, 68))
    for i in range(2):
        filename = '/s' + str(i+2) + '.wav'
        (fs, s) = read(directory + filename)
        mel_coefs[i, :, :] = mfcc(s, fs, filterbank)[:, 0:68]
        codebooks[i, :, :] = lbg(mel_coefs[i, :, :], n_centroid)
    plt.figure(n_speaker + 1)
    s1 = plt.scatter(mel_coefs[0, 6, :], mel_coefs[0, 4, :], s=100,  color='r', marker='o')
    c1 = plt.scatter(codebooks[0, 6, :], codebooks[0, 4, :], s=100, color='r', marker='+')
    s2 = plt.scatter(mel_coefs[1, 6, :], mel_coefs[1, 4, :], s=100,  color='b', marker='o')
    c2 = plt.scatter(codebooks[1, 6, :], codebooks[1, 4, :], s=100, color='b', marker='+')
    plt.grid()
    plt.legend((s1, s2, c1, c2), ('Sp1', 'Sp2', 'Sp1 centroids', 'Sp2 centroids'), scatterpoints=1, loc='upper left')    
    plt.show()
   
    return codebooks_mfcc


def eudistance(d, c):
    n = np.shape(d)[1]
    p = np.shape(c)[1]
    distance = np.empty((n, p))

    if n < p:
        for i in range(n):
            copies = np.transpose(np.tile(d[:, i], (p, 1)))
            distance[i, :] = np.sum((copies - c) ** 2, 0)
    else:
        for i in range(p):
            copies = np.transpose(np.tile(c[:, i], (n, 1)))
            distance[:, i] = np.transpose(np.sum((d - copies) ** 2, 0))
    distance = np.sqrt(distance)
    return distance


def lbg(features, M):
    eps = 0.01
    codebook = np.mean(features, 1)
    distortion = 1
    n_centroid = 1
    while n_centroid < M:
        # double the size of codebook
        new_codebook = np.empty((len(codebook), n_centroid * 2))
        if n_centroid == 1:
            new_codebook[:, 0] = codebook * (1 + eps)
            new_codebook[:, 1] = codebook * (1 - eps)
        else:
            for i in range(n_centroid):
                new_codebook[:, 2 * i] = codebook[:, i] * (1 + eps)
                new_codebook[:, 2 * i + 1] = codebook[:, i] * (1 - eps)
        codebook = new_codebook
        n_centroid = np.shape(codebook)[1]
        D = eudistance(features, codebook)
        while np.abs(distortion) > eps:
            # nearest neighbour search
            prev_distance = np.mean(D)
            nearest_codebook = np.argmin(D, axis=1)
            # cluster vectors and find new centroid
            for i in range(n_centroid):
                codebook[:, i] = np.mean(features[:, np.where(nearest_codebook == i)], 2).T  # add along 3rd dimension
            codebook = np.nan_to_num(codebook)
            D = eudistance(features, codebook)
            distortion = (prev_distance - np.mean(D)) / prev_distance
    return codebook
