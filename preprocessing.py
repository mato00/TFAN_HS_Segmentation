import os
import numpy as np
import pandas as pd
import random

import librosa

import heart_sound_preprocessing as hsp


def load_label(label_path):
    df = pd.read_csv(label_path, names=['START', 'STATE'])
    starts = np.asarray(df['START'])[1: ] * 2000
    starts = starts.astype(int)
    states = np.asarray(df['STATE'], dtype=str)[1: ]

    return starts, states

def split_label(label_path):
    starts, states = load_label(label_path)
    s1 = []
    s2 = []
    sys = []
    dia = []
    noise = []
    count = 0
    for (start, state) in zip(starts, states):
        index = np.array([start, starts[count+1]]) if count < len(starts)-1 else np.array([start])

        if state == 'S1':
            s1.append(index)
        elif state == 'S2':
            s2.append(index)
        elif state == 'systole':
            sys.append(index)
        elif state == 'diastole':
            dia.append(index)
        elif state == '(N':
            noise.append(index)

        count += 1

    return s1, s2, sys, dia, noise, starts

def split_frame(sample, label, init, framesize, _sample_name, sample_save_path, label_save_path):
    """
    locate the initial start point, slice the samples and labels into frames with same
    frame size. cause the signal is downsampled from 2000Hz into 1000Hz, so each start
    index is divided by 2.
    """
    length = len(label)
    hop_size = framesize // 2
    count = 0

    for i in range(0, (length//hop_size)-1):
        data_frame = sample[i*hop_size: (i+2)*hop_size]
        #data_frame = hsp.remove_outliers(data_frame, t=4)
        label_frame = label[i*hop_size: (i+2)*hop_size]
        features = np.zeros((len(data_frame), 3), dtype=np.float32)

        # Add Wiener Filter
        features[:, 0] = hsp.local_wiener(data_frame, mysize=21, local_size=50, fs=1000) # Wiener Filter
        features[:, 1] = hsp.butter_bandpass_filter(features[:, 0], lowcut=30, highcut=60, fs=1000, order=5) # Bandpass Filter
        features[:, 2] = hsp.wavelet_envelop(features[:, 0], 'rbio3.9', level_=6, keep_level_=3) # Wavelet Filter
        '''
        # No Wiener Filter
        features[:, 0] = data_frame # Wiener Filter
        features[:, 1] = hsp.butter_bandpass_filter(data_frame, lowcut=30, highcut=60, fs=1000, order=5) # Bandpass Filter
        features[:, 2] = hsp.wavelet_envelop(data_frame, 'rbio3.9', level_=6, keep_level_=3) # Wavelet Filter
        '''

        augment_frame = hsp.temporal_norm(features)
        np.save(os.path.join(sample_save_path, _sample_name+'_sample_'+str(count)+'.npy'), augment_frame)
        np.save(os.path.join(label_save_path, _sample_name+'_label_'+str(count)+'.npy'), label_frame)
        count += 1

    if (len(sample) - count * hop_size) >= framesize:
        data_frame = sample[len(sample) - framesize: ]
        label_frame = label[i*hop_size: (i+2)*hop_size]
        features = np.zeros((len(data_frame), 3), dtype=np.float32)

        # Add Wiener Filter
        features[:, 0] = hsp.local_wiener(data_frame, mysize=51, local_size=100, fs=1000) # Wiener Filter
        features[:, 1] = hsp.butter_bandpass_filter(features[:, 0], lowcut=30, highcut=60, fs=1000, order=5) # Bandpass Filter
        features[:, 2] = hsp.wavelet_envelop(features[:, 0], 'rbio3.9', level_=6, keep_level_=3) # Wavelet Filter
        '''
        # No Wiener Filter
        features[:, 0] = data_frame # Wiener Filter
        features[:, 1] = hsp.butter_bandpass_filter(data_frame, lowcut=30, highcut=60, fs=1000, order=5) # Bandpass Filter
        features[:, 2] = hsp.wavelet_envelop(data_frame, 'rbio3.9', level_=6, keep_level_=3) # Wavelet Filter
        '''

        augment_frame = hsp.temporal_norm(features)
        np.save(os.path.join(sample_save_path, _sample_name+'_sample_'+str(count)+'.npy'), augment_frame)
        np.save(os.path.join(label_save_path, _sample_name+'_label_'+str(count)+'.npy'), label_frame)
        count += 1

def feat_extract(label_path, wav_path, sample_save_path, label_save_path, framesize):
    for label_json in os.listdir(label_path):
        s1, s2, sys, dia, noise, starts = split_label(os.path.join(label_path, label_json))
        sample_name = label_json.split('.')[0].split('_')[0]
        wav_file = sample_name + '.wav'
        sample_path = os.path.join(wav_path, wav_file)
        print(sample_name)
        ## Read signal data
        y, sr = librosa.load(sample_path, sr=None)
        y_b = hsp.butter_bandpass_filter(y, lowcut=15, highcut=800, fs=sr, order=5)
        y_b = hsp.downsample(y_b, sr, 1000))

        zeros = np.zeros_like(y_b)
        labels = zeros.copy()
        # s1: 0, s2: 2, sys: 1, dia: 3, noise: 4
        for duration in s1:
            end = duration[1]//2 if len(duration) == 2 else -1
            labels[duration[0]//2: end] = 0
        for duration in s2:
            end = duration[1]//2 if len(duration) == 2 else -1
            labels[duration[0]//2: end] = 2
        for duration in sys:
            end = duration[1]//2 if len(duration) == 2 else -1
            labels[duration[0]//2: end] = 1
        for duration in dia:
            end = duration[1]//2 if len(duration) == 2 else -1
            labels[duration[0]//2: end] = 3
        for duration in noise:
            end = duration[1]//2 if len(duration) == 2 else -1
            labels[duration[0]//2: end] = 4
        delete_index = np.argwhere(labels == 4)
        y_b = np.delete(y_b, delete_index, axis=0)
        labels = np.delete(labels, delete_index, axis=0)

        split_frame(y_b, labels, starts, framesize, sample_name, sample_save_path, label_save_path)

    print('Split is done.')


if __name__ == '__main__':
    WAV_PATH = '/data/recording/training-a/'
    LABEL_PATH = '/data/annotation/training-a/'
    FEATURE_SAVE_PATH = '/data/training/train_feat/'
    LABEL_SAVE_PATH = '/data/training/train_label/'

    FRAMESIZE = 2000

    if not os.path.exists(LABEL_SAVE_PATH):
        os.makedirs(LABEL_SAVE_PATH)
    if not os.path.exists(FEATURE_SAVE_PATH):
        os.makedirs(FEATURE_SAVE_PATH)

    feat_extract(label_path=LABEL_PATH, wav_path=WAV_PATH, sample_save_path=FEATURE_SAVE_PATH, label_save_path=LABEL_SAVE_PATH, framesize=FRAMESIZE)
