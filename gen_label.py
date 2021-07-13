import os
import numpy as np
import pandas as pd
from scipy import io

import librosa


def load_label(label_path):
    df = pd.read_csv(label_path, names=['START', 'STATE'])
    starts = np.asarray(df['START']) * 2000
    starts = starts.astype(int)
    states = np.asarray(df['STATE'], dtype=str)

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

    return s1, s2, sys, dia, noise

def gen_label(label_path_, wav_len):
    s1, s2, sys, dia, noise = split_label(label_path_)
    labels = np.zeros(wav_len)
    # s1: 0, s2: 2, sys: 1, dia: 3, noise: 4
    for duration in s1:
        end = duration[1] // 2 if len(duration) == 2 else -1
        labels[duration[0 // 2]: end] = 0
    for duration in s2:
        end = duration[1] // 2 if len(duration) == 2 else -1
        labels[duration[0] // 2: end] = 2
    for duration in sys:
        end = duration[1] // 2 if len(duration) == 2 else -1
        labels[duration[0] // 2: end] = 1
    for duration in dia:
        end = duration[1] // 2 if len(duration) == 2 else -1
        labels[duration[0] // 2: end] = 3
    for duration in noise:
        end = duration[1] // 2 if len(duration) == 2 else -1
        labels[duration[0] // 2: end] = 4

    return labels

if __name__ == '__main__':
    DATA_PATH = '/data/test_data_base/recording/'
    ANN_PATH = '/data/test_data_base/annotation/'
    SAVE_PATH = '/data/test_data_base/labels/'

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for file in os.listdir(DATA_PATH):
        if file.endswith('wav'):
            y, _ = librosa.load(os.path.join(DATA_PATH, file), sr=1000)
            length = len(y)
            index = file.split('.')[0]
            print(index)

            if not os.path.exists(os.path.join(ANN_PATH, index+'.csv')):
                continue
            labels = gen_label(os.path.join(ANN_PATH, index+'.csv'), length)
            np.save(os.path.join(SAVE_PATH, index+'.npy'), labels)
