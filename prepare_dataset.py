import os
import numpy as np
import random

import librosa
from sklearn import preprocessing

import heart_sound_preprocessing as hsp


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

# Normalization Function
def temporal_norm(input):
    x = preprocessing.minmax_scale(input)
    return x

# statistic number of labels and transform labels to the most label each frame.
def label_most(label, label_slice_len):
    label_sliced = []
    for i in range(len(label) // label_slice_len):
        slice = label[label_slice_len*i: label_slice_len*(i+1)]
        slice = slice.astype(np.int64)
        count = np.bincount(slice)
        appro_label = np.argmax(count)
        label_sliced.append(appro_label)
    label_sliced = np.asarray(label_sliced)

    return label_sliced

class SigDataset():
    def __init__(self, batch_size, cv_path, label_path, num_classes, slice_len, channels, fold=1):
        self.batch_size = batch_size
        self.cv_path = cv_path
        self.label_path = label_path
        self.num_classes = num_classes
        self.slice_len = slice_len
        self.fold = fold
        self.channels = channels

    def inputs(self, is_training=True):
        train_set = open(os.path.join(self.cv_path, 'train_cv'+str(self.fold)+'.txt'), 'r').read().splitlines()
        test_set = open(os.path.join(self.cv_path, 'test_cv'+str(self.fold)+'.txt'), 'r').read().splitlines()
        train_data_list = np.asarray(random.sample(train_set, self.batch_size))
        test_data_list = np.asarray(test_set)
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        if is_training:
            for sample in train_data_list:
                sample_name = sample.split('/')[-1][:5]
                index = sample.split('/')[-1].split('_')[-1][: -4]
                label_path = os.path.join(self.label_path, sample_name+'_label_'+index+'.npy')
                #sample = hsp.temporal_norm(np.load(sample))
                sample = np.load(sample)
                label = np.load(label_path)

                label_slice = label_most(label, self.slice_len)
                label_slice = convert_to_one_hot(label_slice, self.num_classes)

                if self.channels == 1:
                    train_data.append(np.reshape(sample, (sample.shape[0], channels)))
                else:
                    train_data.append(sample)
                train_labels.append(label_slice)

            train_data = np.asarray(train_data)
            train_labels = np.asarray(train_labels)

            return train_data, train_labels
        else:
            for sample in test_data_list:
                sample_name = sample.split('/')[-1][:5]
                index = sample.split('/')[-1].split('_')[-1][: -4]
                label_path = os.path.join(self.label_path, sample_name+'_label_'+index+'.npy')
                #sample = hsp.temporal_norm(np.load(sample))
                sample = np.load(sample)
                label = np.load(label_path)

                label_slice = label_most(label, self.slice_len)
                label_slice = convert_to_one_hot(label_slice, self.num_classes)

                if self.channels == 1:
                    test_data.append(np.reshape(sample, (sample.shape[0], channels)))
                else:
                    test_data.append(sample)
                test_labels.append(label_slice)

            test_data = np.asarray(test_data)
            test_labels = np.asarray(test_labels)

            return test_data, test_labels

def read_sample(sample_path_, label_path_, slice_len_, num_classes_):
    sample = np.load(sample_path_)
    label = np.load(label_path_)

    label_slice = label_most(label, slice_len_)
    label_slice = convert_to_one_hot(label_slice, num_classes_)

    return sample, label_slice

if __name__ == '__main__':
    CV_PATH = './split_data/'
    LABEL_PATH = '/data/training_label/'
    NUM_CLASSES = 4
    SLICE_LEN = 50
    CHANNELS = 3

    # Train
    dataset = SigDataset(batch_size=100, cv_path=CV_PATH, label_path=LABEL_PATH, num_classes=NUM_CLASSES, slice_len=SLICE_LEN, channels=CHANNELS)
    x, y = dataset.inputs(is_training=True)
    print(x.shape)
    print(y.shape)

