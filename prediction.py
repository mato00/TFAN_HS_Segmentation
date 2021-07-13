import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import librosa
import tensorflow as tf

from prepare_dataset import read_sample
from segnet import SegNet
import heart_sound_preprocessing as hsp
import math

def correct(preds):
    V = np.zeros_like(preds)
    Idx = np.zeros_like(preds)

    eps = 10**-10
    lnP = np.log(preds + eps)
    #lnP = preds
    V[0] = lnP[0]
    next = np.array([1, 2, 3, 0])
    prev = np.array([3, 0, 1, 2])
    for i in range(1, preds.shape[0]):
        for j in range(preds.shape[1]):
            #s = V[i-1, j] + lnP[i, j]
            #p = V[i-1, prev[j]] + max(lnP[i, j], lnP[i, (j+2)%4])
            s = V[i-1, j] + lnP[i, j]
            p = V[i-1, prev[j]] + lnP[i, j]

            if s > p:
                V[i, j] = s
                Idx[i, j] = j
            else:
                V[i, j] = p
                Idx[i, j] = prev[j]

    opt_path = np.zeros((preds.shape[0], ))
    opt_path[-1] = np.argmax(V[-1, :])
    for i in range(preds.shape[0]-2, 0, -1):
        opt_path[i] = Idx[i+1, int(opt_path[i+1])]

    return opt_path

def feat_extract(data_frame):
    features = np.zeros((len(data_frame), 3), dtype=np.float32)
    
    # Add Wiener Filter
    feat_0 = hsp.local_wiener(data_frame, mysize=51, local_size=100, fs=1000) # Wiener Filter
    features[:, 0] = feat_0
    features[:, 1] = hsp.butter_bandpass_filter(features[:, 0], lowcut=30, highcut=60, fs=1000, order=5) # Bandpass Filter
    features[:, 2] = hsp.wavelet_envelop(features[:, 0], 'rbio3.9', level_=6, keep_level_=3) # Wavelet Filter
    # # No Wiener Filter
    # features[:, 0] = data_frame # Wiener Filter
    # features[:, 1] = hsp.butter_bandpass_filter(data_frame, lowcut=30, highcut=60, fs=1000, order=5) # Bandpass Filter
    # features[:, 2] = hsp.wavelet_envelop(data_frame, 'rbio3.9', level_=6, keep_level_=3) # Wavelet Filter

    augment_frame = hsp.temporal_norm(features)

    return augment_frame

def test_records(model, model_path, data_path, label_path, save_path, sig_len, frame_size, channels):
    def is_wav(f):
        return f.endswith('.wav')
    '''
    def is_wav(f):
        return f.endswith('.npy') # 792 data
    '''
    records = list(filter(is_wav, os.listdir(data_path)))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            model.restore(sess, ckpt.model_checkpoint_path)

        for record in records:
            index = record.split('.')[0]
            print(index)
            record_path = os.path.join(data_path, record)
            y, sr = librosa.load(record_path, sr=None)
            y_b = hsp.butter_bandpass_filter(y, lowcut=15, highcut=800, fs=sr, order=5)
            y_b = hsp.downsample(y_b, sr, 1000)
            if len(y_b) < 2000:
                continue

            hop_size = sig_len // 2
            record_pred = []
            count = math.floor(len(y_b) / hop_size)
            for i in range(count - 1):
                period = y_b[hop_size * i: hop_size * (i+2)]
                test_x = feat_extract(period)

                test_x = np.reshape(test_x, (1, test_x.shape[0], channels))
                rec_preds = sess.run(model.output, feed_dict={model.inputs: test_x, model.training: False})

                if i == 0:
                    record_pred.extend(rec_preds)
                else:
                    record_pred.extend(rec_preds[hop_size // frame_size: ])

            if (len(y_b) - count * hop_size) >= frame_size:
                period = y_b[len(y_b) - sig_len: ]
                test_x = feat_extract(period)

                test_x = np.reshape(test_x, (1, test_x.shape[0], channels))
                rec_preds = sess.run(model.output, feed_dict={model.inputs: test_x, model.training: False})

                record_pred.extend(rec_preds[-1*(len(y_b) // frame_size - count * hop_size // frame_size): ])

            labels = np.load(os.path.join(label_path, index+'.npy'))

            record_pred = np.array(record_pred)
            record_pred = correct(record_pred)
            record_pred = np.array([[i] * 20 for i in record_pred]).flatten()

            np.save(os.path.join(save_path, index+'.npy'), record_pred)

        sess.close()

def test_record(model, model_path, data_path, label_path, index, save_path, sig_len, frame_size, channels):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess: 
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            model.restore(sess, ckpt.model_checkpoint_path)

        record_path = os.path.join(data_path, index+'.wav')
        y, sr = librosa.load(record_path, sr=None)
        y_b = hsp.butter_bandpass_filter(y, lowcut=15, highcut=800, fs=sr, order=5)
        y_b = hsp.downsample(y_b, sr, 1000)
        y_b_norm = hsp.temporal_norm(y_b)

        hop_size = sig_len // 2
        record_pred = []
        count = math.floor(len(y_b) / hop_size)

        for i in range(count - 1):
            period = y_b[hop_size * i: hop_size * (i+2)]
            test_x = feat_extract(period)
            test_x = np.reshape(test_x, (1, test_x.shape[0], channels))

            rec_preds = sess.run(model.output, feed_dict={model.inputs: test_x, model.training: False})

            if i == 0:
                record_pred.extend(rec_preds)
            else:
                record_pred.extend(rec_preds[hop_size // frame_size: ])

        if (len(y_b) - count * hop_size) >= frame_size:
            period = y_b[len(y_b) - sig_len: ]
            test_x = feat_extract(period)
            test_x = np.reshape(test_x, (1, test_x.shape[0], channels))

            rec_preds = sess.run(model.output, feed_dict={model.inputs: test_x, model.training: False})

            record_pred.extend(rec_preds[-1*(len(y_b) // frame_size - count * hop_size // frame_size): ])

        labels = np.load(os.path.join(label_path, index+'.npy'))

        record_pred = np.array(record_pred)        
        record_pred = correct(record_pred)
        record_pred = np.array([[i] * 20 for i in record_pred]).flatten()
        np.save('./result.npy', record_pred)

        sess.close()

if __name__ == '__main__':
    DATA_PATH = '/data/test_data_base/recording/'
    SAVE_PATH = '/data/test_data_base/prediction/proposed/'
    LABEL_PATH = '/data/test_data_base/labels/'
    SIG_LEN = 2000
    WIDTH = 16
    LR = 0.001
    FEATURE_C = 3
    SLICE_LEN = 20
    NUM_CLASS = 4
    MODEL_PATH = './model/model_cv1/'

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    model = SegNet(input_len=SIG_LEN, width=WIDTH, frame_length=SLICE_LEN, lr=LR, classes=NUM_CLASS, feature_channel=FEATURE_C)

    test_records(model, MODEL_PATH, DATA_PATH, LABEL_PATH, SAVE_PATH, SIG_LEN, SLICE_LEN, FEATURE_C)
    # test_record(model, MODEL_PATH, DATA_PATH, LABEL_PATH, 'a0001', SAVE_PATH, SIG_LEN, SLICE_LEN, FEATURE_C)
