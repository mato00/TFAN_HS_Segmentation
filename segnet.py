import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.layers import flatten

from ops import *

def BiLstmCell(input_data, rnn_size, keep_prob, name):
    output = input_data
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

        cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                          cell_bw,
                                                          output,
                                                          dtype=tf.float32)
        output = tf.concat(outputs,2)

    return output

def encoder(sig, channels, name='encoder', training_=True):
    with tf.variable_scope(name):
        c = TempBlock(input_=sig, layer_index=0, ks=3, outchannels_=channels, dilation_rate_=2**0,
                      name='en_tb_{}'.format(0), is_training=training_)
        c = TempBlock(input_=c, layer_index=1, ks=3, outchannels_=channels, dilation_rate_=2**1,
                      name='en_tb_{}'.format(1), is_training=training_)
        c = TempBlock(input_=sig, layer_index=2, ks=3, outchannels_=channels, dilation_rate_=2**2,
                      name='en_tb_{}'.format(2), is_training=training_)
        c = TempBlock(input_=c, layer_index=3, ks=3, outchannels_=channels, dilation_rate_=2**3,
                      name='en_tb_{}'.format(3), is_training=training_)
        c = TempBlock(input_=c, layer_index=4, ks=3, outchannels_=channels, dilation_rate_=2**4,
                      name='en_tb_{}'.format(4), is_training=training_)
        c = TempBlock(input_=c, layer_index=5, ks=3, outchannels_=channels, dilation_rate_=2**5,
                      name='en_tb_{}'.format(5), is_training=training_)
        c = TempBlock(input_=c, layer_index=6, ks=3, outchannels_=channels, dilation_rate_=2**6,
                      name='en_tb_{}'.format(6), is_training=training_)
        c = TempBlock(input_=c, layer_index=7, ks=3, outchannels_=channels, dilation_rate_=2**7,
                      name='en_tb_{}'.format(7), is_training=training_)

        return c

def decoder(features, sig_size_, frame_length_, channels_, num_classes_,
            name='decoder', df_dim_=1, training_=True):
    with tf.variable_scope(name):
        frames = tf.reshape(features, (tf.shape(features)[0]*(sig_size_//frame_length_), frame_length_, channels_), name='l_tr0')
        frames = tf.expand_dims(frames, -1)
        #print(frames.shape)
        d = lrelu(batch_norm(conv2d(frames, df_dim_ * 16, ks=4, s=1, name='d_d0_conv'), name='d_bn0'))
        d = tf.nn.dropout(d, 0.8)

        d = lrelu(batch_norm(conv2d(d, df_dim_ * 32, ks=4, s=1, name='d_d1_conv'), name='d_bn1'))
        d = tf.nn.dropout(d, 0.8)

        d = lrelu(batch_norm(conv2d(d, df_dim_ * 64, ks=4, s=1, name='d_d2_conv'), name='d_bn2'))
        d = tf.nn.dropout(d, 0.8)

        d = flatten(d)
        d = tf.layers.dense(d, 64, activation='relu', kernel_initializer=tf.orthogonal_initializer(), name='d_d1')

        d = tf.reshape(d, (tf.shape(features)[0], (sig_size_ // frame_length_), 64), name='l_tr1')

        d = BiLstmCell(d, 64, 0.8, name='bilstm1')

        d = tf.layers.dense(d, 64, activation='relu', kernel_initializer=tf.orthogonal_initializer(), name='d_d2')

        d_o = tf.layers.dense(d, num_classes_, activation=None, kernel_initializer=tf.orthogonal_initializer(), name='d_logits')
        d_r = tf.reshape(d_o, (tf.shape(features)[0]*(sig_size_//frame_length_), num_classes_), name='l_tr2')


        return d_r, d_o

class SegNet(object):
    def __init__(self, model_name='SegNet', input_len=2000, width=32,
                 frame_length=50, lr=0.001, classes=4, feature_channel=4):
        self.model_name = model_name

        self.num_class = classes
        self.input_len = input_len
        self.width = width
        self.frame_length = frame_length
        self.feature_c = feature_channel

        self.inputs = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.input_len, self.feature_c],
                                           name='input')
        self.labels = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.input_len//self.frame_length, self.num_class])
        self.labels_c = tf.reshape(self.labels, (tf.shape(self.labels)[0] * self.input_len // self.frame_length, self.num_class))
        self.training = tf.placeholder(tf.bool)

        self.learning_rate = tf.Variable(lr, name='learning_rate')
        self.build_graph()
        self.prediction = tf.argmax(tf.nn.softmax(self.logits_c), 1)
        self.output = tf.nn.softmax(self.logits_c)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.labels_c, 1)), tf.float32))
        self.equal = tf.cast(tf.equal(self.prediction, tf.argmax(self.labels_c, 1)), tf.float32)

        self.loss = self.loss_function()
        self.train_op = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True).minimize(self.loss)

    def save(self, sess, save_path_):
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path_)

        return save_path

    def restore(self, sess, restore_path_):
        saver = tf.train.Saver()
        saver.restore(sess, restore_path_)
        print('Model restored from file: {}'.format(restore_path_))

    def build_graph(self):
        self._build_model()

    def loss_function(self):
        loss_c = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels_c, logits=self.logits_c)
        padding = tf.constant([[0, 0], [1, 0], [0, 0]])
        labels = tf.pad(self.labels, padding, mode='SYMMETRIC')
        labels_shift_l = labels[:, :-1, :]
        labels_shift_r = labels[:, 1: , :]
        diff_labels = (labels_shift_l + labels_shift_r) / 2
        diff_labels = tf.reshape(diff_labels, (tf.shape(diff_labels)[0]*tf.shape(diff_labels)[1], self.num_class))
        diff_weight = tf.count_nonzero(diff_labels, axis=-1, dtype=tf.float32)

        logits_d = tf.nn.softmax(self.logits_s, axis=-1)
        logits_d = tf.pad(logits_d, padding, mode='SYMMETRIC')
        logits_shift_l = logits_d[:, :-1, :]
        logits_shift_r = logits_d[:, 1: , :]
        diff_logits = (logits_shift_l + logits_shift_r) / 2
        diff_logits = tf.reshape(diff_logits, (tf.shape(diff_logits)[0]*tf.shape(diff_logits)[1], self.num_class))

        loss_st = tf.reduce_mean(-tf.reduce_sum(diff_labels*tf.log(diff_logits),
                                 reduction_indices=[1]))
        loss_st = tf.multiply(loss_st, diff_weight) # 交界处X2

        self.loss = tf.reduce_mean(tf.add(loss_st, 2*loss_c))

        return self.loss

    def _build_model(self):
        self.features = encoder(self.inputs, self.width, training_=self.training)
        self.logits_c, self.logits_s = decoder(self.features, self.input_len, self.frame_length,
                                               self.width, self.num_class, training_=self.training)

if __name__ == '__main__':
    NUM_CLASSES = 4
    SIG_LEN = 2000
    SLICE_LEN = 20
    BATCH_SIZE = 20
    EPOCHES = 200
    WIDTH = 16
    LR = 0.001
    FOLD = 1
    FEATURE_C = 4
    TRAIN_SIZE = 3200

    model = SegNet(input_len=SIG_LEN, width=WIDTH, frame_length=SLICE_LEN, lr=LR, classes=NUM_CLASSES, feature_channel=FEATURE_C)
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)
