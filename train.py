import numpy as np
import os

import tensorflow as tf
import random
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from segnet import SegNet
from prepare_dataset import SigDataset


def train(model, model_path, train_size, dataset, sig_len_,
          frame_len_, batch_size_, num_class_, epoches):
    train_iters = train_size // batch_size_
    best_val_acc = 0

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            model.restore(sess, ckpt.model_checkpoint_path)
        summary_writer = tf.summary.FileWriter(logdir=LOG_PATH, graph=sess.graph)

        for epoch in range(epoches):
            for step in range((epoch*train_iters), (epoch+1)*train_iters):
                batch_x, batch_y = dataset.inputs(is_training=True)
                #batch_y = np.reshape(batch_y, (batch_size_ * (sig_len_ // frame_len_), num_class_))
                summary_op = tf.summary.merge_all()
                _, loss, acc = sess.run([model.train_op, model.loss, model.accuracy],
                                        feed_dict={model.inputs: batch_x,
                                                   model.labels: batch_y,
                                                   model.training: True})
                if step%20 == 0:
                    logging.info('epoch {:}/{:}, step= {:}/{:}, loss={:.4f}, acc={:.4f}'.format(epoch, epoches, step-(epoch*train_iters), train_iters, loss, acc))

            test_x, test_y = dataset.inputs(is_training=False)
            acc_list = []
            loss_list = []
            for x, y in zip(test_x, test_y):
                x = np.reshape(x, (1, x.shape[0], x.shape[1]))
                y = np.expand_dims(y, 0)
                loss, acc = sess.run([model.loss, model.accuracy], feed_dict={model.inputs: x,
                                                                              model.labels: y,
                                                                              model.training: False})
                acc_list.append(acc)
                loss_list.append(loss)
            logging.info('len of test set : {}'.format(len(test_y)))
            logging.info('acc on test: {:}'.format(np.mean(acc_list)))
            logging.info('loss on test: {:}'.format(np.mean(loss_list)))

            val_acc = np.mean(acc_list)

            save_path = os.path.join(model_path, 'model' + '.ckpt')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save(sess, save_path)

        logging.info('training done.')

if __name__ == '__main__':
    CV_PATH = './split_data/'
    LABEL_PATH = '/data/train_label/'
    LOG_PATH = './model/log_cv1/'
    MODEL_PATH = './model/model_cv1/'
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
    dataset = SigDataset(batch_size=BATCH_SIZE, cv_path=CV_PATH, fold=FOLD, channels=FEATURE_C,
                         label_path=LABEL_PATH, num_classes=NUM_CLASSES, slice_len=SLICE_LEN)

    train(model, MODEL_PATH, TRAIN_SIZE, dataset, SIG_LEN, SLICE_LEN, BATCH_SIZE, NUM_CLASSES, EPOCHES)
