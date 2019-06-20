# -*- coding: utf-8 -*-
import tensorflow as tf




class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_path = self.config.train_path
        self.valid_path = self.config.valid_path
        self.test_path = self.config.test_path
        self.inputX_train, self.label_train = loaddata(self.config, self.train_path, 1)
        self.inputX_valid, self.label_valid = loaddata(self.config, self.valid_path, 10)
        self.inputX_test,  self.label_test  = loadtestdata(self.config, self.test_path, 4)



def loaddata(config, data_path, num):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(parser_function)
    dataset = dataset.shuffle(config.shuffle_num)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.repeat(config.step_num * num)
    iterator = dataset.make_one_shot_iterator()
    inputX, label  = iterator.get_next()
    return inputX, label


def loadtestdata(config, data_path, num):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(parser_function)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    inputX, label  = iterator.get_next()
    return inputX, label




def parser_function(serialized_example):
    features = tf.parse_single_example(serialized_example,
    features={
        'feature': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string)
        })
    label = tf.reshape(tf.decode_raw(features['label'], tf.float32), [5])
    feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32), [600, 40, 1])
    return feature, label
    


def thread_load(path, epoch, batch_size):
    files = tf.train.match_filenames_once(path)
    filename_queue = tf.train.string_input_producer(
        files, num_epochs= epoch)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={
            'feature': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
            })
    label = tf.reshape(tf.decode_raw(features['label'], tf.float32), [5])
    feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32), [600, 40, 1])
    feature_batch, label_batch = tf.train.batch(
        [feature, label], batch_size = batch_size, num_threads=64, capacity=4000)
    feature_batch = tf.cast(feature_batch, tf.float32)    
    label_batch = tf.cast(label_batch, tf.float32)  
    return feature_batch, label_batch




def normalize(inputX):
    return tf.div((inputX - inputX[0,:]), inputX[0,:])




