# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-num' , type=int)
args = parser.parse_args()
pro_num = args.num

def loadhdf5(path):
    raw_data = pd.HDFStore(path)
    df = raw_data['data']
    raw_data.close()
    return df



path = '/media/data1/wangjc/stock_prediciton/data/pro_data/data_all_%d.h5'%(pro_num)

df_train = loadhdf5(path)

df_train_len = df_train.shape[0]
df_label = []
tmp = df_train.LABELNEW.values
count_dict = Counter(tmp)
df_label = [count_dict[i] for i in range(1,6)]
min_label = min(df_label)
prob = [min_label*1.0/label for label in df_label]
df_train = np.array(df_train)





tfrecords_train_filename = '/media/data2/wangjc/data_%d.tfrecords'%(pro_num)
writer_train = tf.python_io.TFRecordWriter(tfrecords_train_filename)
#tfrecords_valid_filename = '/media/data1/wangjc/stock_prediciton/data/tfrecord/valid.tfrecords'
#writer_valid = tf.python_io.TFRecordWriter(tfrecords_valid_filename)
#tfrecords_test_filename = '/media/data1/wangjc/stock_prediciton/data/tfrecord/test.tfrecords'
#writer_test = tf.python_io.TFRecordWriter(tfrecords_test_filename)



                             #读入config文件





count_list = [0, 0, 0, 0, 0]
seq_length = 600
try:
    for i in range(seq_length, df_train_len, 20):
        if (df_train[i, 6]==df_train[i - seq_length, 6])and(df_train[i, 2]==df_train[i - seq_length, 2])and(df_train[i, 1]==df_train[i - seq_length, 1]):
            x = random.random()
            if (df_train[i, 40]==1 and x<=prob[0])or\
                (df_train[i, 40]==2 and x<=(prob[1]*0.9))or\
                (df_train[i, 40]==3 and x<=(prob[2]*0.8))or\
                (df_train[i, 40]==4 and x<=(prob[3]*0.9))or\
                (df_train[i, 40]==5 and x<=prob[4]):
                if df_train[i, 40]==1:
                    label = np.array([1, 0, 0, 0, 0]).astype(np.float32).tostring()
                elif df_train[i, 40]==2:
                    label = np.array([0, 1, 0, 0, 0]).astype(np.float32).tostring()
                elif df_train[i, 40]==3:
                    label = np.array([0, 0, 1, 0, 0]).astype(np.float32).tostring()
                elif df_train[i, 40]==4:
                    label = np.array([0, 0, 0, 1, 0]).astype(np.float32).tostring()
                else:
                    label = np.array([0, 0, 0, 0, 1]).astype(np.float32).tostring()
                feature = (df_train[i - seq_length:i, 0:40]).astype(np.float32).tostring()
                example = tf.train.Example(features=tf.train.Features(
                                feature={
                                'feature': tf.train.Feature(bytes_list = tf.train.BytesList(value=[feature])),
                                'label': tf.train.Feature(bytes_list = tf.train.BytesList(value=[label]))
                                    }))
                writer_train.write(example.SerializeToString())
                count_list[int(df_train[i, 40])-1]+=1
            if i%10000000==0:
                print('train_part :  %d is done!!!'%(i))
    print(count_list)
except KeyboardInterrupt:
    writer_train.close()
