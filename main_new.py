# -*- coding: utf-8 -*-
import tensorflow as tf
from config.config import Config
from model.model_test3 import *
import time
from tools.tools import train_print
import os
import time
from dataset import Dataset
import argparse
import numpy as np


def thread_load(path, epoch, batch_size):
    files = tf.train.match_filenames_once(path)
    filename_queue = tf.train.string_input_producer(
        files, num_epochs= None)
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
        [feature, label], batch_size = batch_size, num_threads=64, capacity=batch_size * 3 + 1000)
    return feature_batch, label_batch


def find_suffix(model_dir, suffix):
    Files = os.listdir(model_dir)
    Files = sorted(Files, reverse = False)
    Files = sorted(Files,  
    key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
    Files.reverse()
    for k in range(len(Files)):
        if os.path.splitext(Files[k])[1]=='.' + suffix:
            suffix_dir = os.path.splitext(Files[k])
            suffix_dir = suffix_dir[0]+suffix_dir[1]
            break
    return suffix_dir


def gpu_config(config):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu        #设置GPU使用参数
    tf_config = tf.ConfigProto()                           #设置GPU用量
    if type(config.per_process_gpu_memory_fraction) is not float:#如果config.per_process_gpu_memory_fraction不是float
        tf_config.gpu_options.allow_growth = True               #则使用动态GPU用量
    else:
        tf_config.gpu_options.per_process_gpu_memory_fraction = config.per_process_gpu_memory_fraction
    return tf_config


def check_model_exist(config):
    model_name = config.model_name                          #获取模型名称，并创建路径model_dir以保存模型
    model_dir = os.path.join('./model_saved', model_name)
    isExist = os.path.exists(model_dir)
    if not isExist:
        os.makedirs(model_dir)
    return model_name, model_dir


def train_new():
    config = Config()                                   #读入config文件
    tf_config = gpu_config(config)
    model_name, model_dir = check_model_exist(config)
    LOG = open( model_dir  +'/model_log.txt',"w")
    tf.reset_default_graph()
    graph = tf.Graph()
    sess = tf.Session(config = tf_config)
    dataset = Dataset(config)
    train_model = Model(config)


    formatd = '+{0:-^18}+{0:-^23}+{0:-^23}+{0:-^25}+{0:-^20}+'.format('-')
    print(formatd)
    notice_print('start buiding model......')
    train_model.inference()                                          #调用inference函数构建网络
    notice_print('start initialize model......')
    sess.run(tf.global_variables_initializer())                         #网络参数初始化
    notice_print('model is initialized ! ! !')
    loss_valid_best = 10000000                                           #设定一个最优loss值
    accu_valid_best = 0
    loss_train_total = 0
    loss_valid_total = 0
    accu_valid_total = 0
    accu_train_total = 0
    train_feat = dataset.inputX_train
    train_label = dataset.label_train
    valid_feat = dataset.inputX_valid
    valid_label = dataset.label_valid   

#    train_feat, train_label = thread_load('/media/data2/wangjc/data_*', 1000, config.batch_size)
#    valid_feat, valid_label = thread_load('/media/data1/wangjc/stock_prediciton/data/tfrecord/train*', 100000, config.batch_size) 
    tf.local_variables_initializer().run(session=sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter('/media/data1/wangjc/stock_prediciton/model_saved/train_summary', sess.graph)
    notice_print('start training......')
    train_p = train_print(0 , 0,0, 0)
    print (train_p)
    LOG.write( train_p + '\n')
    LOG.flush()
    time_start = time.time()
    time_start1 = time.time()
    count = 0
    sum_loss = 0
    sum_accu = 0
    try:
        for step in range(config.step_num):
            inputX_train , label_train = sess.run([train_feat, train_label])  #从dataset中读取一些数据
            inputX_train[:,:,6,:] = np.log10(inputX_train[:,:,6,:])
            inputX_train[:,:,0,:] = inputX_train[:,:,0,:] - 2015
            _, loss_train, lr, g_step, grads, train_summary, accu_train= sess.run(             #sess.run运行train_op，loss损失，learning_rate学习率等
                        [train_model.train_op_change,
                        train_model.loss,
                        train_model.learning_rate,
                        train_model.global_step,
                        train_model.grads,
                        train_model.merge_summary,
                        train_model.accuracy], 
                feed_dict = {
                        train_model.inputX      : inputX_train,
                        train_model.label       : label_train ,
                        train_model.is_training : True,
                        train_model.momentum   : config.momentum,
                        train_model.lrate       : config.learning_rate,
                        train_model.dropout_rate     : config.dropout
                                })
            train_writer.add_summary(train_summary, step) 
            loss_train = loss_train / config.batch_size
            accu_train_total = accu_train_total + accu_train
            loss_train_total = loss_train_total + loss_train
            count = count + 1
            sum_loss = sum_loss + loss_train
            sum_accu = sum_accu + accu_train

            if ((step+1)%config.num_batch == 0):
                for k in range(config.test_epoch):
                    inputX_valid , label_valid = sess.run([valid_feat, valid_label])
                    inputX_valid[:,:,6,:] = np.log10(inputX_valid[:,:,6,:])
                    inputX_valid[:,:,0,:] = inputX_valid[:,:,0,:] - 2015
                    loss_valid, accu_valid = sess.run(
                            [train_model.loss, train_model.accuracy], 
                    feed_dict = {
                            train_model.inputX      : inputX_valid,
                            train_model.label       : label_valid ,
                            train_model.is_training : True,
                            train_model.momentum    : config.momentum,
                            train_model.lrate       : config.learning_rate,
                            train_model.dropout_rate     : 1.0
                                    })
                    loss_valid = loss_valid / config.batch_size
                    accu_valid_total = accu_valid_total + accu_valid
                    loss_valid_total = loss_valid_total + loss_valid
                time_end = time.time()
                accu_valid_total = accu_valid_total / config.test_epoch
                accu_train_total = accu_train_total / config.num_batch
                loss_train_total = loss_train_total / config.num_batch
                loss_valid_total = loss_valid_total / config.test_epoch
                train_p = train_print(batch = step+1 , l_train = loss_train_total,
                 l_valid = loss_valid_total, a_train = accu_train_total,
                 a_valid = accu_valid_total, time = (time_end - time_start))
                print(train_p)
                LOG.write( train_p + '\n')
                LOG.flush()
                if accu_valid_total>=accu_valid_best :
                    accu_valid_best = accu_valid_total
                    saver.save(sess, os.path.join( model_dir , model_name) ,
                     global_step = step + 1)
                loss_train_total = 0
                loss_valid_total = 0
                accu_valid_total = 0
                accu_train_total = 0
                time_start = time.time()
            elif (step+1)%config.num_step_to_show_cost == 0:
                time_end1 = time.time()
                train_p = train_print(batch = step+1 , l_train = sum_loss/count,
                 l_valid = 0, a_train = sum_accu/count,
                 a_valid = 0, time = (time_end1 - time_start1))
                time_start1 = time.time()
                sum_loss = 0
                sum_accu = 0
                count  = 0
                print(train_p)
                LOG.write( train_p + '\n')
                LOG.flush()
        LOG.close()
        sess.close()
        coord.request_stop()
        coord.join(threads)
    except KeyboardInterrupt:
        saver.save(sess, os.path.join( model_dir , model_name) , global_step = g_step + 1)
        LOG.close()
        sess.close()  
        coord.request_stop()
        coord.join(threads)


def train_old():
    config = Config()                                   #读入config文件
    tf_config = gpu_config(config)
    tf.reset_default_graph()
    graph = tf.Graph()
    sess = tf.Session(config = tf_config)
    dataset = Dataset(config)
    formatd = '+{0:-^18}+{0:-^23}+{0:-^23}+{0:-^25}+{0:-^20}+'.format('-')
    print(formatd)
#    train_feat, train_label = thread_load(config.train_path, 1000, config.batch_size)
#    valid_feat, valid_label = thread_load(config.valid_path, 1000, config.batch_size)
    tf.local_variables_initializer().run(session=sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    model_name = config.model_name                          #获取模型名称，并创建路径model_dir以保存模型
    model_dir = os.path.join('./model_saved', model_name)
    meta_name = find_suffix(model_dir, 'meta')
    data_name = find_suffix(model_dir, 'data-00000-of-00001')
    meta_dir = os.path.join('./model_saved', model_name, meta_name)
    data_dir = os.path.join('./model_saved', model_name, data_name)
    notice_print('start loading model......')
    saver = tf.train.import_meta_graph(meta_dir)
    ckpt = tf.train.get_checkpoint_state(model_dir + '/')
    saver.restore(sess, ckpt.model_checkpoint_path)
    notice_print('model:  '+ (meta_name.split('.')[0]) + '  is loaded ！ ！ ！')
    inputX = tf.get_default_graph().get_tensor_by_name('placeholder/inputX:0')
    label = tf.get_default_graph().get_tensor_by_name('placeholder/label:0')
    is_training = tf.get_default_graph().get_tensor_by_name('placeholder/is_training:0')
    lrate = tf.get_default_graph().get_tensor_by_name('placeholder/learning_rate:0')
    momentum = tf.get_default_graph().get_tensor_by_name('placeholder/momentum:0')
    dropout_rate = tf.get_default_graph().get_tensor_by_name('placeholder/dropout_rate:0')
    loss = tf.get_default_graph().get_tensor_by_name('Sum:0')
    global_step = tf.get_default_graph().get_tensor_by_name('Variable:0')
    merge_summary = tf.get_default_graph().get_tensor_by_name('Merge/MergeSummary:0')
    accuracy = tf.get_default_graph().get_tensor_by_name('Mean:0')
    train_op = tf.get_default_graph().get_operation_by_name('group_deps_1')
    train_op_change = tf.get_default_graph().get_operation_by_name('group_deps_2')
    train_op_regular = tf.get_default_graph().get_operation_by_name('group_deps_3')

    LOG = open( model_dir +'/vanka/yearno' +'/model_log.txt',"a")
    train_writer = tf.summary.FileWriter(model_dir+'/vanka/yearno', sess.graph)
    loss_valid_best = 10000000                                           #设定一个最优loss值
    accu_valid_best = 0
    loss_train_total = 0
    loss_valid_total = 0
    accu_valid_total = 0
    accu_train_total = 0
    train_feat = dataset.inputX_train
    train_label = dataset.label_train
    valid_feat = dataset.inputX_valid
    valid_label = dataset.label_valid   
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(model_dir, sess.graph)
    count = 0
    sum_loss = 0
    sum_accu = 0
    time_start = time.time()
    time_start1 = time.time()
    print(train_print())
    try:
        for step in range(config.step_num):
            inputX_train , label_train = sess.run([train_feat, train_label])  #从dataset中读取一些数据
            inputX_train[:,:,6,:] = np.log10(inputX_train[:,:,6,:])
            inputX_train[:,:,0,:] = inputX_train[:,:,0,:] - 2015
#            _, loss_train, lr, step, grads, train_summary, accu_train= sess.run(             #sess.run运行train_op，loss损失，learning_rate学习率等
#            lrate_change = config.learning_rate*pow(config.lr_decay, step*1.0/config.decay_step)
            lrate_change = config.learning_rate
            _, loss_train, lr, g_step, train_summary, accu_train= sess.run(
                        [train_op_change,
                        loss,
                        lrate,
                        global_step,
#                        grads,
                        merge_summary,
                        accuracy], 
                feed_dict = {
                        inputX      : inputX_train,
                        label       : label_train ,
                        is_training : True,
                        momentum   : config.momentum,
                        lrate       : lrate_change,
                        dropout_rate     : config.dropout
                                })
            train_writer.add_summary(train_summary, g_step) 
            loss_train = loss_train / config.batch_size
            accu_train_total = accu_train_total + accu_train
            loss_train_total = loss_train_total + loss_train
            sum_loss = sum_loss + loss_train
            sum_accu = sum_accu + accu_train
            count = count + 1
            if (g_step+1)%config.num_batch == 0:
                for k in range(config.test_epoch):
                    inputX_valid , label_valid = sess.run([valid_feat, valid_label])
                    inputX_valid[:,:,0,:] = inputX_valid[:,:,0,:] - 2015
                    inputX_valid[:,:,6,:] = np.log10(inputX_valid[:,:,6,:])
                    loss_valid, accu_valid = sess.run(
                        [loss, accuracy], 
                            feed_dict = {
                           inputX      : inputX_valid,
                           label       : label_valid ,
                           is_training : True,
                           momentum    : config.momentum,
                           lrate       : config.learning_rate,
                           dropout_rate     : 1.0
                                 })
                    loss_valid = loss_valid / config.batch_size
                    accu_valid_total = accu_valid_total + accu_valid
                    loss_valid_total = loss_valid_total + loss_valid
                time_end = time.time()
                accu_train_total = accu_train_total / config.num_batch
                loss_train_total = loss_train_total / config.num_batch
                accu_valid_total = accu_valid_total / config.test_epoch
                loss_valid_total = loss_valid_total / config.test_epoch
                train_p = train_print(batch = g_step+1 , l_train = loss_train_total, 
                l_valid = loss_valid_total, a_train = accu_train_total,
                a_valid = accu_valid_total, time = (time_end - time_start))
                print(train_p)
                LOG.write( train_p + '\n')
                LOG.flush()
                if accu_valid_total>=accu_valid_best :
                    accu_valid_best = accu_valid_total
                    saver.save(sess, os.path.join( model_dir+'/vanka/yearno' , model_name) , 
                    global_step = g_step + 1)
                loss_train_total = 0
                accu_train_total = 0
                loss_valid_total = 0
                accu_valid_total = 0
                time_start = time.time()
            elif (g_step+1)%config.num_step_to_show_cost == 0:
#            elif (g_step+1)%2 == 0:
                time_end1 = time.time()
                train_p = train_print(batch = g_step+1 , l_train = sum_loss/count,
                 l_valid = 0, a_train = sum_accu/count,
                 a_valid = 0, time = (time_end1 - time_start1))
                time_start1 = time.time()
                sum_loss = 0
                sum_accu = 0
                count  = 0
                print(train_p)
                LOG.write( train_p + '\n')
                LOG.flush()
        LOG.close()
        coord.request_stop()
        coord.join(threads)
        sess.close()
        
    except KeyboardInterrupt:
        saver.save(sess, os.path.join( model_dir + '/vanka/yearno', model_name) , global_step = g_step + 1)
        LOG.close()
        coord.request_stop()
        coord.join(threads)
        sess.close()  




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default = 'train_old', type = str )
    args = parser.parse_args()
    mode = args.mode
    if mode == 'train_new':
        train_new()
    elif mode == 'train_old':
        train_old()
    else: 
        print('error, unexpected mode')


