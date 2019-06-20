# -*- coding: utf-8 -*-
import numpy as np
def get_config():
    return Config()

class Config(object):
    def __init__(self):
        self.model_name = 'May28'
        self.batch_size = 1024
        self.data_num = 2037176  + 1087801 + 1995379 + 2423882 + 3115799       #data_1 = 2037176, data_2 = 1087801, data_3 = 1995379  data_4 = 2423882  data_5 = 3115799  data_6 = 1994376
#        self.data_num = 1192937          #data_2
        self.test_num = 1994376
        self.test_epoch = self.test_num/self.batch_size
        self.is_training = True
        self.regular = False
        self.step_num = 1000000
        self.dropout = 1.0
        self.gpu = "1"
        self.learning_rate = 0.0003
        self.momentum = 0.9
        self.MOVING_AVERAGE_DECAY = 0.9997
        self.conv_weight_decay = 0.00001
        self.fc_weight_decay = 0.00001
        self.lstm_units = 256
        self.lr_decay = 0.8
        self.decay_step = self.data_num/self.batch_size*2
        self.max_grad_norm = -1
        self.seq_length = 30*20
        self.feature_length = 40
        self.per_process_gpu_memory_fraction = 'change'
        self.drop_list = ['TTIME','TDATE' ]
        self.normalize_list = ['OP', 'CP', 'LOP', 'HIP', 'B1', 'B2', 'B3', 'B4', 'B5', 'S1', 'S2', 'S3', 'S4', 'S5']
        self.use_placeholder = False
        self.shuffle_num = np.random.randint(1,100)
        self.num_batch = self.data_num/self.batch_size
        self.num_step_to_show_cost = self.num_batch/10  
        self.train_path = ['/media/data1/wangjc/stock_prediciton/data/tfrecord/data_1.tfrecords', 
        '/media/data1/wangjc/stock_prediciton/data/tfrecord/data_2.tfrecords',
        '/media/data1/wangjc/stock_prediciton/data/tfrecord/data_3.tfrecords',
        '/media/data2/wangjc/data_4.tfrecords',
        '/media/data1/wangjc/stock_prediciton/data/tfrecord/data_5.tfrecords']
#        self.train_path = ['/media/data2/wangjc/data_4.tfrecords']
        self.valid_path = ['/media/data1/wangjc/stock_prediciton/data/tfrecord/data_6.tfrecords']
        self.test_path = '/media/data1/wangjc/stock_prediciton/data/tfrecord/test.tfrecords'
        
#        self.train_path = './data/tfrecord/train.tfrecords'
#        self.valid_path = './data/tfrecord/valid.tfrecords'

