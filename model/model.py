# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:53:06 2019
@author: vanka0051
"""

import tensorflow as tf
import time
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tools.tools import notice_print
from config.config import Config

UPDATE_OPS_COLLECTION = 'UPDATE_OPS'
TRAINING_VARIABLES = 'TRAINING_VARIABLES'
config = Config()

'''
Model类用于创建函数
inference函数用于模型构建推导
loss为损失
optimizer为优化器，使用了adam优化
train_op为训练op
'''
class Model(object):
    def __init__(self, config):
        self.config = config
########################inference funcotion##################################
    def inference(self): #inputX :[batch_size, seq_length, feat_num*20, channel]
        self.inputX , self.label, self.is_training, self.lrate, self.momentum, self.dropout_rate = prepare_placeholder(self.config)#准备placeholder
        with tf.variable_scope("cnn_m20Ds"):
            self.conv1_c1 = conv(self.inputX, 20, 1, 20, 1, 1, '1', name = 'integration') #conv1 :[batch_size, seq_length, feat_num, 1]
            self.bn1_c1 = bn(self.conv1_c1, self.is_training, '1')#第一层首先进行20条数据整合为1条数据的操作（整合一分钟内数据）
            self.relu1 = tf.nn.relu(self.bn1_c1)
        with tf.variable_scope("channel1"):
            with tf.variable_scope("CNN_part"):
                with tf.variable_scope("cnn_3Ds"):
                    self.conv2_3_c1 = conv(self.relu1, 3, self.config.feature_length, 1, self.config.feature_length, 20,'2_3')#conv2 :[batch_size, length, 1, channel]
                    self.bn2_3_c1 = bn(self.conv2_3_c1, self.is_training, '2_3')
                    self.relu2_3_c1 = tf.nn.relu(self.bn2_3_c1)#通过3×feat_len卷积提取特征，将所有特征按20种不同比例融合成20种新特征
                    self.relu2_3_c1 = tf.squeeze(self.relu2_3_c1, 2)
                with tf.variable_scope("cnn_5Ds"):
                    self.conv2_5_c1 = conv(self.relu1, 5, self.config.feature_length, 1, self.config.feature_length, 20,'2_5')#conv2 :[batch_size, length, 1, channel]
                    self.bn2_5_c1 = bn(self.conv2_5_c1, self.is_training, '2_5')
                    self.relu2_5_c1 = tf.nn.relu(self.bn2_5_c1)#通过5×feat_len卷积提取特征，将所有特征按20种不同比例融合成20种新特征
                    self.relu2_5_c1 = tf.squeeze(self.relu2_5_c1, 2)
                with tf.variable_scope("cnn_9Ds"):
                    self.conv2_9_c1 = conv(self.relu1, 9, self.config.feature_length, 1, self.config.feature_length, 20,'2_9')#conv2 :[batch_size, length, 1, channel]
                    self.bn2_9_c1 = bn(self.conv2_9_c1, self.is_training, '2_9')
                    self.relu2_9_c1 = tf.nn.relu(self.bn2_9_c1)#通过9×feat_len卷积提取特征，将所有特征按20种不同比例融合成20种新特征
                    self.relu2_9_c1 = tf.squeeze(self.relu2_9_c1, 2)
                with tf.variable_scope("cnn_maxpool"):
                    self.cnn_out_c1 = tf.concat([self.relu2_3_c1, self.relu2_5_c1, self.relu2_9_c1], 2)#将新特征组合在一起
                    self.cnn_out_c1 = tf.expand_dims(self.cnn_out_c1, 3)
                    self.cnn1_c1 = conv(self.cnn_out_c1, 3, 3, 2, 2, 128, 'cnn1')#将组合特征再通过cnn和maxpool提取特征
                    self.cnn1_c1 = bn(self.cnn1_c1, self.is_training, '11')
                    self.maxpool1_c1 = max_pool(self.cnn1_c1)
                    self.cnn2_c1 = conv(self.maxpool1_c1, 3, 3, 2, 2, 256, 'cnn2')
                    self.cnn2_c1 = bn(self.cnn2_c1, self.is_training, '22')
                    self.maxpool2_c1 = max_pool(self.cnn2_c1)
                    self.cnn3_c1 = conv(self.maxpool2_c1, 3, 3, 2, 2, 512, 'cnn3')
                    self.cnn3_c1 = bn(self.cnn3_c1, self.is_training, '33')
                    self.maxpool3_c1 = max_pool(self.cnn3_c1, stride_h = 4)
                    self.maxpool3_c1 = tf.reshape(self.maxpool3_c1, [-1, 1, 512])
            with tf.variable_scope("BiLSTM_part"):              #BiLSTM部分，3个隐藏层
                with tf.variable_scope("BiLSTM_1"):
                    self.BiLSTM_out1_c1 = BiLSTM(self.maxpool3_c1, self.config.lstm_units)
                with tf.variable_scope("BiLSTM_2"):
                    self.BiLSTM_out2_c1 = BiLSTM(self.BiLSTM_out1_c1, self.config.lstm_units)
                with tf.variable_scope("BiLSTM_3"):
                    self.BiLSTM_out3_c1 = BiLSTM(self.BiLSTM_out2_c1, self.config.lstm_units)
                self.BiLSTM_out3_c1 = tf.squeeze(self.BiLSTM_out3_c1, 1)
            with tf.variable_scope("FC_part"):                  #全连接层将400维特征缩小为200维，再缩小为1。即：预测值
                self.fc1_c1 = fc(self.BiLSTM_out3_c1, 400, num = '1')
                self.fc1_c1 = tf.nn.tanh(self.fc1_c1)
                self.fc1_c1 = tf.nn.dropout(self.fc1_c1, keep_prob = self.dropout_rate)
                self.fc2_c1 = fc(self.fc1_c1, 200, num = '2')


        with tf.variable_scope("channel2"):
            with tf.variable_scope("CNN_part"):
                with tf.variable_scope("cnn_3Ds"):
                    self.conv2_3_c2 = conv(self.relu1, 3, self.config.feature_length, 1, self.config.feature_length, 20,'2_3')#conv2 :[batch_size, length, 1, channel]
                    self.bn2_3_c2 = bn(self.conv2_3_c2, self.is_training, '2_3')
                    self.relu2_3_c2 = tf.nn.relu(self.bn2_3_c2)
                    self.relu2_3_c2 = tf.squeeze(self.relu2_3_c2, 2)
                with tf.variable_scope("cnn_5Ds"):
                    self.conv2_5_c2 = conv(self.relu1, 5, self.config.feature_length, 1, self.config.feature_length, 20,'2_5')#conv2 :[batch_size, length, 1, channel]
                    self.bn2_5_c2 = bn(self.conv2_5_c2, self.is_training, '2_5')
                    self.relu2_5_c2 = tf.nn.relu(self.bn2_5_c2)
                    self.relu2_5_c2 = tf.squeeze(self.relu2_5_c2, 2)
                with tf.variable_scope("cnn_9Ds"):
                    self.conv2_9_c2 = conv(self.relu1, 9, self.config.feature_length, 1, self.config.feature_length, 20,'2_9')#conv2 :[batch_size, length, 1, channel]
                    self.bn2_9_c2 = bn(self.conv2_9_c2, self.is_training, '2_9')
                    self.relu2_9_c2 = tf.nn.relu(self.bn2_9_c2)
                    self.relu2_9_c2 = tf.squeeze(self.relu2_9_c2, 2)
                with tf.variable_scope("cnn_maxpool"):
                    self.cnn_out_c2 = tf.concat([self.relu2_3_c2, self.relu2_5_c2, self.relu2_9_c2], 2)
                    self.cnn_out_c2 = tf.expand_dims(self.cnn_out_c2, 3)
                    self.cnn1_c2 = conv(self.cnn_out_c2, 3, 3, 2, 2, 128, 'cnn1')
                    self.cnn1_c2 = bn(self.cnn1_c2, self.is_training, '11')
                    self.maxpool1_c2 = max_pool(self.cnn1_c2)
                    self.cnn2_c2 = conv(self.maxpool1_c2, 3, 3, 2, 2, 256, 'cnn2')
                    self.cnn2_c2 = bn(self.cnn2_c2, self.is_training, '22')
                    self.maxpool2_c2 = max_pool(self.cnn2_c2)
                    self.cnn3_c2 = conv(self.maxpool2_c2, 3, 3, 2, 2, 512, 'cnn3')
                    self.cnn3_c2 = bn(self.cnn3_c2, self.is_training, '33')
                    self.maxpool3_c2 = max_pool(self.cnn3_c2, stride_h = 4)
                    self.maxpool3_c2 = tf.reshape(self.maxpool3_c2, [-1, 1, 512])
            with tf.variable_scope("BiLSTM_part"):
                with tf.variable_scope("BiLSTM_1"):
                    self.BiLSTM_out1_c2 = BiLSTM(self.maxpool3_c2, self.config.lstm_units)
                with tf.variable_scope("BiLSTM_2"):
                    self.BiLSTM_out2_c2 = BiLSTM(self.BiLSTM_out1_c2, self.config.lstm_units)
                with tf.variable_scope("BiLSTM_3"):
                    self.BiLSTM_out3_c2 = BiLSTM(self.BiLSTM_out2_c2, self.config.lstm_units)
                self.BiLSTM_out3_c2 = tf.squeeze(self.BiLSTM_out3_c2, 1)
            with tf.variable_scope("FC_part"):
                self.fc1_c2 = fc(self.BiLSTM_out3_c2, 400, num = '1')
                self.fc1_c2 = tf.nn.tanh(self.fc1_c2)
                self.fc1_c2 = tf.nn.dropout(self.fc1_c2, keep_prob = self.dropout_rate)
                self.fc2_c2 = fc(self.fc1_c2, 200, num = '2')



        with tf.variable_scope("channel3"):
            with tf.variable_scope("CNN_part"):
                with tf.variable_scope("cnn_3Ds"):
                    self.conv2_3_c3 = conv(self.relu1, 3, self.config.feature_length, 1, self.config.feature_length, 20,'2_3')#conv2 :[batch_size, length, 1, channel]
                    self.bn2_3_c3 = bn(self.conv2_3_c3, self.is_training, '2_3')
                    self.relu2_3_c3 = tf.nn.relu(self.bn2_3_c3)
                    self.relu2_3_c3 = tf.squeeze(self.relu2_3_c3, 2)
                with tf.variable_scope("cnn_5Ds"):
                    self.conv2_5_c3 = conv(self.relu1, 5, self.config.feature_length, 1, self.config.feature_length, 20,'2_5')#conv2 :[batch_size, length, 1, channel]
                    self.bn2_5_c3 = bn(self.conv2_5_c3, self.is_training, '2_5')
                    self.relu2_5_c3 = tf.nn.relu(self.bn2_5_c3)
                    self.relu2_5_c3 = tf.squeeze(self.relu2_5_c3, 2)
                with tf.variable_scope("cnn_9Ds"):
                    self.conv2_9_c3 = conv(self.relu1, 9, self.config.feature_length, 1, self.config.feature_length, 20,'2_9')#conv2 :[batch_size, length, 1, channel]
                    self.bn2_9_c3 = bn(self.conv2_9_c3, self.is_training, '2_9')
                    self.relu2_9_c3 = tf.nn.relu(self.bn2_9_c3)
                    self.relu2_9_c3 = tf.squeeze(self.relu2_9_c3, 2)
                with tf.variable_scope("cnn_maxpool"):
                    self.cnn_out_c3 = tf.concat([self.relu2_3_c3, self.relu2_5_c3, self.relu2_9_c3], 2)
                    self.cnn_out_c3 = tf.expand_dims(self.cnn_out_c3, 3)
                    self.cnn1_c3 = conv(self.cnn_out_c3, 3, 3, 2, 2, 128, 'cnn1')
                    self.cnn1_c3 = bn(self.cnn1_c3, self.is_training, '11')
                    self.maxpool1_c3 = max_pool(self.cnn1_c3)
                    self.cnn2_c3 = conv(self.maxpool1_c3, 3, 3, 2, 2, 256, 'cnn2')
                    self.cnn2_c3 = bn(self.cnn2_c3, self.is_training, '22')
                    self.maxpool2_c3 = max_pool(self.cnn2_c3)
                    self.cnn3_c3 = conv(self.maxpool2_c3, 3, 3, 2, 2, 512, 'cnn3')
                    self.cnn3_c3 = bn(self.cnn3_c3, self.is_training, '33')
                    self.maxpool3_c3 = max_pool(self.cnn3_c3, stride_h = 4)
                    self.maxpool3_c3 = tf.reshape(self.maxpool3_c3, [-1, 1, 512])
            with tf.variable_scope("BiLSTM_part"):
                with tf.variable_scope("BiLSTM_1"):
                    self.BiLSTM_out1_c3 = BiLSTM(self.maxpool3_c3, self.config.lstm_units)
                with tf.variable_scope("BiLSTM_2"):
                    self.BiLSTM_out2_c3 = BiLSTM(self.BiLSTM_out1_c3, self.config.lstm_units)
                with tf.variable_scope("BiLSTM_3"):
                    self.BiLSTM_out3_c3 = BiLSTM(self.BiLSTM_out2_c3, self.config.lstm_units)
                self.BiLSTM_out3_c3 = tf.squeeze(self.BiLSTM_out3_c3, 1)
            with tf.variable_scope("FC_part"):
                self.fc1_c3 = fc(self.BiLSTM_out3_c3, 400, num = '1')
                self.fc1_c3 = tf.nn.tanh(self.fc1_c3)
                self.fc1_c3 = tf.nn.dropout(self.fc1_c3, keep_prob = self.dropout_rate)
                self.fc2_c3 = fc(self.fc1_c3, 200, num = '2')
        with tf.variable_scope("final_fc"):
            self.fc_out = tf.concat([self.fc2_c1, self.fc2_c2, self.fc2_c3], 1)
            self.inference_out = fc(self.fc_out, 5 )
            self.inference_out_softmax = tf.nn.softmax(self.inference_out)
########################loss funcotion##################################
        self.loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels = self.label ,logits = self.inference_out))
        self.correct_prediction = tf.equal(tf.argmax(self.inference_out_softmax, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.loss_regular = self.loss + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.global_step = tf.Variable(0, trainable=False)
        self.reset_global_step = tf.assign(self.global_step, 1)
        initial_learning_rate = tf.Variable(
            self.config.learning_rate, trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            initial_learning_rate, self.global_step, self.config.decay_step,
            self.config.lr_decay, name='lr')
        
#        self.ema = tf.train.ExponentialMovingAverage(config.MOVING_AVERAGE_DECAY, self.global_step)
#        tf.add_to_collection(UPDATE_OPS_COLLECTION, self.ema.apply([self.loss]))


        self.optimizer_regular = tf.train.AdamOptimizer(self.learning_rate, name = 'Adam_regular')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name = 'Adam_decay')
        self.optimizer_change = tf.train.AdamOptimizer(learning_rate = self.lrate, name = 'Adam_change')
        self.vs = tf.trainable_variables()
        grads_and_vars = self.optimizer.compute_gradients(self.loss,
                                                            self.vs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if
                            g is not None]
        self.grads = [grad for (grad, var) in grads_and_vars]
        self.vs = [var for (grad, var) in grads_and_vars]
        if self.config.max_grad_norm > 0:
            self.grads, hehe = tf.clip_by_global_norm(
                self.grads, self.config.max_grad_norm)
        self.batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
        self.batchnorm_updates_op = tf.group(*self.batchnorm_updates)
        self.apply_gradient_op = self.optimizer.apply_gradients(
            zip(self.grads, self.vs),
            global_step=self.global_step, name = 'Train_decay')
        self.train_op = tf.group(self.apply_gradient_op, self.batchnorm_updates_op)


############################change##########################################3
        grads_and_vars_change = self.optimizer_change.compute_gradients(self.loss,
                                                            self.vs)
        grads_and_vars_change = [(g, v) for g, v in grads_and_vars_change if
                            g is not None]
        self.grads_change = [grad for (grad, var) in grads_and_vars_change]
        self.vs_change = [var for (grad, var) in grads_and_vars_change]

        self.apply_gradient_op_change = self.optimizer_change.apply_gradients(
            zip(self.grads_change, self.vs_change),
            global_step=self.global_step, name = 'Train_change')
        self.train_op_change = tf.group(self.apply_gradient_op_change, self.batchnorm_updates_op)

###########################regular##############################3
        grads_and_vars_regular = self.optimizer_regular.compute_gradients(self.loss_regular,
                                                            self.vs)
        grads_and_vars_regular = [(g, v) for g, v in grads_and_vars_regular if
                            g is not None]
        self.grads_regular = [grad for (grad, var) in grads_and_vars_regular]
        self.vs_regular = [var for (grad, var) in grads_and_vars_regular]
        if self.config.max_grad_norm > 0:
            self.grads_regular, hehe = tf.clip_by_global_norm(
                self.grads_regular, self.config.max_grad_norm)
        self.apply_gradient_op_regular = self.optimizer_regular.apply_gradients(
            zip(self.grads_regular, self.vs_regular),
            global_step=self.global_step, name = 'Train_regular')
        self.train_op_regular = tf.group(self.apply_gradient_op_regular, self.batchnorm_updates_op)


        
########################summary part######################################

#        tf.summary.scalar('loss_avg', self.ema.average(self.loss))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
        self.merge_summary = tf.summary.merge_all(key = 'summaries')
        formatd = '+{0:-^18}+{0:-^23}+{0:-^23}+{0:-^25}+{0:-^20}+'.format('-')
        print(formatd)
        notice_print('model is built ! ! !')



'''
准备placeholder 
inputX输入数据
label标签
is_training是否处于训练模式（用于控制batch_normalization）
lrate学习率
momentum动量
dropout_rate遗忘率
'''
def prepare_placeholder(config):
    with tf.variable_scope("placeholder"):
        inputX = tf.placeholder(tf.float32, shape = [None, config.seq_length, config.feature_length, 1], name = 'inputX')
    #   inputX = normalize(inputX)
    #    inputX = tf.expand_dims(inputX, -1)
        label = tf.placeholder(tf.float32, shape = [None, 5], name = 'label')
        is_training = tf.placeholder(tf.bool, shape = [], name = 'is_training')
        lrate = tf.placeholder(tf.float32, shape = [], name = 'learning_rate')
        momentum = tf.placeholder(tf.float32, shape = [], name = 'momentum')
        dropout_rate = tf.placeholder(tf.float32, shape = [], name = 'dropout_rate')
    return inputX, label, is_training, lrate, momentum, dropout_rate




'''
以下函数说明
BiLSTM用于创建BiLSTM
bn用于创建batch_normaliztion层
conv用于创建卷积层
fc用于创建全连接层。注意起输入为[batch_size, num_units_in]
_get_variable用于创建网络层中需要用到的变量，例如weights，bias等
_max_pool用于创建最大值池化层
'''

def BiLSTM(inputX ,  num_units):
    x_shape = tf.shape(inputX)
    batch_size = x_shape[0]
    cell_fw = tf.contrib.rnn.LSTMCell(num_units= num_units)
    cell_bw = tf.contrib.rnn.LSTMCell(num_units= num_units)
    init_statef = cell_fw.zero_state(batch_size, dtype=tf.float32)
    init_stateb = cell_fw.zero_state(batch_size, dtype=tf.float32)
            # tensor of shape: [max_time, batch_size, input_size]
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                        cell_bw, inputs  = inputX, 
                        initial_state_fw = init_statef,
                        initial_state_bw = init_stateb)
    output_fw, output_bw = outputs
#    output_states = output_fw
    output_states = tf.concat([output_fw, output_bw], 2)
    hidden = output_states
    output = hidden
    output = tf.nn.relu(hidden)   
    return output



def bn(inputX, is_training, num):
    x_shape = inputX.get_shape()
    params_shape = x_shape[-1:]
    axis = list(range(len(x_shape) - 1))
    beta = _get_variable('beta' + num, params_shape, initializer = tf.zeros_initializer)
    gamma = _get_variable('gama'+ num, params_shape, initializer = tf.ones_initializer)
    moving_mean = _get_variable('moving_mean'+ num, params_shape, initializer = tf.zeros_initializer)
    moving_variance = _get_variable('moving_variance'+ num, params_shape ,initializer = tf.ones_initializer)
    mean, variance = tf.nn.moments(inputX, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.997)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, 0.997)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))
    inputX = tf.nn.batch_normalization(inputX, mean, variance, beta, gamma, 0.001)
    return inputX


def conv(inputX, kheight, kwidth, stride_h, stride_w, filters_out, num, name = None):
    filters_in = inputX.get_shape()[-1]
    shape = [kheight, kwidth, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer()
#    initializer = tf.constant_initializer(5)
    weights = _get_variable('weights' + num,
                            shape = shape,
                            initializer = initializer,
                            weight_decay =  config.conv_weight_decay 
                            )
    if name != None:
        inputX = tf.nn.conv2d(inputX , weights, [1, stride_h, stride_w, 1], padding = 'SAME')
    else :
        inputX = tf.nn.conv2d(inputX , weights, [1, stride_h, stride_w, 1], padding = 'SAME', name = name)
    return inputX


def fc(inputX, num_units_out, num = 'None'):
    num_units_in = inputX.get_shape()[1]
    initializer = tf.truncated_normal_initializer(
        stddev = 0.01 )
#    initializer = tf.constant_initializer(5)
    weights = _get_variable('weights' + num, 
                            shape = [num_units_in, num_units_out],
                            initializer = initializer,
                            weight_decay = config.fc_weight_decay
    )
    bias = _get_variable('bias'+ num,
                        shape = num_units_out,
                        initializer = tf.zeros_initializer
    )
    inputX = tf.nn.xw_plus_b(inputX, weights, bias)
    return inputX


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay = 0.0,
                  dtype = 'float',
                  trainable = True
                        ):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else :
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, TRAINING_VARIABLES]
    return tf.get_variable( name,
                            shape = shape,
                            initializer = initializer,
                            regularizer = regularizer,
                            dtype = dtype,
                            collections = collections,
                            trainable = trainable
                            )


def max_pool(inputX, kheight = 3, kwidth = 3, stride_h = 2, stride_w = 2):
    return tf.nn.max_pool(inputX,
                          ksize=[1, kheight, kwidth, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding='SAME')

