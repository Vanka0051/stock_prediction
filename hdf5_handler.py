# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import sys
import os
from config.config import Config
import time
import argparse



        
def data_process():
    ##########################读取数据
    config = Config()
    
    path = './data/'
    data_list = os.listdir(path)
    df1 = loadhdf5(path + data_list[0])
    print 'df1 is loaded'    
    df2 = loadhdf5(path + data_list[1])
    print 'df2 is loaded'  
    df3 = loadhdf5(path + data_list[2])
    print 'df3 is loaded'  
    df4 = loadhdf5(path + data_list[3])
    print 'df4 is loaded'
    df5 = loadhdf5(path + data_list[4])
    print 'df5 is loaded' 

    df = df1.append(df2)
    del df2, df1
    df = df.append(df3)
    del df3
    df = df.append(df4)
    del df4
    df = df.append(df5)
    del df5
    
    LOG = open('./data/datacleaning.txt', 'w')
##################################

    df = df.drop('TTIME', axis = 1)                               #将TTIME列drop掉

    data_index = df.index.values                                 #获得index的所有值
    data_length = len(data_index)
    print_write('data total length = %d'%(data_length), LOG)
    count = 1
    df.insert(0, 'TIME', df.index.values)                       #新建一个列TIME，其值为index的值（方便以SECCODE， TDATE， TIME排序）
    print_write('start spliting TDATE and TIME into year, month, day, hour, minute, second............', LOG)
    df = df.sort_values(by = ['SECCODE', 'TDATE', 'TIME'])      #以SECCODE， TDATE， TIME 排序
    for add_name in ['YEARS', 'MONTHS', 'DAYS', 'HOURS', 'MINUTES', 'SECONDS']: #新建6列，初始值赋0
        df.insert(count, add_name, np.zeros(data_length))
        count = count +1
    df.insert(42, 'label', np.zeros(data_length))               #新建label列，初始值赋0
    for i in range(data_length):                                #遍历所有数据，将时间数据分离
        df['YEARS'].values[i] = int(str(df['TDATE'].values[i])[0:4])
        df['MONTHS'].values[i] = int(str(df['TDATE'].values[i])[5:7])
        df['DAYS'].values[i] = int(str(df['TDATE'].values[i])[8:10])
        df['MINUTES'].values[i] = int(str(df['TIME'].values[i])[-4] + str(df['TIME'].values[i])[-3])
        df['SECONDS'].values[i] = int(str(df['TIME'].values[i])[-2] + str(df['TIME'].values[i])[-1])
        if str(df['TIME'].values[i])[0] =='9':
            df['HOURS'].values[i] = int(str(df['TIME'].values[i])[0])
        else :
            df['HOURS'].values[i] = int(str(df['TIME'].values[i])[0] + str(df['TIME'].values[i])[1])
        if (i+1)%1000000==0:
            print_write('%d is done' %(i+1), LOG)
    print_write('spliting TDATE and TIME is done ! ! !', LOG)
    df_split_time = pd.HDFStore('./data/data_split_time.h5')
    df_split_time['data'] = df
    df_split_time.close()
    print_write('spliting TDATE and TIME is saved ! ! !', LOG)
    problem_list = []
    count = 0
    key_name = df.keys().values
    key_length = len(key_name)
    print_write('start filling nan .........', LOG)

    test = pd.isnull(df).any()
    test_key = test.keys()
    nan_list = []
    for name in test_key:
        if test[name]:
            nan_list.append(name)
    for name in nan_list:                               #遍历所有数据，如果遇到nan则用上一时刻且code相同的数据填补。
        for i in range(data_length):                    #若上一时刻code不同，则用下一时刻且code相同的数据填补。
            if pd.isnull(df[name].values[i]):           #若下一时刻code相同的数据也为nan，则考虑下第二时刻，若也为nan则下第三时刻。最多推迟5个时刻
                if (df['SECCODE'].values[i-1] == df['SECCODE'].values[i]) and (pd.notnull(df[name].values[i-1])):
                    df[name].values[i] = df[name].values[i-1]
                else:
                    while( pd.isnull(df[name].values[i + count])):
                        count = count + 1
                    df[name].values[i] = df[name].values[i+count]
                    count = 0
            if (i+1)%1000000==0:
                print_write(name + ': %d is done' %(i+1), LOG)
    print_write('filling nan is done', LOG)
    df_fill_nan = pd.HDFStore('./data/data_fill_nan.h5')
    df_fill_nan['data'] = df
    df_fill_nan.close()
    print_write('filling nan is saved ! ! !', LOG)

    print_write('problem_list: '+ str(problem_list), LOG)
    print_write('start normalize data with name : ' + str(config.normalize_list), LOG)
    for name in config.normalize_list:
        for i in range(data_length):
            df[name].values[i] = (df[name].values[i] - df['LASTCLOSE'].values[i]) * 1.0/df['LASTCLOSE'].values[i]
            if (i+1)%1000000==0:
                print_write(name + ': %d is done' %(i+1), LOG)
    
    print_write(str(pd.isnull(df).any()), LOG)
    df_final = pd.HDFStore('./data/data_final.h5')
    df_final['data'] = df
    df_final.close()
    
def compress_process():
    path = './data/data_final.h5'
    config = Config()
    print('h5 file is loading.....')
    df = loadhdf5(path)
    print('h5 file is loaded ! ! !')

    df = df.drop(['TDATE', 'TIME'], axis = 1)
    key_name = df.keys().values
    key_length = len(key_name)
    data_length = len(df.index.values)
    for name in key_name:
        df[name] = df[name].astype('float16')
    df_final = pd.HDFStore('./data/data_compress.h5')
    df_final['data'] = df
    df_final.close()

def label_process():

    path = './data/data_ftp.h5'
    print('h5 file is loading.....')
    df = loadhdf5(path)
    print('h5 file is loaded ! ! !')
#    df = df[0 : 10000000]
    df = df.dropna(axis = 0, how = 'any')
    key_name = df.keys().values
    key_length = len(key_name)
    data_length = len(df.index.values)
    df.insert(key_length, 'MPRICE', np.float32(np.zeros(data_length)))
    df.insert(key_length+1, 'MPRICE15', np.float32(np.zeros(data_length)))
    df.insert(key_length+2, 'LABEL', np.float32(np.zeros(data_length)))
    forecast_length = 300
    for i in range (0, data_length - forecast_length - 20, 20):
        if (df['TDATE'].values[i] == df['TDATE'].values[i + forecast_length]) and (df['SECCODE'].values[i] == df['SECCODE'].values[i + forecast_length]):
            df['MPRICE15'].values[i] = (df['TM'].values[i + forecast_length + 19] - df['TM'].values[i+ forecast_length])/(df['TQ'].values[i + forecast_length + 19] - df['TQ'].values[i+ forecast_length])
            df['MPRICE'].values[i] = (df['TM'].values[i + 19] - df['TM'].values[i])/(df['TQ'].values[i + 19] - df['TQ'].values[i])
            df['LABEL'].values[i] = (df['MPRICE15'].values[i] - df['MPRICE'].values[i])/ df['MPRICE'].values[i]
        else :
            df['MPRICE'].values[i] = np.nan
            df['MPRICE15'].values[i] = np.nan
            df['LABEL'].values[i] = np.nan
        for j in range(1,20):
            df['LABEL'].values[i + j] = df['LABEL'].values[i]
            df['MPRICE'].values[i + j] = df['MPRICE'].values[i]
            df['MPRICE15'].values[i + j] = df['MPRICE15'].values[i]
        if i%1000000==0:
            print('%d is done !'%(i))
    #df = df.drop(['TDATE'], axis = 1)
    df['MPRICE15'].values[data_length-forecast_length-20 :data_length] = np.nan
    df['MPRICE'].values[data_length-forecast_length-20 :data_length] = np.nan
    df['LABEL'].values[data_length-forecast_length-20 :data_length] = np.nan

    df = df.dropna(axis = 0, how = 'any')
    df_label = pd.HDFStore('./data/data_label.h5', complevel = 5, complib = 'zlib')
    df_label['data'] = df
    df_label.close()


def str2int(input_list):
    tmp_list = []
    for i in range(len(input_list)):
        tmp_list.append(int(input_list[i][1:]))
    tmp_list = sorted(tmp_list)
    return tmp_list
################检查数据长度######################

def check_length(raw_data, keys, alldata_length):
    keys_length = len(keys)
    key_name = str2int(keys)
    datalost_keys_list = [ind for ind in keys if raw_data[ind].shape[0]!=alldata_length]
    datalost_keys_list = str2int(datalost_keys_list)
    return key_name, datalost_keys_list

################取出数据长度有问题的数据的具体长度######################
def check_certain_length(raw_data, datalost_keys_list):
    datalost_length_list = []
    for i in range(len(datalost_keys_list)):
        datalost_length_list.append(raw_data[str(datalost_keys_list[i])].shape[0])
    return datalost_length_list

################检查特征长度######################
def check_feat_length(raw_data, keys):
    key_lost_list = [ind for ind in keys if raw_data[ind].shape[1]!=36]
    return key_lost_list

################检查nan项######################
def check_null(raw_data, keys):
    null_list = []
    keys_length = len(keys)
    for i in range(keys_length):
        df = raw_data[keys[i]]
        if pd.isnull(df).any().any():
            null_list.append(keys[i])
    null_list = str2int(null_list)
    return null_list

def check_alldata_length(raw_data, keys):
    sum_ = 0
    for i in range(len(keys)):
        df = raw_data[keys[i]]
        sum_ = sum_ + df.shape[0]
    return sum_



def new_data_analyse():
    path = '/media/data1/wangjc/stock_prediciton/data/raw_data/tickdata_2017-08.h5'
    raw_data = pd.HDFStore(path)
    keys = raw_data.keys()
    keys_length = len(keys)
    key_name, datalost_keys_list = check_length(raw_data, keys, 110400)
    datalost_length_list = check_certain_length(raw_data, datalost_keys_list)
    key_lost_list = check_feat_length(raw_data, keys)
    null_list = check_null(raw_data, keys)
    sum_ = check_alldata_length(raw_data, keys)


def new_data_process(num):
    count_2 = 0
    df300 = list(pd.read_csv('./document/000300cons.csv').Code.values)
    data_analyse = pd.read_csv('./document/data_analyse.csv')
    keys = data_analyse.code_id.values[num - 1][1:-1].split(', ')
    keys = map(int, keys)
    keys_300 = [name for name in df300 if name in keys]


    null_list = data_analyse.null_list.values[num - 1][1:-1].split(', ')
    null_list = map(int, null_list)
    lost_list = data_analyse.lost_list.values[num - 1][1:-1].split(', ')
    lost_list = map(int, lost_list)
    keys_300_null = [name for name in keys_300 if name in null_list]
    keys_300_lost = [name for name in keys_300 if name in lost_list]

    path = '/media/data1/wangjc/stock_prediciton/data/raw_data/tickdata_2017-0%d.h5'%(num)
    raw_data = pd.HDFStore(path)
    keys_length = len(keys)
    count_1 = 0


    for name in keys_300:
        df = raw_data[str(name)]
        df_length = df.shape[0]
        df = df.drop(['TTIME'], axis = 1)
        df_keys = df.keys()
        count = 0
        if name in keys_300_null:
            for key_name in df_keys[1:]:
                for i in range(df_length):                    #若上一时刻code不同，则用下一时刻且code相同的数据填补。
                    if pd.isnull(df[key_name].values[i]):           #若下一时刻code相同的数据也为nan，则考虑下第二时刻，若也为nan则下第三时刻。最多推迟5个时刻
                        if (pd.notnull(df[key_name].values[i-1])):
                            df[key_name].values[i] = df[key_name].values[i-1]
                        else:
                            while(pd.isnull(df[key_name].values[i + count])):
                                count = count + 1
                            df[key_name].values[i] = df[key_name].values[i+count]
                            count = 0


        df.insert(0, 'TIME', df.index.values)
        df = df.sort_values(by = [ 'TDATE', 'TIME'])
        count = 0
        data_length = df.shape[0]
        for add_name in ['YEARS', 'MONTHS', 'DAYS', 'HOURS', 'MINUTES', 'SECONDS']:
            df.insert(count, add_name, np.float16(np.zeros(data_length)))
            count = count +1
        df.YEARS = df.TDATE.map(years_f)
        df.MONTHS = df.TDATE.map(months_f)
        df.DAYS = df.TDATE.map(days_f)
        df.HOURS = df.TIME.map(hours_f)
        df.MINUTES = df.TIME.map(minutes_f)
        df.SECONDS = df.TIME.map(seconds_f)
        df = df.drop(['TIME'], axis = 1)



        for name in ['YEARS', 'MONTHS', 'DAYS', 'HOURS', 'MINUTES', 'SECONDS']:
            df[name] = df[name].astype('float16')

        df.insert(41, 'MPRICE', np.float32(np.zeros(data_length)))
        df.insert(41+1, 'MPRICE15', np.float32(np.zeros(data_length)))
        df.insert(41+2, 'LABEL', np.float32(np.zeros(data_length)))
        forecast_length = 15*20
        for i in range (0, data_length - forecast_length - 20, 20):
            if (df['TDATE'].values[i] == df['TDATE'].values[i + forecast_length]):
                df['MPRICE15'].values[i] = (df['TM'].values[i + forecast_length + 19] - df['TM'].values[i+ forecast_length])/(df['TQ'].values[i + forecast_length + 19] - df['TQ'].values[i+ forecast_length])
                df['MPRICE'].values[i] = (df['TM'].values[i + 19] - df['TM'].values[i])/(df['TQ'].values[i + 19] - df['TQ'].values[i])
                df['LABEL'].values[i] = (df['MPRICE15'].values[i] - df['MPRICE'].values[i])/ df['MPRICE'].values[i]
            else :
                df['MPRICE'].values[i] = np.nan
                df['MPRICE15'].values[i] = np.nan
                df['LABEL'].values[i] = np.nan
            for j in range(1,20):
                df['LABEL'].values[i + j] = df['LABEL'].values[i]
                df['MPRICE'].values[i + j] = df['MPRICE'].values[i]
                df['MPRICE15'].values[i + j] = df['MPRICE15'].values[i]

        #df = df.drop(['TDATE'], axis = 1)
        df['MPRICE15'].values[data_length-forecast_length-20 :data_length] = np.nan
        df['MPRICE'].values[data_length-forecast_length-20 :data_length] = np.nan
        df['LABEL'].values[data_length-forecast_length-20 :data_length] = np.nan
        df = df.drop(['TDATE', 'MPRICE', 'MPRICE15'], axis = 1)
        df = df.dropna(axis = 0, how = 'any')

        df.insert(41, 'LABELNEW', np.float32(np.zeros(df.shape[0])))
        df.LABELNEW = df.LABEL.map(label_f)
        df = df.drop(['LABEL'], axis = 1)
        for name in ['TM', 'CM', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'BV1', 'BV2', 'BV3', 'BV4', 'BV5', 'TQ', 'CQ']:
            df[name] = df[name].map(log_f)
        for name in ['OP', 'CP', 'LOP', 'HIP', 'B1', 'B2', 'B3', 'B4', 'B5', 'S1', 'S2', 'S3', 'S4', 'S5']:
            df[name]= df.apply(lambda df:(df[name] - df.LASTCLOSE)/df.LASTCLOSE, axis = 1)
            df[name].astype('float32')
        if count_1==0:
            df1 = df
            count_1 = count_1 + 1
        df1 = df1.append(df)
        count_2 = count_2 + 1
        if count_2%10==0:
            print count_2
    df_all = pd.HDFStore('/media/data1/wangjc/stock_prediciton/data/pro_data/data_%d.h5'%(num),
             complevel = 5, complib = 'zlib')
    df_all['data'] = df1
    df_all.close()











def loadhdf5(path):
    raw_data = pd.HDFStore(path)
    df = raw_data['data']
    raw_data.close()
    return df
def print_write(string,  LOG):
    print string
    LOG.write(string + '\n')
    LOG.flush()
def years_f(x):
    return int(str(x)[0:4])
def months_f(x):
    return int(str(x)[5:7])
def days_f(x):
    return int(str(x)[8:10])
def minutes_f(x):
    return int(str(x)[-4:-2])
def seconds_f(x):
    return int(str(x)[-2:])
def hours_f(x):
    if str(x)[0] =='9':
        return int(9)
    else:
        return int(str(x)[0:2])
def log_f(x):
    if x<=0:
            return 0
    else:
            return np.log10(x)
def label_f(x):
    if x>=0.007:
        return 1
    elif x<0.007 and x>0.003:
        return 2
    elif x<=0.003 and x>=-0.003:
        return 3
    elif x>-0.007 and x<-0.003:
        return 4
    else:
        return 5

def price_f(x, y):
    y = (y - x['LASTCLOSE']) * 1.0/x['LASTCLOSE']



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default = 'label_process', type = str )
    parser.add_argument('-num', default = 2, type = int )
    args = parser.parse_args()
    num = args.num
    new_data_process(num)

#    if mode == 'data_process':
#        data_process()
#    elif mode == 'label_process':
#        label_process()
#    elif mode == 'compress_process':
#        compress_process()
#    else: 
#        print('error, unexpected mode')







