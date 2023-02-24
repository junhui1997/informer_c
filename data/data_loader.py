import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
from utils.tools import get_fold

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


"""
    这个df本身分布起来就是按照纵向是seq_len分布的，所以我之后处理时候可以参照一下
"""
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init

        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        #通过border来划分了不同的train，test_val数据，简单粗暴的截取成了三段!!这里并没有做shuffle
        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            # 忽略了datetime，提取出了所有columns的name，注意默认feature是M
            # 提取了除datetime外的数据
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            #一对一时候使用了target进行预测
            df_data = df_raw[[self.target]]

        if self.scale:
            # 划定了train data的范围
            # 利用scaler来正则化数据，注意这里使用的是fit
            # 之后利用transform来生成data，注意fit时候是使用的整个train_data，而生成的数据是对整个df，这个符合我们正常的理解，注意这里是borders  ：
            # 因为我们训练时候只能观测到train部分的数据，所以正则化是基于train来做的，然后应用到整个数据中去
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 没仔细看，总之是将一个日期给转换成了一个数字进行编码
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        #data_x就是训练数据集
        self.data_x = data[border1:border2]

        #if inverse这里是使用的原始数据，不然使用的是正则化之后的数据
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    #最后输出分别是data,label,以及time_stamp的data和label
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        #(seq_len96,7),(label_len+pred_len72,7),(seq_len96,4),(label_len+pred_len72,4)
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        # 通过border来划分的生成的不同的data_x
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        #scaler写在self里面即可
        return self.scaler.inverse_transform(data)


def create_grouped_array(data, group_col='series_id', drop_cols=['series_id', 'measurement_number']):
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in data.groupby(group_col)])
    return X_grouped

class Dataset_rob(Dataset):
    def __init__(self, flag='train', size=48,
                 enc_in=5, scale=True, inverse=False, cols=None, task=''):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size
        self.enc_in = enc_in

        # init
        # pred是我自己加的，原本他是写在了不同的dataset里面，现在我直接给写到同一个里面去了，下面pred_ds里面是对本次读取到的所有数据scale了一下，这样应该不存在信息泄漏吧
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pred':3 }
        self.flag = flag

        self.scale = scale
        self.inverse = inverse
        self.enc = LabelEncoder()
        self.scaler = StandardScaler()
        self.prepare_data()
        self.__read_data__()

    def prepare_data(self):
        folder = 'data/navi_rob/'
        SAMPLE = folder + 'sample_submission.csv'
        train_path = folder + 'X_train.csv'
        target_path = folder + 'y_train.csv'
        pred_path = folder + 'X_test.csv'

        ID_COLS = ['series_id', 'measurement_number']

        x_cols = {
            'series_id': np.uint32,
            'measurement_number': np.uint32,
            'orientation_X': np.float32,
            'orientation_Y': np.float32,
            'orientation_Z': np.float32,
            'orientation_W': np.float32,
            'angular_velocity_X': np.float32,
            'angular_velocity_Y': np.float32,
            'angular_velocity_Z': np.float32,
            'linear_acceleration_X': np.float32,
            'linear_acceleration_Y': np.float32,
            'linear_acceleration_Z': np.float32
        }

        y_cols = {
            'series_id': np.uint32,
            'group_id': np.uint32,
            'surface': str
        }

        # cwd = os.getcwd() #这个位置不是dataloader所在的位置
        x_trn = pd.read_csv(train_path, usecols=x_cols.keys(), dtype=x_cols)
        x_pred = pd.read_csv(pred_path, usecols=x_cols.keys(), dtype=x_cols)
        y_trn = pd.read_csv(target_path, usecols=y_cols.keys(), dtype=y_cols)
        self.y_enc = self.enc.fit_transform(y_trn['surface'])
        self.x_data_trn = create_grouped_array(x_trn)
        # pred里面存放着没有相应label，直接进行预测的数据，这个是用于kaggle提交，或者实际应用中使用，对于正常写论文可以不需要
        self.x_data_pred = create_grouped_array(x_pred)
        self.class_name = self.enc.inverse_transform([i for i in range(len(y_trn['surface'].unique()))])

    def __read_data__(self):



        # train val test 0.8:0.1:0.1
        # vt:val&test
        x_train, x_vt, y_train, y_vt = train_test_split(self.x_data_trn, self.y_enc, test_size=0.2)
        x_val, x_test, y_val, y_test = train_test_split(x_vt, y_vt, test_size=0.5)


        #在这里先没有考虑seq_len,对于这个数据集来说最长是128

        if self.scale:
            # 划定了train data的范围
            # 利用scaler来正则化数据，注意这里使用的是fit
            # 之后利用transform来生成data，注意fit时候是使用的整个train_data，而生成的数据是对整个df，这个符合我们正常的理解，注意这里是borders  ：
            # 因为我们训练时候只能观测到train部分的数据，所以正则化是基于train来做的，然后应用到整个数据中去
            self.scaler.fit(x_train)
            #在这里直接给划分开来：划分成train，test，pred三种
            # ndarray不需要value,df才需要
            if self.flag == 'train':
                self.data_x = self.scaler.transform(x_train)
                self.data_y = y_train
                self.ds_len = len(y_train)
            elif self.flag == 'val':
                self.data_x = self.scaler.transform(x_val)
                self.data_y = y_val
                self.ds_len = len(y_val)
            elif self.flag == 'test':
                self.data_x = self.scaler.transform(x_test)
                self.data_y = y_test
                self.ds_len = len(y_test)
        else:
            if self.flag == 'train':
                self.data_x = x_train
                self.data_y = y_train
                self.ds_len = len(y_train)
            elif self.flag == 'val':
                self.data_x = x_val
                self.data_y = y_val
                self.ds_len = len(y_val)
            elif self.flag == 'test':
                self.data_x = x_test
                self.data_y = y_test
                self.ds_len = len(y_test)
        end = 1

    # 最后输出分别是data,label,以及time_stamp的data和label
    def __getitem__(self, index):
        data = self.data_x[index][:self.seq_len, :self.enc_in]
        # 原本是个裸露的int，为了使用crossentropy需要改变一下
        label = torch.tensor(self.data_y[index]).to(torch.long)
        return data, label

    def __len__(self):
        return self.ds_len

    def get_encoder(self):
        return self.enc

    def get_class_name(self):
        return self.class_name



class Dataset_jigsaw(Dataset):
    def __init__(self, flag='train', size=48,
                 enc_in=5, scale=True, inverse=False, cols=None,task='jigsaw_kt'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size
        self.enc_in = enc_in
        self.task = task

        # init
        # pred是我自己加的，原本他是写在了不同的dataset里面，现在我直接给写到同一个里面去了，下面pred_ds里面是对本次读取到的所有数据scale了一下，这样应该不存在信息泄漏吧
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pred':3 }
        self.flag = flag

        self.scale = scale
        self.inverse = inverse
        self.enc = LabelEncoder()
        self.scaler = StandardScaler()
        self.prepare_data()
        self.__read_data__()

    def prepare_data(self):
        folder = '../jigsaw/'
        if self.task == 'jigsaw_kt':
            df = pd.read_pickle(folder+'Knot_Tying.pkl')
        elif self.task == 'jigsaw_np':
            df = pd.read_pickle(folder+'Needle_Passing.pkl')
        elif self.task == 'jigsaw_su':
            df = pd.read_pickle(folder+'Suturing.pkl')
        val_list = []
        label_list = []
        #四分之一的采样率
        for i in range(0, df.shape[0]-self.seq_len, 4):
            if df.iloc[i]['label'] != df.iloc[i+self.seq_len]['label']:
                continue
            # 10是因为第11列开始才是有效数据，详情请看dataloader里面写的
            val = df.iloc[i:i + self.seq_len, 11:11 + self.enc_in].to_numpy()
            label = df.iloc[i]['label']
            val_list.append(val)
            label_list.append(label)
        self.x_data_trn = pd.DataFrame({"value list": val_list, "label": label_list})
        self.y_enc = self.enc.fit_transform(self.x_data_trn['label'])
        self.class_name = self.enc.inverse_transform([i for i in range(len(self.x_data_trn['label'].unique()))])

    def __read_data__(self):

        # train val test 0.8:0.1:0.1
        # vt:val&test
        x_train, x_vt, y_train, y_vt = train_test_split(self.x_data_trn, self.y_enc, test_size=0.2)
        x_val, x_test, y_val, y_test = train_test_split(x_vt, y_vt, test_size=0.5)

        # 在这里先没有考虑seq_len,对于这个数据集来说最长是128
        self.scale = False
        if self.scale:
            # 划定了train data的范围
            # 利用scaler来正则化数据，注意这里使用的是fit
            # 之后利用transform来生成data，注意fit时候是使用的整个train_data，而生成的数据是对整个df，这个符合我们正常的理解，注意这里是borders  ：
            # 因为我们训练时候只能观测到train部分的数据，所以正则化是基于train来做的，然后应用到整个数据中去
            self.scaler.fit(x_train)
            # 在这里直接给划分开来：划分成train，test，pred三种
            # ndarray不需要value,df才需要
            if self.flag == 'train':
                self.data_x = self.scaler.transform(x_train)
                self.data_y = y_train
                self.ds_len = len(y_train)
            elif self.flag == 'val':
                self.data_x = self.scaler.transform(x_val)
                self.data_y = y_val
                self.ds_len = len(y_val)
            elif self.flag == 'test':
                self.data_x = self.scaler.transform(x_test)
                self.data_y = y_test
                self.ds_len = len(y_test)
        else:
            if self.flag == 'train':
                self.data_x = x_train
                self.data_y = y_train
                self.ds_len = len(y_train)
            elif self.flag == 'val':
                self.data_x = x_val
                self.data_y = y_val
                self.ds_len = len(y_val)
            elif self.flag == 'test':
                self.data_x = x_test
                self.data_y = y_test
                self.ds_len = len(y_test)
        end = 1

    # 最后输出分别是data,label,以及time_stamp的data和label
    def __getitem__(self, index):
        # 这个enc_in参数写在了上面，没有像navi_rob写在这里
        data = self.data_x['value list'].iloc[index].astype('float64')
        # 原本是个裸露的int，为了使用crossentropy需要改变一下
        label = torch.tensor(self.data_y[index]).to(torch.long)
        return data, label

    def __len__(self):
        return self.ds_len

    def get_encoder(self):
        return self.enc

    def get_class_name(self):
        return self.class_name


class Dataset_jigsaw_g(Dataset):
    def __init__(self, flag='train', size=48,
                 enc_in=5, scale=True, inverse=False, cols=None,task='jigsaw_kt_g'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size
        self.enc_in = enc_in
        self.task = task

        # init
        # pred是我自己加的，原本他是写在了不同的dataset里面，现在我直接给写到同一个里面去了，下面pred_ds里面是对本次读取到的所有数据scale了一下，这样应该不存在信息泄漏吧
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pred':3 }
        self.flag = flag

        self.scale = scale
        self.inverse = inverse
        self.enc = LabelEncoder()
        self.scaler = StandardScaler()
        self.prepare_data()
        self.__read_data__()

    def prepare_data(self):
        folder = '../jigsaw/'
        if self.task == 'jigsaw_kt_g':
            df = pd.read_pickle(folder+'Knot_Tying.pkl')
        elif self.task == 'jigsaw_np_g':
            df = pd.read_pickle(folder+'Needle_Passing.pkl')
        elif self.task == 'jigsaw_su_g':
            df = pd.read_pickle(folder+'Suturing.pkl')
        val_list = []
        label_list = []
        #四分之一的采样率
        for i in range(self.seq_len, df.shape[0],60):
            if df.iloc[i - self.seq_len]['file_name'] != df.iloc[i]['file_name']:
                continue
            # 11是因为第12列开始才是有效数据，详情请看dataloader里面写的
            # 这里直接使用了最后一个点的gesture作为label来计算的
            val = df.iloc[i - self.seq_len:i, 11:11 + self.enc_in].to_numpy()
            label = df.iloc[i]['gesture']
            val_list.append(val)
            label_list.append(label)
        self.x_data_trn = pd.DataFrame({"value list": val_list, "gesture": label_list})
        self.enc.fit(self.x_data_trn['gesture'])
        self.class_name = self.enc.inverse_transform([i for i in range(len(self.x_data_trn['gesture'].unique()))])

    def __read_data__(self):
        # train val test 0.8:0.1:0.1
        num_fold = 10
        df_kflod = get_fold(self.x_data_trn, num_fold, 'gesture')
        x_train = df_kflod.loc[(df_kflod['fold'] <= 7)]
        x_test = df_kflod.loc[(df_kflod['fold'] == 8)]
        x_val = df_kflod.loc[(df_kflod['fold'] == 9)]
        y_train = self.enc.transform(x_train['gesture'])
        y_test = self.enc.transform(x_test['gesture'])
        y_val = self.enc.transform(x_val['gesture'])
        # # vt:val&test
        # x_train, x_vt, y_train, y_vt = train_test_split(self.x_data_trn, self.y_enc, test_size=0.2)
        # x_val, x_test, y_val, y_test = train_test_split(x_vt, y_vt, test_size=0.5)

        # 在这里先没有考虑seq_len,对于这个数据集来说最长是128
        self.scale = False
        if self.scale:
            # 划定了train data的范围
            # 利用scaler来正则化数据，注意这里使用的是fit
            # 之后利用transform来生成data，注意fit时候是使用的整个train_data，而生成的数据是对整个df，这个符合我们正常的理解，注意这里是borders  ：
            # 因为我们训练时候只能观测到train部分的数据，所以正则化是基于train来做的，然后应用到整个数据中去
            self.scaler.fit(x_train)
            # 在这里直接给划分开来：划分成train，test，pred三种
            # ndarray不需要value,df才需要
            if self.flag == 'train':
                self.data_x = self.scaler.transform(x_train)
                self.data_y = y_train
                self.ds_len = len(y_train)
            elif self.flag == 'val':
                self.data_x = self.scaler.transform(x_val)
                self.data_y = y_val
                self.ds_len = len(y_val)
            elif self.flag == 'test':
                self.data_x = self.scaler.transform(x_test)
                self.data_y = y_test
                self.ds_len = len(y_test)
        else:
            if self.flag == 'train':
                self.data_x = x_train
                self.data_y = y_train
                self.ds_len = len(y_train)
            elif self.flag == 'val':
                self.data_x = x_val
                self.data_y = y_val
                self.ds_len = len(y_val)
            elif self.flag == 'test':
                self.data_x = x_test
                self.data_y = y_test
                self.ds_len = len(y_test)
        end = 1

    # 最后输出分别是data,label,以及time_stamp的data和label
    def __getitem__(self, index):
        # 这个enc_in参数写在了上面，没有像navi_rob写在这里
        data = self.data_x['value list'].iloc[index].astype('float64')
        # 原本是个裸露的int，为了使用crossentropy需要改变一下
        label = torch.tensor(self.data_y[index]).to(torch.long)
        return data, label

    def __len__(self):
        return self.ds_len

    def get_encoder(self):
        return self.enc

    def get_class_name(self):
        return self.class_name