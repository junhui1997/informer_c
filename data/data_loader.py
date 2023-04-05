import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import get_fold

from utils.tools import StandardScaler
from utils.tools import StandardScaler_classification
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


"""
    这个df本身分布起来就是按照纵向是seq_len分布的，所以我之后处理时候可以参照一下
"""



def create_grouped_array(data, group_col='series_id', drop_cols=['series_id', 'measurement_number']):
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in data.groupby(group_col)])
    return X_grouped

class Dataset_rob(Dataset):
    def __init__(self, flag='train', size=48,
                 enc_in=5, scale=True, inverse=False, cols=None, task='', args=None):
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
                 enc_in=5, scale=True, inverse=False, cols=None,task='jigsaw_kt', args=None):
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
        self.scaler = StandardScaler_classification()
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
        if self.scale:
            # 划定了train data的范围
            # 利用scaler来正则化数据，注意这里使用的是fit
            # 之后利用transform来生成data，注意fit时候是使用的整个train_data，而生成的数据是对整个df，这个符合我们正常的理解，注意这里是borders  ：
            # 因为我们训练时候只能观测到train部分的数据，所以正则化是基于train来做的，然后应用到整个数据中去
            self.scaler.fit(x_train, "value list")
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
                 enc_in=5, scale=False, inverse=False, cols=None,task='jigsaw_kt_g', args=None):
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
        self.scaler = StandardScaler_classification()
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

        len_file = len(df['file_name'].unique())
        file_names = df['file_name'].unique()
        train_ratio = 0.8
        val_ratio = 0.9
        train_files = file_names[0:int(train_ratio * len_file)]
        val_files = file_names[int(train_ratio * len_file):int(val_ratio * len_file)]
        test_files = file_names[int(val_ratio * len_file):]
        if self.flag == 'train':
            self.files = train_files
        elif self.flag == 'val':
            self.files = val_files
        elif self.flag == 'test':
            self.files = test_files
        df = df[df['file_name'].isin(self.files)]

        val_list = []
        label_list = []

        #四分之一的采样率
        for i in range(self.seq_len, df.shape[0],2):
            if df.iloc[i - self.seq_len]['file_name'] != df.iloc[i]['file_name']:
                continue
            # 11是因为第12列开始才是有效数据，详情请看dataloader里面写的
            # 这里直接使用了最后一个点的gesture作为label来计算的
            val = df.iloc[i - self.seq_len:i, 11:11 + self.enc_in].to_numpy()
            label = df.iloc[i]['gesture']
            file_name = df.iloc[i]['file_name']
            val_list.append(val)
            label_list.append(label)
        self.x_data_trn = pd.DataFrame({"value list": val_list, "gesture": label_list})
        self.enc.fit(self.x_data_trn['gesture'])
        self.class_name = self.enc.inverse_transform([i for i in range(len(self.x_data_trn['gesture'].unique()))])

    def __read_data__(self):
        # train val test 0.8:0.1:0.1
        self.data_x = self.x_data_trn
        self.data_y = self.enc.transform(self.x_data_trn['gesture'])
        self.ds_len = len(self.data_y)

        # 在这里先没有考虑seq_len,对于这个数据集来说最长是128
        # if self.scale:
        #     # 划定了train data的范围
        #     # 利用scaler来正则化数据，注意这里使用的是fit
        #     # 之后利用transform来生成data，注意fit时候是使用的整个train_data，而生成的数据是对整个df，这个符合我们正常的理解，注意这里是borders  ：
        #     # 因为我们训练时候只能观测到train部分的数据，所以正则化是基于train来做的，然后应用到整个数据中去
        #     self.scaler.fit(x_train, "value list")
        #     # 在这里直接给划分开来：划分成train，test，pred三种
        #     # ndarray不需要value,df才需要
        #     if self.flag == 'train':
        #         self.data_x = self.scaler.transform(x_train)
        #         self.data_y = y_train
        #         self.ds_len = len(y_train)
        #     elif self.flag == 'val':
        #         self.data_x = self.scaler.transform(x_val)
        #         self.data_y = y_val
        #         self.ds_len = len(y_val)
        #     elif self.flag == 'test':
        #         self.data_x = self.scaler.transform(x_test)
        #         self.data_y = y_test
        #         self.ds_len = len(y_test)
        # else:
        #     if self.flag == 'train':
        #         self.data_x = x_train
        #         self.data_y = y_train
        #         self.ds_len = len(y_train)
        #     elif self.flag == 'val':
        #         self.data_x = x_val
        #         self.data_y = y_val
        #         self.ds_len = len(y_val)
        #     elif self.flag == 'test':
        #         self.data_x = x_test
        #         self.data_y = y_test
        #         self.ds_len = len(y_test)
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