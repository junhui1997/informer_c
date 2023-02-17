import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_jigsaw_gvk(Dataset):
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
        for i in range(0, df.shape[0]-self.seq_len,6):
            if df.iloc[i]['file_name'] != df.iloc[i+self.seq_len]['file_name']:
                continue
            # 10是因为第11列开始才是有效数据，详情请看dataloader里面写的
            # 这里直接使用了最后一个点的gesture作为label来计算的
            val = df.iloc[i:i + self.seq_len, 10:10 + self.enc_in].to_numpy()
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