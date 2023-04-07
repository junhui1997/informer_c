import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import get_fold
# from sklearn.preprocessing import StandardScaler
from PIL import Image

from utils.tools import StandardScaler_classification
from data.transform_list import transform_train,transform_test


import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')


def change_time(ori_time, delta):
    ori_time = ori_time.split('_')
    ori_time = [int(time) for time in ori_time]
    dt = datetime(ori_time[0], ori_time[1], ori_time[2], ori_time[3], ori_time[4], ori_time[5])
    # 加上1秒
    dt += timedelta(seconds=delta)
    # 输出结果
    year = dt.strftime('%Y')  # 年，例如：2022
    month = dt.strftime('%m')  # 月，例如：11
    day = dt.strftime('%d')  # 日，例如：27
    hour = dt.strftime('%H')  # 时，例如：16
    minute = dt.strftime('%M')  # 分，例如：26
    second = dt.strftime('%S')  # 秒，例如：15
    time = '{}_{}_{}_{}_{}_{}'.format(year, month, day, hour, minute,second)
    return time
# 对于视觉任务不考虑enc_in
# 两个capture感觉区别不大啊，直接使用其中一个跑跑，事后再使用另一个跑
class Dataset_tunel_kv(Dataset):
    def __init__(self, flag='train', size=48,
                 enc_in=5, scale=True, inverse=False, cols=None, task='tunel_kv', args=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size
        #self.seq_lenv = int(self.seq_len**(0.5))
        self.seq_lenv = args.seq_lenv
        self.args = args
        self.enc_in = enc_in
        self.task = task

        # init
        self.flag = flag

        self.scale = scale
        self.inverse = inverse
        self.enc = LabelEncoder()
        self.scaler = StandardScaler_classification()
        self.prepare_data()
        self.__read_data__()

    def prepare_data(self):
        # 注意这里写的时候是以main_informer出发写的
        df = pd.read_pickle('data/tunel_rob/tunel_rob.pickle')
        self.image_floder_2d = '../tunel_rob/data'
        self.image_floder_3d = '../tunel_rob/data3d'
        # 存成csv时候有个问题就是会多一第一行index，所有最好用pickle

        val_list = []
        label_list = []
        timestamp_list = []
        factor = 1
        for i in range(self.seq_len, df.shape[0], factor):

            # 11是因为第12列开始才是有效数据，详情请看dataloader里面写的
            # 这里直接使用了最后一个点的gesture作为label来计算的
            val = df.iloc[i - self.seq_len:i, 2:2 + self.enc_in].to_numpy().astype('float64')
            time_stamp = df.iloc[i]['time_stamp']
            label = df.iloc[i]['action']

            val_list.append(val)
            timestamp_list.append(time_stamp)
            label_list.append(label)

        df_clean = pd.DataFrame({"value list": val_list, "label": label_list, "time_stamp": timestamp_list})
        # label更为平衡化
        df_forward = df_clean[df_clean['label'] == 'forward']
        df_back = df_clean[df_clean['label'] == 'back']
        df_right = df_clean[df_clean['label'] == 'right']
        df_left = df_clean[df_clean['label'] == 'left']
        l_list = [len(df_forward), len(df_back), len(df_right), len(df_left)]
        min_l = min(l_list)

        df_list = [df_forward[:min_l], df_back[:min_l], df_right[:min_l], df_left[:min_l]]
        self.x_data_trn = pd.concat(df_list, ignore_index=True)
        self.enc.fit(self.x_data_trn['label'])
        self.class_name = self.enc.inverse_transform([i for i in range(len(self.x_data_trn['label'].unique()))])

    def __read_data__(self):
        # train val test 0.8:0.1:0.1
        num_fold = 10
        df_kflod = get_fold(self.x_data_trn, num_fold, 'label')
        x_train = df_kflod.loc[(df_kflod['fold'] <= 7)]
        x_test = df_kflod.loc[(df_kflod['fold'] == 8)]
        x_val = df_kflod.loc[(df_kflod['fold'] == 9)]
        y_train = self.enc.transform(x_train['label'])
        y_test = self.enc.transform(x_test['label'])
        y_val = self.enc.transform(x_val['label'])
        # # vt:val&test
        # x_train, x_vt, y_train, y_vt = train_test_split(self.x_data_trn, self.y_enc, test_size=0.2)
        # x_val, x_test, y_val, y_test = train_test_split(x_vt, y_vt, test_size=0.5)

        # 在这里先没有考虑seq_len,对于这个数据集来说最长是128
        if self.flag == 'train':
            self.data_x = x_train
            self.data_y = y_train
            self.ds_len = len(y_train)
        elif self.flag == 'val':
            self.data_x = x_val
            self.data_y = y_val
            self.ds_len = len(y_val)
        elif self.flag == 'test' or self.flag == 'pred':
            self.data_x = x_test
            self.data_y = y_test
            self.ds_len = len(y_test)

        end = 1

    # 最后输出分别是data,label,以及time_stamp的data和label
    def __getitem__(self, index):
        imgs = None
        if self.task == 'tunel_kv' or self.task == 'tunel_v':
            for i in range(self.seq_lenv):
                time_stamp = change_time(self.data_x.iloc[index]['time_stamp'], -self.seq_lenv+i)
                img = np.array(Image.open('{}/{}.jpg'.format(self.image_floder_3d, time_stamp)))
                img = img / 255
                if self.flag == 'train':
                    img = transform_test(img)
                elif self.flag == 'val':
                    img = transform_test(img)
                elif self.flag == 'test' or self.flag == 'pred':
                    img = transform_test(img)
                img.permute(2, 0, 1)
                img = img.to(torch.float)
                img = img.unsqueeze(0)
                if imgs is None:
                    imgs = img
                else:
                    imgs = torch.cat((imgs, img), dim=0)
            if self.args.dual_img:
                for i in range(self.seq_lenv):
                    time_stamp = change_time(self.data_x.iloc[index]['time_stamp'], -self.seq_lenv + i)
                    img = np.array(Image.open('{}/{}.jpg'.format(self.image_floder_3d, time_stamp)))
                    img = img / 255
                    if self.flag == 'train':
                        img = transform_test(img)
                    elif self.flag == 'val':
                        img = transform_test(img)
                    elif self.flag == 'test' or self.flag == 'pred':
                        img = transform_test(img)
                    img.permute(2, 0, 1)
                    img = img.to(torch.float)
                    img = img.unsqueeze(0)
                    imgs = torch.cat((imgs, img), dim=0)

        if self.task == 'tunel_kv' or self.task == 'tunel_k':
            data = self.data_x['value list'].iloc[index].astype('float64')
        # 原本是个裸露的int，为了使用crossentropy需要改变一下
        label = torch.tensor(self.data_y[index]).to(torch.long)
        if self.task == 'tunel_kv':
            return [imgs, data], label
        elif self.task == 'tunel_v':
            return imgs, label
        elif self.task == 'tunel_k':
            return data, label

    def __len__(self):
        return self.ds_len

    def get_encoder(self):
        return self.enc

    def get_class_name(self):
        return self.class_name