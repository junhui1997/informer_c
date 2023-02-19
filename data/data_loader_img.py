import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import get_fold
# from sklearn.preprocessing import StandardScaler
from PIL import Image

from utils.tools import StandardScaler
from data.transform_list import transform_train,transform_test


import warnings
warnings.filterwarnings('ignore')

# 对于视觉任务不考虑enc_in
# 两个capture感觉区别不大啊，直接使用其中一个跑跑，事后再使用另一个跑
class Dataset_jigsaw_gv(Dataset):
    def __init__(self, flag='train', size=48,
                 enc_in=5, scale=True, inverse=False, cols=None,task='jigsaw_kt_g'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size
        self.seq_len2 = self.seq_len**2
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
        if self.task == 'jigsaw_kt_gv':
            df = pd.read_pickle(folder+'Knot_Tying.pkl')
            self.image_floder = '../jigsaw/video_slice/Knot_Tying/'
        elif self.task == 'jigsaw_np_gv':
            df = pd.read_pickle(folder+'Needle_Passing.pkl')
            self.image_floder = '../jigsaw/video_slice/Needle_Passing/'
        elif self.task == 'jigsaw_su_gv':
            df = pd.read_pickle(folder+'Suturing.pkl')
            self.image_floder = '../jigsaw/video_slice/Suturing/'
        val_list = []
        label_list = []
        frame_list = []
        filename_l = []
        #四分之一的采样率
        for i in range(self.seq_len, df.shape[0], 4):
            if df.iloc[i - self.seq_len]['file_name'] != df.iloc[i]['file_name']:
                continue
            # 11是因为第12列开始才是有效数据，详情请看dataloader里面写的
            # 这里直接使用了最后一个点的gesture作为label来计算的
            val = df.iloc[i - self.seq_len:i, 11:11 + self.enc_in].to_numpy()
            frame = df.iloc[i]['frame']
            filename = df.iloc[i]['file_name'].split('.')[0]
            label = df.iloc[i]['gesture']

            val_list.append(val)
            frame_list.append(frame)
            filename_l.append(filename)
            label_list.append(label)
        self.x_data_trn = pd.DataFrame({"value list": val_list, "gesture": label_list, "file_name": filename_l, "frame": frame_list})
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
        imgs = None
        for i in range(self.seq_len):
            file_name = "{}_capture1_frame_{}".format(self.data_x.iloc[index]['file_name'], int(self.data_x.iloc[index]['frame'])-self.seq_len+i)
            img = np.array(Image.open('{}/{}.jpg'.format(self.image_floder, file_name)))
            img = img / 255
            if self.flag == 'train':
                img = transform_train(img)
            elif self.flag == 'val':
                img = transform_test(img)
            elif self.flag == 'test':
                img = transform_test(img)
            img.permute(2, 0, 1)
            img = img.to(torch.float)
            img = img.unsqueeze(0)
            if imgs is None:
                imgs = img
            else:
                imgs = torch.cat((imgs, img), dim=0)

        #data = self.data_x['value list'].iloc[index].astype('float64')
        # 原本是个裸露的int，为了使用crossentropy需要改变一下
        label = torch.tensor(self.data_y[index]).to(torch.long)
        return imgs, label

    def __len__(self):
        return self.ds_len

    def get_encoder(self):
        return self.enc

    def get_class_name(self):
        return self.class_name