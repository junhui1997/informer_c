from data.data_loader import Dataset_rob,Dataset_jigsaw,Dataset_jigsaw_g
from data.data_loader_tunel import Dataset_tunel_kv
from data.data_loader_img import Dataset_jigsaw_gv
from data.data_loader_imgk import Dataset_jigsaw_gvk
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from models.ctt import ctt
from models.ctt_kv import ctt_kv
from models.conv_lstm import conv_lstm
from models.tcn import tcn
from module_box.all_loss import focal_loss
from module_box.edit_score_dist import calculate_edit_score
from module_box.all_plot import plot_color_code
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import seaborn as sns
import pickle

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary as summary_t
from torchinfo import summary as summary_info
from torch import autograd
import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        # 从exp_basic里面继承了init的函数，在那里面执行了_build_model
        # 获取device也是这里

    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
            'ctt': ctt,
            'ctt_kv': ctt_kv,
            'conv_lstm': conv_lstm,
            'tcn': tcn

        }
        if self.args.model in model_dict.keys():
            # s_layer的作用是？普通的e_layer是encoder layer的数目，而对于s来说是3,2,1这样的话是怎么看呢，number of stack encoder layer
            # 这一行用在了stack里面，之后有需求再看
            # e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            e_layers = self.args.e_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.c_out, 
                self.args.seq_len,
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device,
                self.args.num_classes,
                self.args
            ).float()


        ##在这里model已经完成了实例化,batch_size是有一个输入的参数的
        if self.args.model=='informer' and self.args.show_para:
            summary_info(
                model,
                input_size=(self.args.batch_size, self.args.seq_len,self.args.enc_in),
                col_names=["output_size", "num_params"],
            )
        elif self.args.model=='ctt' and self.args.show_para:
            summary_info(
                model,
                input_size=(self.args.batch_size, self.args.seq_len, 3, 224, 244),
                col_names=["output_size", "num_params"],
            )

        # to device写在了exp basic里面去了
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'navi_rob':Dataset_rob,
            'jigsaw_kt':Dataset_jigsaw,
            'jigsaw_np':Dataset_jigsaw,
            'jigsaw_su':Dataset_jigsaw,
            'jigsaw_kt_g': Dataset_jigsaw_g,
            'jigsaw_np_g': Dataset_jigsaw_g,
            'jigsaw_su_g': Dataset_jigsaw_g,
            'jigsaw_kt_gv': Dataset_jigsaw_gv,
            'jigsaw_np_gv': Dataset_jigsaw_gv,
            'jigsaw_su_gv': Dataset_jigsaw_gv,
            'jigsaw_kt_gvk': Dataset_jigsaw_gvk,
            'jigsaw_np_gvk': Dataset_jigsaw_gvk,
            'jigsaw_su_gvk': Dataset_jigsaw_gvk,
            'tunel_kv': Dataset_tunel_kv,
            'tunel_v': Dataset_tunel_kv,
            'tunel_k': Dataset_tunel_kv,

        }
        # 注意data这里还没有完成实例化，完成实例化之后再调用instance
        Data = data_dict[self.args.task]



        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size;
        elif flag=='pred':
            shuffle_flag = True; drop_last = False; batch_size = 1;
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size;
        data_set = Data(
            flag=flag,
            size=args.seq_len,
            enc_in=args.enc_in,
            inverse=args.inverse,
            cols=args.cols,
            task=args.task,
            args=args
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        self.enc = data_set.get_encoder()
        self.class_name = data_set.get_class_name()
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(model_optim, 'min', patience=2)
        return model_optim
    def _select_scheduler(self,optimizer):
        if self.args.lradj == 'type4':
            # patientce = 2,代表的是3次val 没有下降后开始降低learning rate
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=1)
        else:
            scheduler = None
        return scheduler
    def _select_criterion(self):
        if self.args.loss == 'norm':
            criterion = nn.CrossEntropyLoss()
        elif self.args.loss == 'focal':
            criterion = focal_loss()
        return criterion

    #计算val_loss
    def vali(self, vali_loader, criterion):
        #首先设置的是eval模式
        self.model.eval()
        total_loss = []
        correct, total = 0, 0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                pred = self._process_one_batch(batch_x)
                true = batch_y.to(self.device)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)
                # calculate accy
                pred_idx = F.log_softmax(pred, dim=1).argmax(dim=1)
                total += true.size(0)  # 统计了batch_size
                correct += (pred_idx == true).sum().item()

        total_loss = np.average(total_loss)
        acc = correct / total
        self.model.train()
        return total_loss, acc

    # 计算分类里面准确率
    def vali_accuracy(self, vali_loader):
        #首先设置的是eval模式
        self.model.eval()
        correct, total = 0, 0
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            pred = self._process_one_batch(batch_x)
            pred = F.log_softmax(pred, dim=1).argmax(dim=1)
            true = batch_y.to(self.device)
            total += true.size(0) # 统计了batch_size
            correct += (pred == true).sum().item()
        acc = correct / total
        self.model.train()
        return acc

    def train(self, setting):
        #这里相当于每一个都实例化了一次dataset和dataloader，这样如果dataset里面本身的要处理的信息就比较多的话会导致说重复创建牺牲很多性能
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        #test_data, test_loader = vali_data, vali_loader
        test_data, test_loader = self._get_data(flag='test')
        self.args.iterations_per_epoch = len(train_loader) # 这一项是为了控制cos_learningRate

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim)
        criterion = self._select_criterion()
        self.all_train_loss = []
        self.all_val_loss = []
        self.all_test_loss =[]
        self.all_accuracy = []
        best_accuracy = 0

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1

                # 需要每一个batch时候zero_grad！
                model_optim.zero_grad()
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None and torch.isnan(param.grad).any():
                #         print("nan gradient found")
                #         print("name:", name)

                #torch.autograd.set_detect_anomaly(True)


                pred = self._process_one_batch(batch_x)
                true = batch_y.to(self.device)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                # 每100个iter重新更新一次预估时间
                if (i+1) % 200==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    # 从开始到现在的时间/经历了多少iteration
                    speed = (time.time()-time_now)/iter_count
                    # train_step反映了一个epoch有多少个iter
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # with autograd.detect_anomaly():
                    #     loss.backward()
                    #     model_optim.step()
                    loss.backward()
                    model_optim.step()
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None and torch.isnan(param.grad).any():
                    #         print("nan gradient found")
                    #         print("name:", name)
                    a = 1

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, current_accuray = self.vali(vali_loader, criterion)
            test_loss, _ = self.vali(test_loader, criterion)
            self.all_train_loss.append(train_loss)
            self.all_val_loss.append(vali_loss)
            self.all_test_loss.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            self.all_accuracy.append(current_accuray)
            print('current accuracy in Epoch: {0} is {1:.7f}'.format(epoch + 1, current_accuray))
            if current_accuray > best_accuracy:
                best_accuracy = current_accuray

            # 根据val loss来储存判断是否储存当前模型，test_loss是纯未知量，因为模型是根据val里面来修正训练的
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args, scheduler, vali_loss)

        # 最后时候给加载上去,self.model最后保存了最好的结果
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        print('best accuracy is ', best_accuracy)
        self.val_acc = best_accuracy
        
        return self.model

    # 在test里面将想输出的结果给输出出来
    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        predict_list = []
        label_list = []
        correct, total = 0, 0
        for i, (batch_x, batch_y) in enumerate(test_loader):
            pred = self._process_one_batch(batch_x)
            pred = F.log_softmax(pred, dim=1).argmax(dim=1)
            true = batch_y.to(self.device)
            total += true.size(0)  # 统计了batch_size
            correct += (pred == true).sum().item()
            predict_list += pred.tolist()
            label_list += batch_y.tolist()
        acc = correct / total
        predict_encoder = self.enc.inverse_transform(predict_list)
        print('predict label', predict_encoder[0:10])
        label_encoder = self.enc.inverse_transform(label_list)
        print('true label', label_encoder[0:10])
        cf_matrix = confusion_matrix(predict_encoder, label_encoder)
        df = pd.DataFrame(cf_matrix, index=self.class_name, columns=self.class_name)

        #calculate edit score and distance
        edit_distance, edit_score = calculate_edit_score(predict_list, label_list)


        # result save
        folder_path = './results/' + setting +'acc{0:0.4f}'.format(acc)+'/'
        self.folder_path = folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('test accuracy is ', acc)
        # save confusion matrix and the heat map
        # 是为了避免实验重复时候出现的图层重叠
        df.to_pickle(folder_path+'confusion_m.csv')
        sns.heatmap(df, annot=True, cbar=None, cmap="YlGnBu", fmt="d")
        plt.title("Confusion Matrix"), plt.tight_layout()
        plt.xlabel("True Class"),
        plt.ylabel("Predicted Class")
        plt.savefig(folder_path+'confusion_matrix.png')
        plt.clf()

        #save edit score and distance
        # 打开一个文本文件，如果不存在则创建它
        with open(folder_path+'output.txt', 'w') as f:
            # 将print语句的输出写入文件中
            print("Edit distance:", edit_distance, file=f)
            print("Edit score:", edit_score, file=f)
            print("val acc:", self.val_acc, file=f)
            print("test acc:", acc, file=f)

        # save accuracy plot for each class
        confusion_np = df.values
        TP = np.diag(confusion_np)
        FN = confusion_np.sum(axis=1) - TP
        FP = confusion_np.sum(axis=0) - TP
        TN = confusion_np.sum() - (TP + FN + FP)
        # 计算每个类别的准确率
        accuracy = TP / (TP + FP)
        # 创建条形图
        plt.bar(range(len(accuracy)), accuracy)
        plt.xticks(range(len(accuracy)), list(df.columns))
        plt.ylabel('Accuracy')
        plt.xlabel('True label')
        plt.title(f'Confusion matrix with accuracy {acc:.4f}')
        plt.savefig(folder_path + 'each_accuracy.jpg')
        plt.clf()


        # save loss plot
        # 避免有时候被early stopping掉
        x = [i+1 for i in range(len(self.all_train_loss))]
        plt.plot(x, self.all_train_loss, label='train loss')
        plt.plot(x, self.all_val_loss, label='val loss')
        plt.plot(x, self.all_test_loss, label='test loss')
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(folder_path+'loss.jpg')
        plt.clf()

        # save accuracy plot
        plt.plot(x, self.all_accuracy, label='val accuracy')
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('accuracy %')
        plt.savefig(folder_path + 'accuracy.jpg')
        plt.clf()

        return

    # setting是在外面生成的
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')


        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path)).to(self.device)

        self.model.eval()

        predict_list = []
        label_list = []
        correct, total = 0, 0
        for i, (batch_x, batch_y) in enumerate(pred_loader):
            pred = self._process_one_batch(batch_x)
            pred = F.log_softmax(pred, dim=1).argmax(dim=1)
            true = batch_y.to(self.device)
            total += true.size(0)  # 统计了batch_size
            correct += (pred == true).sum().item()
            predict_list += pred.tolist()
            label_list += batch_y.tolist()


        
        # result save
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        with open(self.folder_path+'predict.pkl', 'wb') as f:
            pickle.dump(predict_list, f)

        with open(self.folder_path+'label.pkl', 'wb') as f:
            pickle.dump(label_list, f)

        plot_color_code(predict_list, self.folder_path, 'predict_color_code', self.args.num_classes)
        plot_color_code(label_list, self.folder_path, 'label_color_code', self.args.num_classes)
        
        return

    # 这里只输出predict的label
    # return: [batch_size,num_classes]
    def _process_one_batch(self, batch_x):
        #已知batch代表的是真实的数值
        if type(batch_x) == list:
            batch_x[0] = batch_x[0].float().to(self.device)
            batch_x[1] = batch_x[1].float().to(self.device)
            #print(type(batch_x[0]))
        else:
            batch_x = batch_x.float().to(self.device)

        # 混合精度训练
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                #输出注意力机制
                if self.args.output_attention:
                    #预测的时候同样给输入进去即可，注意timestamp_y也是直接输入进去即可
                    outputs = self.model(batch_x)[0]
                else:
                    outputs = self.model(batch_x)
        else:
            #如果使用了输出注意力机制的话，那么输出的结果会比较多这样的话需要选择第一个即可
            if self.args.output_attention:
                outputs = self.model(batch_x)[0]
            else:
                outputs = self.model(batch_x)

        return outputs
