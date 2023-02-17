import argparse
import os
import torch

from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--task', type=str, required=True, default='navi_rob', help='data')


parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--show_para',  type=int, default=0, help='show model parameter and calculate the size of memory')


# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]') #注意这里默认就是timefix
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]


data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'navi_rob':{'folder':'./data/navi_rob/','enc_in':10,'c_out':1,'num_classes':10},
    # 熟练度分类，只使用了运动学数据
    'jigsaw_kt':{'folder':'./data/jigsaw/','enc_in':10,'c_out':1,'num_classes':3},
    'jigsaw_np':{'folder':'./data/jigsaw/','enc_in':10,'c_out':1,'num_classes':3},
    'jigsaw_su':{'folder':'./data/jigsaw/','enc_in':10,'c_out':1,'num_classes':3},
    # gesture分类问题，只使用了运动学数据
    'jigsaw_kt_g':{'folder':'./data/jigsaw/','enc_in':10,'c_out':1,'num_classes':11},
    'jigsaw_np_g':{'folder':'./data/jigsaw/','enc_in':10,'c_out':1,'num_classes':11},
    'jigsaw_su_g':{'folder':'./data/jigsaw/','enc_in':10,'c_out':1,'num_classes':11},
    # gesture分类问题，只使用了图像数据
    # 对于视觉来说只有一个维度，不像运动学数据可能会同时用好几个的
    'jigsaw_kt_gv':{'folder':'./data/jigsaw/','enc_in':1,'c_out':1,'num_classes':11},
    'jigsaw_np_gv':{'folder':'./data/jigsaw/','enc_in':1,'c_out':1,'num_classes':11},
    'jigsaw_su_gv':{'folder':'./data/jigsaw/','enc_in':1,'c_out':1,'num_classes':11},
    # 同时使用了运动学和图像数据
    'jigsaw_kt_gvk':{'folder':'./data/jigsaw/','enc_in':10,'c_out':1,'num_classes':11},
    'jigsaw_np_gvk':{'folder':'./data/jigsaw/','enc_in':10,'c_out':1,'num_classes':11},
    'jigsaw_su_gvk':{'folder':'./data/jigsaw/','enc_in':10,'c_out':1,'num_classes':11},

}

if args.task in data_parser.keys():
    data_info = data_parser[args.task]
    args.folder_path = data_info['folder']
    args.enc_in = data_info['enc_in'] #这里是dimension不是seq_len,利用七个维度来进行预测
    args.c_out = data_info['c_out'] #c_out没有用，原本的用处是用来决定输出的维度
    args.num_classes = data_info['num_classes']
else:
    print('there is no such a predefined task')

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]


print('Args in experiment:')
print(args)

Exp = Exp_Informer

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_sl{}_dm{}_nh{}_el{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}_epo{}'.format(args.model, args.task,
                args.seq_len,
                args.d_model, args.n_heads, args.e_layers, args.d_ff, args.attn, args.factor,
                args.embed, args.distil, args.mix, args.des, ii, args.train_epochs)

    exp = Exp(args) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    # if args.do_predict:
    #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #     exp.predict(setting, True)

    torch.cuda.empty_cache()
