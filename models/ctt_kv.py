import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.classification_head import ClassificationHead
from module_box.feature_extraction import cnn_feature,cnn_feature50
from module_box.token_learner import token_learner
from module_box.weighted_sum import WeightedSum
"""
    这里c_out决定了输出是多少
    这里我理解的是，如果是MS的话是所有的来去预测OT这项指标，不然的话就是多对多分别去预测
"""


class ctt_kv(nn.Module):
    def __init__(self, enc_in, c_out, seq_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), num_classes=1, args=None):
        super(ctt_kv, self).__init__()

        self.s = args.s
        self.args =args
        self.cnn_feature = cnn_feature(grad=True)
        #self.cnn_feature = cnn_feature50(grad=True)
        self.token_learner = token_learner(S=self.s)
        # for mutiple input
        if args.dual_img:
            seq_lenv = args.seq_lenv*2
            self.weighted_sum = WeightedSum(self.args.seq_lenv*self.s, d_model)
        self.cnn_features = nn.ModuleList([cnn_feature(grad=True) for _ in range(seq_lenv)])
        self.token_learners = nn.ModuleList([token_learner(S=self.s) for _ in range(seq_lenv)])
        self.device = device

        # attn这里是选择不同的attention，一共有两种一种是普通的attention一种是prob attention
        self.attn = attn
        # 是否输出encoder的注意力
        self.output_attention = output_attention

        # Encoding
        self.kine_enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.enc_embedding = DataEmbedding(512, d_model, embed, dropout)

        # Attention
        Attnk = ProbAttention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder ，distil true的时候才有卷积,传入是三个list
        # encoder的结构是encoder layer，conv layer，之后是layer norm
        # 卷积层数目比encoder数目少一
        # 每次执行conv时候seq_len减半
        # attention_layer的四个输入attention,d_model,n_head,mix
        # attention块的输入
        # 固定先使用了四层kine
        self.kine_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attnk(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(4 - 1)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        # 最后通过nn.linear输出成想要的形状
        self.projection = ClassificationHead(d_model, num_classes)

        self.new_fc = torch.nn.Linear(512,num_classes)

    def forward(self, x, enc_self_mask=None):
        x_img, x_kine = x
        batch_size,seq_len,_,_,_ = x_img.shape
        # !!!!!非继承的tensor切记要移动到cuda中去
        x_features = torch.Tensor(batch_size, seq_len*self.s, 512).to(self.device)
        for i in range(seq_len):
            # x_feature在token learner之后是[batch_size,self.s,512]
            x_feature = self.cnn_features[i](x_img[:, i, :, :, :])
            x_feature = self.token_learners[i](x_feature)
            x_features[:, i*self.s:(i+1)*self.s, :] = x_feature
        fusion_type = 'weight'
        # fusion_type = 'direct'
        if self.args.dual_img:
            if fusion_type == 'weight':
                #x_features = x_features[:, :self.args.seq_lenv*self.s, :]+x_features[:, self.args.seq_lenv*self.s:, :]
                features_2d = x_features[:, :self.args.seq_lenv*self.s, :]
                features_3d = x_features[:, self.args.seq_lenv*self.s:, :]
                x_features = self.weighted_sum(features_2d,features_3d)
            else:
                pass
        x_kine = self.kine_enc_embedding(x_kine)
        x_kine, kine_attn = self.kine_encoder(x_kine)
        x_fu = torch.cat((x_features,x_kine), dim=1)
        x_fu = self.enc_embedding(x_fu)
        x_fu, attns = self.encoder(x_fu, attn_mask=enc_self_mask)
        x_fu = self.projection(x_fu)
        # 这里是为了输出注意力图像，参见colab里面那个
        if self.output_attention:
            return x_fu, attns
        else:
            return x_fu