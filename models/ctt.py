import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.classification_head import ClassificationHead

"""
    这里c_out决定了输出是多少
    这里我理解的是，如果是MS的话是所有的来去预测OT这项指标，不然的话就是多对多分别去预测
"""


class ctt(nn.Module):
    def __init__(self, enc_in, c_out, seq_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), num_classes=1):
        super(ctt, self).__init__()

        # attn这里是选择不同的attention，一共有两种一种是普通的attention一种是prob attention
        self.attn = attn
        # 是否输出encoder的注意力
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder ，distil true的时候才有卷积,传入是三个list
        # encoder的结构是encoder layer，conv layer，之后是layer norm
        # 卷积层数目比encoder数目少一
        # 每次执行conv时候seq_len减半
        # attention_layer的四个输入attention,d_model,n_head,mix
        # attention块的输入
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

    def forward(self, x_enc, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)
        # 这里是为了输出注意力图像，参见colab里面那个
        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out