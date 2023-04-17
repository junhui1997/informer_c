import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.classification_head import ClassificationHead
from module_box.feature_extraction import cnn_feature, cnn_feature50
from module_box.lstm import LSTMClassifier

"""
    这里c_out决定了输出是多少
    这里我理解的是，如果是MS的话是所有的来去预测OT这项指标，不然的话就是多对多分别去预测
"""


class conv_lstm(nn.Module):
    def __init__(self, enc_in, c_out, seq_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), num_classes=1, args=None):
        super(conv_lstm, self).__init__()

        self.args = args
        if args.dual_img:
            seq_lenv = args.seq_lenv*2
        self.cnn_features = nn.ModuleList([cnn_feature(feature_type='linear', grad=True) for _ in range(seq_lenv)])
        self.cnn_dim = 1000
        self.device = device
        self.output_attention = output_attention

        self.lstm_c = LSTMClassifier(input_dim=self.cnn_dim, hidden_dim=256, layer_dim=3, output_dim=num_classes)


    def forward(self, x_enc, enc_self_mask=None):
        batch_size, seq_len, _, _, _ = x_enc.shape
        x_features = torch.Tensor(batch_size, seq_len, self.cnn_dim).to(self.device)
        for i in range(seq_len):
            # x_feature在token learner之后是[batch_size,self.s,512]
            x_feature = self.cnn_features[i](x_enc[:, i, :, :, :])
            x_features[:, i, :] = x_feature

        out = self.lstm_c(x_features)
        attns = 0
        if self.output_attention:
            return out, attns
        else:
            return out