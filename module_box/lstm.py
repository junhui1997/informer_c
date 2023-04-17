import torch
import torch.nn as nn

"""
input:【batch_size,seq_len,dim]
input dim:就是dim
hidden_dim:输出线性层中的尺寸
layer_dim:lstm层数
output_dim:分类的个数
"""
class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs
    # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = 64
        self.layer_dim = layer_dim
        # 注意这里设置了batch_first所以第一个维度是batch，lstm第二个input是输出的维度，第三个是lstm的层数
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.hidden_dim2)
        self.fc2 = nn.Linear(self.hidden_dim2, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        # init_hidden并不是魔法函数，是每次循环时候手动执行更新的
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # (N, L, D * H_{out})(N,L,D∗H_out) D代表的是direction，如果是双向lstm的话则d为2 else 1，L代表的是sequence
        # 因为我们预测的是这个sequence结尾时候的数值，所以是-1.最后一个维度H_out代表的是projection的维度
        out = self.fc(out[:, -1, :])
        out = self.fc2(out)
        return out

    def init_hidden(self, x):
        # (lstm层的个数，batch_size,输出层的个数)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]