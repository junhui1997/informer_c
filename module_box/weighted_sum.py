import torch
import torch.nn as nn


class WeightedSum(nn.Module):
    def __init__(self, seq_len, d_model):
        super(WeightedSum, self).__init__()
        self.weights = nn.Parameter(torch.rand(1, seq_len, d_model))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, tensor1, tensor2):
        # 归一化权重
        normalized_weights = torch.softmax(self.weights, dim=1)

        # 计算加权和
        weighted_sum = normalized_weights * tensor1 + (1 - normalized_weights) * tensor2
        return weighted_sum


