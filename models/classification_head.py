import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, n_classes)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.norm(x)
        x = self.linear(x)
        return x