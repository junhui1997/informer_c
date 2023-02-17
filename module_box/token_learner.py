import torch
import torch.nn as nn



# input:[b,c,h,w]
# out: [b,c], out
# x*weight，经过注意力层遮罩后的输出
class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.sgap = nn.AvgPool2d(2)

    def forward(self, x):
        # mx:[b,1,h,w]
        # avg:[b,1,h,w]
        # combiner:[b,2,h,w]
        # fmap:[b,1,h,w],同时利用了最大值和平均值信息，将两通道数据1*1卷积为单通道数据
        # weight:[b,1,h,w]，就是scale
        mx = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        combined = torch.cat([mx, avg], dim=1)
        fmap = self.conv(combined)
        weight_map = torch.sigmoid(fmap)
        out = (x * weight_map).mean(dim=(-2, -1))

        return out, x * weight_map


class token_learner(nn.Module):
    def __init__(self, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = nn.ModuleList([SpatialAttention() for _ in range(S)])

    def forward(self, x):
        B, C, _, _ = x.shape
        Z = torch.Tensor(B, self.S, C)
        for i in range(self.S):
            Ai, _ = self.tokenizers[i](x)  # [B, C]
            Z[:, i, :] = Ai
        return Z