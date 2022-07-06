import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)

# class AAEmbed(nn.Module):
#     """simple convolutional transformation"""
#     def __init__(self,num_input_channels=1024,num_target_channels=128):
#         super(AAEmbed, self).__init__()
#         self.num_input_channels = num_input_channels
#         self.num_target_channels = num_target_channels
#         self.num_mid_channels = self.num_target_channels * 2
#
#         def con1x1(in_channels,out_channels,stride=1):
#             return nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0,stride=stride,bias=False)
#         def con3x3(in_channels,out_channels,stride=1):
#             return nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=0,stride=stride,bias=False)
#
#         self.regressor = nn.Sequential(
#             con1x1(self.num_input_channels,self.num_mid_channels),
#             nn.BatchNorm2d(self.num_mid_channels),
#             nn.ReLU(inplace=True),
#             con3x3(self.num_mid_channels,self.num_mid_channels),
#             nn.BatchNorm2d(self.num_mid_channels),
#             nn.ReLU(inplace=True),
#             con1x1(self.num_mid_channels,self.num_target_channels),
#         )
#
#     def forward(self,x):
#         x = self.regressor(x)
#         return x

def get_feat_shapes(student, teacher, input_size):
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        _, feat_s = student(data)
        _, feat_t = teacher(data)
    feat_s_shapes = [f.shape for f in feat_s["feats"]]
    feat_t_shapes = [f.shape for f in feat_t["feats"]]
    return feat_s_shapes, feat_t_shapes
