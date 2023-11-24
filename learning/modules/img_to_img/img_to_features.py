import torch
from torch import nn as nn

from learning.modules.resnet.resnet_13_s import ResNet13S
from learning.modules.cuda_module import CudaModule

class ImgToFeatures(CudaModule):
    def __init__(self, channels, out_channels, img_w, img_h):
        super(ImgToFeatures, self).__init__()
        self.feature_net = ResNet13S(channels, down_pad=True)
        self.do_down_conv = channels != out_channels
        self.conv_out = nn.Conv2d(channels, out_channels, 1, 1, 1)
        self.act = nn.LeakyReLU()

    def cuda(self, device=None):
        CudaModule.cuda(self, device)
        self.feature_net.cuda()

    def init_weights(self):
        self.feature_net.init_weights()

        torch.nn.init.kaiming_uniform_(self.conv_out.weight)
        self.conv_out.bias.data.fill_(0)

    def forward(self, input):
        x = self.feature_net(input)
        if self.do_down_conv:
            x = self.act(self.conv_out(x))
        return x