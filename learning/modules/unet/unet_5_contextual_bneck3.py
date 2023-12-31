import torch
from torch.autograd import Variable
from torch import nn as nn
import torch.nn.functional as F


class DoubleConv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(cin, cin, k, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(cin, cout, k, stride=1, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img):
        x = self.conv1(img)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        return x


class DoubleDeconv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(DoubleDeconv, self).__init__()
        self.conv1 = nn.ConvTranspose2d(cin, cout, k, stride=1, padding=padding)
        self.conv2 = nn.ConvTranspose2d(cout, cout, k, stride=stride, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img, output_size):
        # TODO: 2 is stride
        osize1 = [int(i/2) for i in output_size]
        x = self.conv1(img, output_size=osize1)
        x = F.leaky_relu(x)
        x = self.conv2(x, output_size=output_size)
        return x


class Unet5ContextualBneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embedding_size, hc1=32, hb1=16, hc2=256, stride=2, split_embedding=False):
        super(Unet5ContextualBneck, self).__init__()

        self.hc1 = hc1
        self.hb1 = hb1
        self.hc2 = hc2

        self.split_embedding = split_embedding

        self.embedding_size = embedding_size
        if split_embedding:
            self.emb_block_size = int(embedding_size / 5)
        else:
            self.emb_block_size = embedding_size

        # inchannels, outchannels, kernel size
        self.conv1 = DoubleConv(in_channels, hc1, 3, stride=stride, padding=1)
        self.conv2 = DoubleConv(hc1, hc1, 3, stride=stride, padding=1)
        self.conv3 = DoubleConv(hc1, hc1, 3, stride=stride, padding=1)
        self.conv4 = DoubleConv(hc1, hc1, 3, stride=stride, padding=1)
        self.conv5 = DoubleConv(hc1, hc1, 3, stride=stride, padding=1)

        self.deconv1 = DoubleDeconv(hc1, hc1, 3, stride=stride, padding=1)
        self.deconv2 = DoubleDeconv(hc1 + hb1, hc1, 3, stride=stride, padding=1)
        self.deconv3 = DoubleDeconv(hc1 + hb1, hc1, 3, stride=stride, padding=1)
        self.deconv4 = DoubleDeconv(hc1 + hb1, hc2, 3, stride=stride, padding=1)
        self.deconv5 = nn.ConvTranspose2d(hb1 + hc2, out_channels, 3, stride=stride, padding=1)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(hc1)
        self.norm3 = nn.InstanceNorm2d(hc1)
        self.norm4 = nn.InstanceNorm2d(hc1)
        self.norm5 = nn.InstanceNorm2d(hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(hc1)
        self.dnorm3 = nn.InstanceNorm2d(hc1)
        self.dnorm4 = nn.InstanceNorm2d(hc1)
        self.dnorm5 = nn.InstanceNorm2d(hc2)

        self.fnorm1 = nn.InstanceNorm2d(hb1)
        self.fnorm2 = nn.InstanceNorm2d(hb1)
        self.fnorm3 = nn.InstanceNorm2d(hb1)
        self.fnorm4 = nn.InstanceNorm2d(hb1)

        self.lang19 = nn.Linear(self.emb_block_size, hc1 * hb1)
        self.lang28 = nn.Linear(self.emb_block_size, hc1 * hb1)
        self.lang37 = nn.Linear(self.emb_block_size, hc1 * hb1)
        self.lang46 = nn.Linear(self.emb_block_size, hc1 * hb1)
        self.lang55 = nn.Linear(self.emb_block_size, hc1 * hc1)

    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        self.conv3.init_weights()
        self.conv4.init_weights()
        self.conv5.init_weights()
        self.deconv1.init_weights()
        self.deconv2.init_weights()
        self.deconv3.init_weights()
        self.deconv4.init_weights()
        #self.deconv5.init_weights()

    def forward(self, input, embedding):
        x1 = self.norm2(self.act(self.conv1(input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if embedding is not None:
            embedding = F.normalize(embedding, p=2, dim=1)

            if self.split_embedding:
                block_size = self.emb_block_size
                emb1 = embedding[:, 0*block_size:1*block_size]
                emb2 = embedding[:, 1*block_size:2*block_size]
                emb3 = embedding[:, 2*block_size:3*block_size]
                emb4 = embedding[:, 3*block_size:4*block_size]
                emb5 = embedding[:, 4*block_size:5*block_size]
            else:
                emb1 = emb2 = emb3 = emb4 = emb5 = embedding

            # These conv filters are different for each element in the batch, but the functional convolution
            # operator assumes the same filters across the batch.
            # TODO: Verify if slicing like this is a terrible idea for performance
            x1f = Variable(torch.zeros_like(x1[:,0:self.hb1,:,:].data))
            x2f = Variable(torch.zeros_like(x2[:,0:self.hb1,:,:].data))
            x3f = Variable(torch.zeros_like(x3[:,0:self.hb1,:,:].data))
            x4f = Variable(torch.zeros_like(x4[:,0:self.hb1,:,:].data))
            x5f = Variable(torch.zeros_like(x5.data))

            batch_size = embedding.size(0)
            for i in range(batch_size):
                lf1 = F.normalize(self.lang19(emb1[i:i+1])).view([self.hb1, self.hc1, 1, 1])
                lf2 = F.normalize(self.lang28(emb2[i:i+1])).view([self.hb1, self.hc1, 1, 1])
                lf3 = F.normalize(self.lang37(emb3[i:i+1])).view([self.hb1, self.hc1, 1, 1])
                lf4 = F.normalize(self.lang46(emb4[i:i+1])).view([self.hb1, self.hc1, 1, 1])
                lf5 = F.normalize(self.lang55(emb5[i:i+1])).view([self.hc1, self.hc1, 1, 1])

                x1f[i:i+1] = F.conv2d(x1[i:i+1], lf1)
                x2f[i:i+1] = F.conv2d(x2[i:i+1], lf2)
                x3f[i:i+1] = F.conv2d(x3[i:i+1], lf3)
                x4f[i:i+1] = F.conv2d(x4[i:i+1], lf4)
                x5f[i:i+1] = F.conv2d(x5[i:i+1], lf5)

            x1 = self.fnorm1(x1f)
            x2 = self.fnorm2(x2f)
            x3 = self.fnorm3(x3f)
            x4 = self.fnorm4(x4f)
            x5 = x5f

        x6 = self.act(self.deconv1(x5, output_size=x4.size()))
        x46 = torch.cat([x4, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1, x9], 1)
        out = self.deconv5(x19, output_size=input.size())

        return out