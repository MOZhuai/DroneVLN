import torch
from torch.autograd import Variable
from torch import nn
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


class NestedUnet5ContextualBneck2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embedding_size, hc1=32, hb1=16, hc2=256, stride=2, split_embedding=False):
        super(NestedUnet5ContextualBneck2, self).__init__()

        self.hc1 = hc1
        self.hb1 = hb1
        self.hc2 = hc2

        self.split_embedding = split_embedding

        self.embedding_size = embedding_size
        if split_embedding:
            self.emb_block_size = int(embedding_size / 5)
        else:
            self.emb_block_size = embedding_size

        self.num_filters = [32, 64, 128, 256, 512]

        # nested unet with same number of output channels
        # inchannels, outchannels, kernel size
        # down-sampling
        self.conv0_0 = DoubleConv(in_channels, self.num_filters[0], 3, stride=stride, padding=1)
        self.conv1_0 = DoubleConv(self.num_filters[0], self.num_filters[1], 3, stride=stride, padding=1)
        self.conv2_0 = DoubleConv(self.num_filters[1], self.num_filters[2], 3, stride=stride, padding=1)
        self.conv3_0 = DoubleConv(self.num_filters[2], self.num_filters[3], 3, stride=stride, padding=1)
        self.conv4_0 = DoubleConv(self.num_filters[3], self.num_filters[4], 3, stride=stride, padding=1)

        # self.conv0_1 = DoubleConv(num_filters[0]+num_filters[1], num_filters[0], 3, stride=stride, padding=1)
        # self.conv1_1 = DoubleConv(num_filters[1]+num_filters[2], num_filters[1], 3, stride=stride, padding=1)
        # self.conv2_1 = DoubleConv(num_filters[2]+num_filters[3], num_filters[2], 3, stride=stride, padding=1)
        # self.conv3_1 = DoubleConv(num_filters[3]+num_filters[4], num_filters[3], 3, stride=stride, padding=1)
        self.conv0_1 = DoubleConv(self.num_filters[0]*2, self.num_filters[0], 3, stride=stride, padding=1)
        self.conv1_1 = DoubleConv(self.num_filters[1]*2, self.num_filters[1], 3, stride=stride, padding=1)
        self.conv2_1 = DoubleConv(self.num_filters[2]*2, self.num_filters[2], 3, stride=stride, padding=1)
        self.conv3_1 = DoubleConv(self.num_filters[3]*2, self.num_filters[3], 3, stride=stride, padding=1)

        # self.conv0_2 = DoubleConv(num_filters[0]*2+num_filters[1], num_filters[0], 3, stride=stride, padding=1)
        # self.conv1_2 = DoubleConv(num_filters[1]*2+num_filters[2], num_filters[1], 3, stride=stride, padding=1)
        # self.conv2_2 = DoubleConv(num_filters[2]*2+num_filters[3], num_filters[2], 3, stride=stride, padding=1)
        self.conv0_2 = DoubleConv(self.num_filters[0]*3, self.num_filters[0], 3, stride=stride, padding=1)
        self.conv1_2 = DoubleConv(self.num_filters[1]*3, self.num_filters[1], 3, stride=stride, padding=1)
        self.conv2_2 = DoubleConv(self.num_filters[2]*3, self.num_filters[2], 3, stride=stride, padding=1)

        # self.conv0_3 = DoubleConv(num_filters[0]*3+num_filters[1], num_filters[0], 3, stride=stride, padding=1)
        # self.conv1_3 = DoubleConv(num_filters[1]*3+num_filters[2], num_filters[1], 3, stride=stride, padding=1)
        self.conv0_3 = DoubleConv(self.num_filters[0]*4, self.num_filters[0], 3, stride=stride, padding=1)
        self.conv1_3 = DoubleConv(self.num_filters[1]*4, self.num_filters[1], 3, stride=stride, padding=1)

        self.conv0_4 = DoubleConv(self.num_filters[0]*5, self.num_filters[0], 3, stride=stride, padding=1)

        # up-sampling
        self.deconv1_0 = DoubleDeconv(self.num_filters[1], self.num_filters[0], 3, stride=stride, padding=1)
        self.deconv2_0 = DoubleDeconv(self.num_filters[2], self.num_filters[1], 3, stride=stride, padding=1)
        self.deconv3_0 = DoubleDeconv(self.num_filters[3], self.num_filters[2], 3, stride=stride, padding=1)
        self.deconv4_0 = DoubleDeconv(self.num_filters[4], self.num_filters[3], 3, stride=stride, padding=1)

        self.deconv1_1 = DoubleDeconv(self.num_filters[1], self.num_filters[0], 3, stride=stride, padding=1)
        self.deconv2_1 = DoubleDeconv(self.num_filters[2], self.num_filters[1], 3, stride=stride, padding=1)
        self.deconv3_1 = DoubleDeconv(self.num_filters[3]+hb1, self.num_filters[2], 3, stride=stride, padding=1)

        self.deconv1_2 = DoubleDeconv(self.num_filters[1], self.num_filters[0], 3, stride=stride, padding=1)
        self.deconv2_2 = DoubleDeconv(self.num_filters[2]+hb1, self.num_filters[1], 3, stride=stride, padding=1)

        # self.deconv1_3 = DoubleDeconv(hc1+hb1, hc1, 3, stride=stride, padding=1)
        self.deconv1_3 = DoubleDeconv(self.num_filters[1]+hb1, self.num_filters[0], 3, stride=stride, padding=1)

        # self.deconv0_4 = nn.ConvTranspose2d(hc1+hb1, out_channels, 3, stride=stride, padding=1)
        self.deconv0_4 = nn.ConvTranspose2d(self.num_filters[0]+hb1, out_channels, 3, stride=stride, padding=1)

        self.act = nn.LeakyReLU()

        # self.dropout = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.5)

        # norm in encoder process
        self.norm0_0 = nn.InstanceNorm2d(self.num_filters[0])
        self.norm1_0 = nn.InstanceNorm2d(self.num_filters[1])
        self.norm2_0 = nn.InstanceNorm2d(self.num_filters[2])
        self.norm3_0 = nn.InstanceNorm2d(self.num_filters[3])
        self.norm4_0 = nn.InstanceNorm2d(self.num_filters[4])

        # norm in decoder process
        self.dnorm0_1 = nn.InstanceNorm2d(self.num_filters[0])

        self.dnorm1_1 = nn.InstanceNorm2d(self.num_filters[1])
        self.dnorm0_2 = nn.InstanceNorm2d(self.num_filters[0])

        self.dnorm2_1 = nn.InstanceNorm2d(self.num_filters[2])
        self.dnorm1_2 = nn.InstanceNorm2d(self.num_filters[1])
        self.dnorm0_3 = nn.InstanceNorm2d(self.num_filters[0])

        self.dnorm3_1 = nn.InstanceNorm2d(self.num_filters[3])
        self.dnorm2_2 = nn.InstanceNorm2d(self.num_filters[2])
        self.dnorm1_3 = nn.InstanceNorm2d(self.num_filters[1])
        self.dnorm0_4 = nn.InstanceNorm2d(self.num_filters[0])

        self.fnorm1 = nn.InstanceNorm2d(hb1)
        self.fnorm2 = nn.InstanceNorm2d(hb1)
        self.fnorm3 = nn.InstanceNorm2d(hb1)
        self.fnorm4 = nn.InstanceNorm2d(hb1)

        self.lang19 = nn.Linear(self.emb_block_size, self.num_filters[0] * hb1)
        self.lang28 = nn.Linear(self.emb_block_size, self.num_filters[1] * hb1)
        self.lang37 = nn.Linear(self.emb_block_size, self.num_filters[2] * hb1)
        self.lang46 = nn.Linear(self.emb_block_size, self.num_filters[3] * hb1)
        self.lang55 = nn.Linear(self.emb_block_size, self.num_filters[4] * self.num_filters[4])

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def init_weights(self):
        self.conv0_0.init_weights()
        self.conv1_0.init_weights()
        self.conv2_0.init_weights()
        self.conv3_0.init_weights()
        self.conv4_0.init_weights()

        self.conv0_1.init_weights()
        self.conv1_1.init_weights()
        self.conv2_1.init_weights()
        self.conv3_1.init_weights()

        self.conv0_2.init_weights()
        self.conv1_2.init_weights()
        self.conv2_2.init_weights()

        self.conv0_3.init_weights()
        self.conv1_3.init_weights()

        self.conv0_4.init_weights()

        self.deconv1_0.init_weights()
        self.deconv2_0.init_weights()
        self.deconv3_0.init_weights()
        self.deconv4_0.init_weights()

        self.deconv1_1.init_weights()
        self.deconv2_1.init_weights()
        self.deconv3_1.init_weights()

        self.deconv1_2.init_weights()
        self.deconv2_2.init_weights()

        self.deconv1_3.init_weights()

    def forward(self, input, embedding):

        # calculate the layer before the embedding concatenation part
        x0_0 = self.norm0_0(self.act(self.conv0_0(input)))

        # L1
        x1_0 = self.norm1_0(self.act(self.conv1_0(x0_0)))
        x0_1 = self.dnorm0_1(self.act(self.conv0_1(self.up(torch.cat([x0_0, self.deconv1_0(x1_0, output_size=x0_0.size())], 1)))))

        # L2
        x2_0 = self.norm2_0(self.act(self.conv2_0(x1_0)))
        x1_1 = self.dnorm1_1(self.act(self.conv1_1(self.up(torch.cat([x1_0, self.deconv2_0(x2_0, output_size=x1_0.size())], 1)))))
        x0_2 = self.dnorm0_2(self.act(self.conv0_2(self.up(torch.cat([x0_0, x0_1, self.deconv1_1(x1_1, output_size=x0_0.size())], 1)))))

        # L3
        x3_0 = self.norm3_0(self.act(self.conv3_0(x2_0)))
        x2_1 = self.dnorm2_1(self.act(self.conv2_1(self.up(torch.cat([x2_0, self.deconv3_0(x3_0, output_size=x2_0.size())], 1)))))
        x1_2 = self.dnorm1_2(self.act(self.conv1_2(self.up(torch.cat([x1_0, x1_1, self.deconv2_1(x2_1, output_size=x1_0.size())], 1)))))
        x0_3 = self.dnorm0_3(self.act(self.conv0_3(self.up(torch.cat([x0_0, x0_1, x0_2, self.deconv1_2(x1_2, output_size=x0_0.size())], 1)))))

        x4_0 = self.norm4_0(self.act(self.conv4_0(x3_0)))

        # x4_0, x3_0, x2_1, x1_2, x0_3 need to concatenate the embedding vector
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
            x1f = Variable(torch.zeros_like(x0_3[:, 0:self.hb1, :, :].data))
            x2f = Variable(torch.zeros_like(x1_2[:, 0:self.hb1, :, :].data))
            x3f = Variable(torch.zeros_like(x2_1[:, 0:self.hb1, :, :].data))
            x4f = Variable(torch.zeros_like(x3_0[:, 0:self.hb1, :, :].data))
            x5f = Variable(torch.zeros_like(x4_0.data))

            batch_size = embedding.size(0)

            for i in range(batch_size):
                lf1 = F.normalize(self.lang19(emb1[i:i+1])).view([self.hb1, self.num_filters[0], 1, 1])
                lf2 = F.normalize(self.lang28(emb2[i:i+1])).view([self.hb1, self.num_filters[1], 1, 1])
                lf3 = F.normalize(self.lang37(emb3[i:i+1])).view([self.hb1, self.num_filters[2], 1, 1])
                lf4 = F.normalize(self.lang46(emb4[i:i+1])).view([self.hb1, self.num_filters[3], 1, 1])
                lf5 = F.normalize(self.lang55(emb5[i:i+1])).view([self.num_filters[4], self.num_filters[4], 1, 1])

                x1f[i:i+1] = F.conv2d(x0_3[i:i+1], lf1)
                x2f[i:i+1] = F.conv2d(x1_2[i:i+1], lf2)
                x3f[i:i+1] = F.conv2d(x2_1[i:i+1], lf3)
                x4f[i:i+1] = F.conv2d(x3_0[i:i+1], lf4)
                x5f[i:i+1] = F.conv2d(x4_0[i:i+1], lf5)
            x0_3l = self.fnorm1(x1f)
            x1_2l = self.fnorm2(x2f)
            x2_1l = self.fnorm3(x3f)
            x3_0l = self.fnorm4(x4f)
            x4_0l = x5f

        x4_0 = self.act(self.deconv4_0(x4_0l, output_size=x3_0.size()))
        x3_1 = self.dnorm3_1(self.act(self.deconv3_1(torch.cat([x3_0l, x4_0], 1), output_size=x2_0.size())))
        x2_2 = self.dnorm2_2(self.act(self.deconv2_2(torch.cat([x2_1l, x3_1], 1), output_size=x1_0.size())))
        x1_3 = self.dnorm1_3(self.act(self.deconv1_3(torch.cat([x1_2l, x2_2], 1), output_size=x0_0.size())))
        out = self.deconv0_4(torch.cat([x0_3l, x1_3], 1), output_size=input.size())

        # x3_1 = self.dnorm3_1(self.act(torch.cat([x3_0, self.deconv4_0(x4_0, output_size=x3_0.size())], 1)))
        # x2_2 = self.dnorm2_2(self.act(torch.cat([x2_1, self.deconv3_1(x3_1)], 1)))
        # x1_3 = self.dnorm1_3(self.act(torch.cat([x1_2, self.deconv2_2(x2_2)], 1)))
        # x0_4 = self.dnorm0_4(self.act(torch.cat([x0_3, self.deconv1_3(x1_3)], 1)))
        # out = self.deconv0_4(x0_4)

        return out