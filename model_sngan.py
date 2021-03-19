import torch
import torch.nn as nn
from snlayer.snconv1d import SNConv1d
from snlayer.snlinear import SNLinear
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            SNConv1d(in_channels=1, out_channels=32, kernel_size=32, stride=2, padding=15),  # batch,32,8192
            nn.LeakyReLU(0.02)
        )
        self.conv2 = nn.Sequential(
            SNConv1d(32, 64, 32, 2, 15),  # batch,64,4096
            nn.LeakyReLU(0.02),
        )
        self.conv3 = nn.Sequential(
            SNConv1d(64, 64, 32, 2, 15),  # batch,64,2048
            nn.LeakyReLU(0.02)
        )
        self.conv4 = nn.Sequential(
            SNConv1d(64, 128, 32, 2, 15),  # batch,128,1024
            nn.LeakyReLU(0.02)
        )
        self.conv5 = nn.Sequential(
            SNConv1d(128, 128, 32, 2, 15),  # batch,128,512
            nn.LeakyReLU(0.02)
        )
        self.conv6 = nn.Sequential(
            SNConv1d(128, 256, 32, 2, 15),  # batch,256,256
            nn.LeakyReLU(0.02)
        )
        self.conv7 = nn.Sequential(
            SNConv1d(256, 512, 32, 2, 15),  # batch,512,128
            nn.LeakyReLU(0.02)
        )
        self.conv8 = nn.Sequential(
            SNConv1d(512, 1024, 32, 2, 15),  # batch,1024,64
            nn.LeakyReLU(0.02)
        )
        self.conv9 = nn.Sequential(
            SNConv1d(1024, 1, kernel_size=1, stride=1),  # batch,1,64
            nn.LeakyReLU(0.02)
        )
        self.fc = nn.Sequential(
            SNLinear(in_features=64, out_features=1),
            nn.Sigmoid()
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # down sample
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, stride=2, padding=15)  # batch,16,8192
        self.in1 = nn.BatchNorm1d(16)
        self.conv1_f = nn.PReLU()

        self.conv2 = nn.Conv1d(16, 32, 32, 2, 15)  # batch,32,4096
        self.in2 = nn.BatchNorm1d(32)
        self.conv2_f = nn.PReLU()

        self.conv3 = nn.Conv1d(32, 32, 32, 2, 15)  # batch,32,2048
        self.in3 = nn.BatchNorm1d(32)
        self.conv3_f = nn.PReLU()

        self.conv4 = nn.Conv1d(32, 64, 32, 2, 15)  # batch,64,1024
        self.in4 = nn.BatchNorm1d(64)
        self.conv4_f = nn.PReLU()

        self.conv5 = nn.Conv1d(64, 64, 32, 2, 15)  # batch,64,512
        self.in5 = nn.BatchNorm1d(64)
        self.conv5_f = nn.PReLU()

        self.conv6 = nn.Conv1d(64, 128, 32, 2, 15)  # bacth,128,256
        self.in6 = nn.BatchNorm1d(128)
        self.conv6_f = nn.ReLU()

        self.conv7 = nn.Conv1d(128, 128, 32, 2, 15)  # batch,128,128
        self.in7 = nn.BatchNorm1d(128)
        self.conv7_f = nn.PReLU()

        self.conv8 = nn.Conv1d(128, 256, 32, 2, 15)  # batch,256,64
        self.in8 = nn.BatchNorm1d(256)
        self.conv8_f = nn.PReLU()

        # up sample
        self.deconv9 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  # batch,128,128
        self.in9 = nn.BatchNorm1d(128)
        self.deconv9_f = nn.PReLU()

        self.deconv10 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  # batch,128,256
        self.in10 = nn.BatchNorm1d(128)
        self.deconv10_f = nn.PReLU()

        self.deconv11 = nn.ConvTranspose1d(256, 64, 32, 2, 15)  # batch,64,512
        self.in11 = nn.BatchNorm1d(64)
        self.deconv11_f = nn.PReLU()

        self.deconv12 = nn.ConvTranspose1d(128, 64, 32, 2, 15)  # batch,64,1024
        self.in12 = nn.BatchNorm1d(64)
        self.deconv12_f = nn.PReLU()

        self.deconv13 = nn.ConvTranspose1d(128, 32, 32, 2, 15)  # batch,32,2048
        self.in13 = nn.BatchNorm1d(32)
        self.deconv13_f = nn.PReLU()

        self.deconv14 = nn.ConvTranspose1d(64, 32, 32, 2, 15)  # batch,32,4096
        self.in14 = nn.BatchNorm1d(32)
        self.deconv14_f = nn.PReLU()

        self.deconv15 = nn.ConvTranspose1d(64, 16, 32, 2, 15)  # batch,16,8192
        self.in15 = nn.BatchNorm1d(16)
        self.deconv15_f = nn.PReLU()

        self.deconv16 = nn.ConvTranspose1d(32, 1, 32, 2, 15)  # batch,1,16384
        self.deconv16_tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.in1(c1)
        down1 = self.conv1_f(c1)

        c2 = self.conv2(down1)
        c2 = self.in2(c2)
        down2 = self.conv2_f(c2)

        c3 = self.conv3(down2)
        c3 = self.in3(c3)
        down3 = self.conv3_f(c3)

        c4 = self.conv4(down3)
        c4 = self.in4(c4)
        down4 = self.conv4_f(c4)

        c5 = self.conv5(down4)
        c5 = self.in5(c5)
        down5 = self.conv5_f(c5)

        c6 = self.conv6(down5)
        c6 = self.in6(c6)
        down6 = self.conv6_f(c6)

        c7 = self.conv7(down6)
        c7 = self.in7(c7)
        down7 = self.conv7_f(c7)

        c8 = self.conv8(down7)
        c8 = self.in8(c8)
        down8 = self.conv8_f(c8)

        d9 = self.deconv9(down8)
        d9 = self.in9(d9)
        up9 = self.deconv9_f(torch.cat((d9, down7), dim=1))

        d10 = self.deconv10(up9)
        d10 = self.in10(d10)
        up10 = self.deconv10_f(torch.cat((d10, down6), dim=1))

        d11 = self.deconv11(up10)
        d11 = self.in11(d11)
        up11 = self.deconv11_f(torch.cat((d11, down5), dim=1))

        d12 = self.deconv12(up11)
        d12 = self.in12(d12)
        up12 = self.deconv12_f(torch.cat((d12, down4), dim=1))

        d13 = self.deconv13(up12)
        d13 = self.in13(d13)
        up13 = self.deconv13_f(torch.cat((d13, down3), dim=1))

        d14 = self.deconv14(up13)
        d14 = self.in14(d14)
        up14 = self.deconv14_f(torch.cat((d14, down2), dim=1))

        d15 = self.deconv15(up14)
        d15 = self.in15(d15)
        up15 = self.deconv15_f(torch.cat((d15, down1), dim=1))

        d16 = self.deconv16(up15)
        out = self.deconv16_tanh(d16)
        return out


class Steganalyzer(nn.Module):
    def __init__(self):
        super(Steganalyzer, self).__init__()
        kernel = np.array([[1, -1, 0, 0, 0],
                           [1, -2, 1, 0, 0],
                           [1, -3, 3, -1, 0],
                           [1, -4, 6, -4, 1]], dtype=float)
        kernel = kernel.reshape((4, 1, 5)).astype(np.float32)
        kernel = torch.from_numpy(kernel)
        self.kernel = nn.Parameter(data=kernel, requires_grad=True)

        self.first_conv = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.group1 = nn.Sequential(
            nn.Conv1d(8, 8, 5, 1, 2),
            nn.Conv1d(8, 16, 1, 1, 0)  # batch,16,16384
        )
        self.group2 = nn.Sequential(
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1, 1, 0),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1)  # batch,32,8192
        )
        self.group3 = nn.Sequential(
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1, 1, 0),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1)  # batch,64,4096
        )
        self.group4 = nn.Sequential(
            nn.Conv1d(64, 64, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1, 1, 0),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1)  # batch,128,2048
        )
        self.group5 = nn.Sequential(
            nn.Conv1d(128, 128, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1, 1, 0),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1)  # batch,256,1024
        )
        self.group6 = nn.Sequential(
            nn.Conv1d(256, 256, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1, 1, 0),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # batch,512,1
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.conv1d(x, self.kernel, padding=2)
        x = self.first_conv(x)
        x = torch.clamp(x, -3, 3)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        x = self.group6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

