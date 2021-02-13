import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary as torchsum
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import time
import math



class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class conv_block(nn.Module):
    def __init__(self, in_size, out_size, conv, kernel_size = 4, stride = 2, padding = 1, activation = "relu", BN = True, bias = False, hook = False):
        super(conv_block, self).__init__()
        self.hook = hook
        self.activation = activation
        if conv == "Conv2d":
            if kernel_size == 3 or kernel_size == 4 :
                padding = 1
            elif kernel_size == 7:
                padding = 3
            else: padding = 0
            if self.hook == False:
                self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            else:
                self.hook_conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        if conv == "ConvTranspose2d":
            self.conv = nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias = bias)
        if activation != "False":
            self.relu = nn.LeakyReLU(0.2, inplace=True) if activation=="LeakyRelu" else nn.ReLU(inplace=True)
        self.BN = nn.InstanceNorm2d(out_size) if BN == True else False

    def forward(self, x):
        if self.activation == "False":
            return self.conv(x)
        else:
            if self.BN == False:
                if self.hook == True: return self.relu(self.hook_conv(x))
                else: return self.relu(self.conv(x))
            else:
                if self.hook == True: return self.BN(self.relu(self.hook_conv(x)))
                else: return self.BN(self.relu(self.conv(x)))

class B_Block(nn.Module):
    def __init__(self, in_size, BN=True):
        super(B_Block, self).__init__()
        act = "LeakyRelu"
        self.layer1 = conv_block(in_size, in_size, conv="Conv2d",
                            kernel_size=3, stride=1, padding=1, BN=BN,
                            activation=act)
        self.layer2 = conv_block(in_size, in_size, conv="Conv2d",
                            kernel_size=3, stride=1, padding=1, BN=BN,
                            activation="False")
        self.relu = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x):
        x = self.layer1(x)
        return self.relu(self.layer2(x) + x)

class C_Block(nn.Module):
    def __init__(self, in_size,out_size, BN=True):
        super(C_Block, self).__init__()
        act = "LeakyRelu"
        self.layer1 = conv_block(in_size, out_size, conv="Conv2d",
                            kernel_size=3, stride=2, padding=1, BN=BN,
                            activation=act)
        self.layer2 = conv_block(out_size, out_size, conv="Conv2d",
                            kernel_size=3, stride=1, padding=1, BN=BN,
                            activation="False")
        self.skip_layer = nn.Conv2d(in_size, out_size,
                                    kernel_size=1, stride=2, padding=0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.layer1(x)
        return self.relu(self.layer2(y) + self.skip_layer(x))

class T_Block(nn.Module):
    def __init__(self, in_size, out_size, BN=True):
        super(T_Block, self).__init__()
        act = "LeakyRelu"
        self.layer1 = conv_block(in_size, in_size, conv="Conv2d",
                                 kernel_size=3, stride=1, padding=1, BN=BN,
                                 activation=act)
        self.layer2 = conv_block(in_size, in_size, conv="Conv2d",
                                 kernel_size=3, stride=1, padding=1, BN=BN,
                                 activation=act)
        self.layer3 = conv_block(in_size, out_size, conv = "ConvTranspose2d",
                                 BN=BN,
                                 activation=act)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)



class AE(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(AE, self).__init__()
        # Encoder
        self.input_size = input_size                            # 256 * 160

        en_act = "LeakyRelu"
        de_act = "LeakyRelu"

        self.init_layer1 = conv_block(3, hidden_size[0], conv = "Conv2d",
                                      kernel_size=3, stride=1, padding=1,
                                      activation=en_act)
        self.en_layer1 = B_Block(hidden_size[0])
        self.en_layer2 = B_Block(hidden_size[0])                        # skip1
        self.en_layer3 = C_Block(hidden_size[0],hidden_size[1])         # 128 * 80
        self.en_layer4 = B_Block(hidden_size[1])                        # skip2
        self.en_layer5 = C_Block(hidden_size[1], hidden_size[2])        # 64 * 40
        self.en_layer6 = B_Block(hidden_size[2])                        # skip3
        self.en_layer7 = C_Block(hidden_size[2], hidden_size[3])        # 32 * 20
        self.en_layer8 = B_Block(hidden_size[3])                        # skip4
        self.en_layer9 = C_Block(hidden_size[3], hidden_size[0])        # 16 * 10

        # self.mid_layer1 = conv_block(hidden_size[0], hidden_size[0]*2, conv = "Conv2d",
        #                              kernel_size=3, stride=1, padding=1,
        #                              activation=en_act)
        # self.mid_layer2 = conv_block(hidden_size[0]*2, hidden_size[0], conv = "Conv2d",
        #                              kernel_size=3, stride=1, padding=1,
        #                              activation=en_act)
        self.mid_layer3 = conv_block(hidden_size[0], hidden_size[3], conv = "ConvTranspose2d",
                                     activation=de_act)                 # 32*20

        self.de_layer1 = T_Block(hidden_size[3]*2, hidden_size[2])        # skip4
        self.de_layer2 = T_Block(hidden_size[2]*2, hidden_size[1])        # skip3
        self.de_layer3 = T_Block(hidden_size[1]*2, hidden_size[0])        # skip2
        self.de_layer4 = conv_block(hidden_size[0]*2, hidden_size[0], conv = "Conv2d",
                                    kernel_size=3, stride=1, padding=1, BN = False,
                                    activation=de_act)                  # skip1
        self.de_layer5 = conv_block(hidden_size[0], hidden_size[0], conv = "Conv2d",
                                    kernel_size=3, stride=1, padding=1, BN=False,
                                    activation=de_act)
        self.de_layer6 = nn.Conv2d(hidden_size[0], 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.init_layer1(x)
        x = self.en_layer1(x)
        skip1 = self.en_layer2(x)
        skip2 = self.en_layer3(skip1)
        skip2 = self.en_layer4(skip2)
        skip3 = self.en_layer5(skip2)
        skip3 = self.en_layer6(skip3)
        skip4 = self.en_layer7(skip3)
        skip4 = self.en_layer8(skip4)
        x = self.en_layer9(skip4)

        # middle = self.mid_layer1(middle)
        # middle = self.mid_layer2(middle)
        x = self.mid_layer3(x)
        x = self.de_layer1(torch.cat((x, skip4), dim = 1))
        x = self.de_layer2(torch.cat((x, skip3), dim = 1))
        x = self.de_layer3(torch.cat((x, skip2), dim = 1))
        x = self.de_layer4(torch.cat((x, skip1), dim = 1))
        x = self.de_layer5(x)
        return self.de_layer6(x)


if __name__ == '__main__':
    hidden_size = [64, 128, 256, 512, 800]        # last -> latent dimension // increase complexity, increase latent dimension
    model = AE(hidden_size, input_size=123)
    # print(model)
    #
    torchsum(model, (3, 256, 160))
    # tmp = torch.randn((1,3,256,160))
    # model(tmp)