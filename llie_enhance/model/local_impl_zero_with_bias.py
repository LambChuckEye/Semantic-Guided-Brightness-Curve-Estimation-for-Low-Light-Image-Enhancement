import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math

from timm.models.layers import trunc_normal_
from model.blocks import CBlock_ln, SwinTransformerBlock
from model.global_net import Global_pred

class Local_pred(nn.Module):
    def __init__(self, dim=16, number=4, type='ccc'):
        super(Local_pred, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type =='ccc':  
            #blocks1, blocks2 = [block for _ in range(number)], [block for _ in range(number)]
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type =='ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type =='cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
        self.mul_blocks = nn.Sequential(*blocks1, nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())


    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        mul = self.mul_blocks(img1)
        add = self.add_blocks(img1)

        return mul, add

# Short Cut Connection on Final Layer
class Local_pred_S(nn.Module):
    def __init__(self, in_dim=3, dim=16, number=4, type='ccc'):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        blocks3 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        # blocks3 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        # blocks4 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        # blocks5 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        # blocks6 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        # blocks7 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        # blocks8 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]


        # self.r1_blocks = nn.Sequential(*blocks1)
        # self.r2_blocks = nn.Sequential(*blocks2)
        # self.r3_blocks = nn.Sequential(*blocks3)
        # self.r4_blocks = nn.Sequential(*blocks4)
        # self.r5_blocks = nn.Sequential(*blocks5)
        # self.r6_blocks = nn.Sequential(*blocks6)
        # self.r7_blocks = nn.Sequential(*blocks7)
        # self.r8_blocks = nn.Sequential(*blocks8)

        self.r1_blocks = nn.Sequential(*blocks1)
        self.r2_blocks = nn.Sequential(*blocks2)
        self.add_blocks = nn.Sequential(*blocks3)
        # self.r3_blocks = nn.Sequential(*blocks3)
        # self.r4_blocks = nn.Sequential(*blocks4)


        # self.r1_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        # self.r2_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        # self.r3_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        # self.r4_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        # self.r5_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        # self.r6_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        # self.r7_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        # self.r8_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())

        self.r1_end = nn.Sequential(nn.Conv2d(dim, 12, 3, 1, 1), nn.Tanh())
        self.r2_end = nn.Sequential(nn.Conv2d(dim, 12, 3, 1, 1), nn.Tanh())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        # self.r3_end = nn.Sequential(nn.Conv2d(dim, 6, 3, 1, 1), nn.Tanh())
        # self.r4_end = nn.Sequential(nn.Conv2d(dim, 6, 3, 1, 1), nn.Tanh())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
            
            

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        # short cut connection

        p1 = self.r1_blocks(img1) + img1
        p2 = self.r2_blocks(img1) + img1

        add = self.add_blocks(img1) + img1
        # p3 = self.r3_blocks(img1) 
        # p4 = self.r4_blocks(img1) 

        # r1 = self.r1_blocks(img1) + img1
        # r2 = self.r2_blocks(img1) + img1
        # r3 = self.r3_blocks(img1) + img1
        # r4 = self.r4_blocks(img1) + img1
        # r5 = self.r5_blocks(img1) + img1
        # r6 = self.r6_blocks(img1) + img1
        # r7 = self.r7_blocks(img1) + img1
        # r8 = self.r8_blocks(img1) + img1

        r1, r2, r3, r4 = torch.split(self.r1_end(p1), 3, dim=1)
        r5, r6, r7, r8 = torch.split(self.r2_end(p2), 3, dim=1)
        
        add = self.add_end(add)
        # r5, r6 = torch.split(self.r3_end(p3), 3, dim=1)
        # r7, r8 = torch.split(self.r4_end(p4), 3, dim=1)


        # r1 = self.r1_end(r1)
        # r2 = self.r2_end(r2)
        # r3 = self.r3_end(r3)
        # r4 = self.r4_end(r4)
        # r5 = self.r5_end(r5)
        # r6 = self.r6_end(r6)
        # r7 = self.r7_end(r7)
        # r8 = self.r8_end(r8)
        

        img = img + r1*(torch.pow(img,2)-img)
        img = img + r2*(torch.pow(img,2)-img)
        img = img + r3*(torch.pow(img,2)-img)
        enhance_image_1 = img + r4*(torch.pow(img,2)-img)
        img = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)
        img = img + r6*(torch.pow(img,2)-img)
        img = img + r7*(torch.pow(img,2)-img)
        enhance_image = img + r8*(torch.pow(img,2)-img)
        
        # 添加权重
        enhance_image.add(add)
        
        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
        
        return enhance_image_1,enhance_image,r

class IAT(nn.Module):
    def __init__(self, in_dim=3, with_global=True, type='lol'):
        super(IAT, self).__init__()
        #self.local_net = Local_pred()
        
        self.local_net = Local_pred_S(in_dim=in_dim)

        self.with_global = with_global
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim, type=type)

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low):
        #print(self.with_global)
        mul, add = 1,1
        _,img_high,_ = self.local_net(img_low)

        if not self.with_global:
            return mul, add, img_high
        
        else:
            gamma, color = self.global_net(img_low)
            b = img_high.shape[0]
            img_high = img_high.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
            img_high = torch.stack([self.apply_color(img_high[i,:,:,:], color[i,:,:])**gamma[i,:] for i in range(b)], dim=0)
            img_high = img_high.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
            return mul, add, img_high


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    img = torch.Tensor(1, 3, 400, 600)
    net = IAT()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    _, _, high = net(img)