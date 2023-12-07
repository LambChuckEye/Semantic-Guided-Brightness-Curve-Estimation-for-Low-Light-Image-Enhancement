# MIRNET模型加全局色彩调节
# 使用预训练的mirnetv2

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math

from timm.models.layers import trunc_normal_
from model.blocks import CBlock_ln, SwinTransformerBlock
from model.global_net import Global_pred
#
from model.my_global_net import my_Global_pred
# 替换成pem的u2net
# from model.u2net_pem import U2NETP


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
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type =='ccc':
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type =='ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type =='cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)

        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
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
        mul = self.mul_blocks(img1) + img1
        add = self.add_blocks(img1) + img1
        mul = self.mul_end(mul)
        add = self.add_end(add)

        return mul, add

class IAT(nn.Module):
    def __init__(self, in_dim=3, with_global=True, type='lol'):
        super(IAT, self).__init__()

        self.with_global = with_global
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim, type=type)
            
        self.curve = Curve(in_dim=in_dim)
            

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    
    def forward(self, img_low):
        #print(self.with_global)
        mul, add = 1,1
        # img_high = (img_low.mul(mul)).add(add)

        # img_high = self.mir2net(img_low)
        
        img_high = self.curve(img_low)
        img_high = img_high.to(torch.float)
        
        if not self.with_global:
            return mul, add, img_high
        
        else:
            gamma, color = self.global_net(img_low)
            b = img_high.shape[0]
            img_high = img_high.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
            img_high = torch.stack([self.apply_color(img_high[i,:,:,:], color[i,:,:])**gamma[i,:] for i in range(b)], dim=0)
            img_high = img_high.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
            return mul, add, img_high
        
# 色彩曲线模型
class Curve(nn.Module):
    def __init__(self, in_dim=3):
        super(Curve, self).__init__()

            
        self.Rreflect = my_Global_pred(in_channels=in_dim,q_nums=256)
        self.Greflect = my_Global_pred(in_channels=in_dim,q_nums=256)
        self.Breflect = my_Global_pred(in_channels=in_dim,q_nums=256)
        
        self.normr = nn.LayerNorm(256)
        self.normg = nn.LayerNorm(256)
        self.normb = nn.LayerNorm(256)

    # 由一阶导数复原原函数
    def apply_A(self, array):
        array = torch.sigmoid(array)
        A = torch.tensor(np.triu(np.ones((1, 256, 256)))).cuda()
        array = array[:, np.newaxis]
        # print(array)
        # 复原一阶导数
        array = torch.matmul(array.to(torch.float64), A)
        array = array.reshape((-1, 256))
        # print("array",array)
        # return torch.sigmoid(array)
        
        # min max归一化
        array = (array - array.min()) /(array.max()-array.min())
        return array
        
        
    # 对原图像应用色彩曲线
    def apply_curve(self,img,r,g,b):
        # 输入的数据均需要像素整数形式
        curve = torch.stack([r,g,b],1).squeeze(2)
        img = img.mul(255).clamp(0,255)
        img = img.permute(2,3,0,1)
        
        # one-hot 编码
        img = F.one_hot(img.to(torch.int64),num_classes = 256)
        
        img = torch.unsqueeze(img,4)# (h,w,b,c,1,d)
        curve = torch.unsqueeze(curve,-1)# (b,c,d,1)

        img = img.to(torch.float64)
        res = torch.matmul(img,curve)# (h,w,b,c,1,1)

        res = torch.squeeze(res,-1)
        res = torch.squeeze(res,-1)# ((h,w,b,c)

        res = res.permute(2,3,0,1)# (b,c,h,w)
        return res
    
    
    def forward(self, img_low):
        #print(self.with_global)
        # img_high = (img_low.mul(mul)).add(add)

        # img_high = self.mir2net(img_low)
        
        # 计算曲线导数
        rr = self.Rreflect(img_low)
        gr = self.Greflect(img_low)
        br = self.Breflect(img_low)
        # batch norm
        # rr = self.normr(rr)
        # gr = self.normr(gr)
        # br = self.normr(br)
        # print("rr: ",rr)
        # print("gr: ",gr)
        # print("br: ",br)
        
        # 应用矩阵A
        rr = self.apply_A(rr)
        gr = self.apply_A(gr)
        br = self.apply_A(br)
        
        # print("rr1: ",rr)
        # print("gr1: ",gr)
        # print("br1: ",br)
        
        #应用曲线
        img_high = self.apply_curve(img_low,rr,gr,br)
        
        return img_high
        
        


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    img = torch.Tensor(1, 3, 400, 600)
    net = IAT()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    _, _, high = net(img)
    print(high.shape)