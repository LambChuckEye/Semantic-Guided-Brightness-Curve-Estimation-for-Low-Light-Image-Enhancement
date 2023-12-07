# 使用预训练的mirnet

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import sys
import argparse
import time
import numpy as np
from tqdm import tqdm
from torchvision.models import vgg16

from data_loaders.lol_v1_whole import lowlight_loader_new

# 原始zeroDCE模型
# from model.IAT_originZeroDCE import IAT
# from model.IAT_main import IAT
# from model.IAT_local_impl_zero_with_bias import IAT
# from model.IAT_local_U2Net import IAT
from model.IAT_local_MIR2_pretrain import IAT,Curve


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=1)
parser.add_argument('--img_path', type=str, default='/home/wsz/workspace/Data/LOLdataset/our485/low/')
parser.add_argument('--img_val_path', type=str, default='/home/wsz/workspace/Data/LOLdataset/eval15/low/')

parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=1e-8)
parser.add_argument('--pretrain_dir', type=str, default=None)
parser.add_argument('--num_epochs', type=int, default=800)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="workdirs/snapshots_folder_lol_v1_curve_tackmem")

config = parser.parse_args()

print(config)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

if not os.path.exists(config.snapshots_folder):
    os.makedirs(config.snapshots_folder)

# Model Setting
model = IAT().cuda()
if config.pretrain_dir is not None:
    model.load_state_dict(torch.load(config.pretrain_dir))

# Data Setting
train_dataset = lowlight_loader_new(images_path=config.img_path)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True)
val_dataset = lowlight_loader_new(images_path=config.img_val_path, mode='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

device = next(model.parameters()).device
print('the device is:', device)

# Loss & Optimizer Setting & Metric
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()

for param in vgg_model.parameters():
    param.requires_grad = False

# L1_loss = CharbonnierLoss()
L1_loss = nn.L1Loss()
L1_smooth_loss = F.smooth_l1_loss

# loss_network = LossNetwork(vgg_model)
# loss_network.eval()

ssim_high = 0
psnr_high = 0

# mirnet相关
# weights = "/home/wsz/workspace/Illumination-Adaptive-Transformer/IAT_enhance/enhancement_lol.pth"
# mirnet = MIRNet_v2().cuda()
# checkpoint = torch.load(weights)
# mirnet.load_state_dict(checkpoint['params'])
# print("===>Testing using weights: ",weights)


# mirnet.eval()
model.train()
print('######## Start IAT Training #########')
for epoch in range(config.num_epochs):
    # adjust_learning_rate(optimizer, epoch)
    print('the epoch is:', epoch)
    for iteration, imgs in enumerate(train_loader):
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        # low_img = mirnet(low_img)
        # Checking!
        #visualization(low_img, 'show/low', iteration)
        #visualization(high_img, 'show/high', iteration)
        optimizer.zero_grad()
        model.train()
        _,_,enhance_img= model(low_img)

        # loss = L1_loss(enhance_img, high_img)
        loss = L1_smooth_loss(enhance_img, high_img)
        loss.backward()
        
        optimizer.step()
        scheduler.step()

        if ((iteration + 1) % config.display_iter) == 0:
            print("Loss at iteration", iteration + 1, ":", loss.item())

  
