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
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import vgg16

from data_loaders.lol_v1_whole import lowlight_loader_new

# 原始zeroDCE模型
# from model.IAT_originZeroDCE import IAT
# from model.IAT_main import IAT
# from model.IAT_local_impl_zero_with_bias import IAT
from model.decom_net import Decom_Loss, Decom_Net, DecomNet


from IQA_pytorch import SSIM
from utils import PSNR, adjust_learning_rate, validation, LossNetwork, visualization,my_validation

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--img_path', type=str, default='/home/wsz/workspace/Data/LOLdataset/our485/low/')
parser.add_argument('--img_val_path', type=str, default='/home/wsz/workspace/Data/LOLdataset/eval15/low/')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--pretrain_dir', type=str, default='/home/wsz/workspace/Illumination-Adaptive-Transformer/IAT_enhance/workdirs/snapshots_folder_lol_v1_patch_decom_221019/best_Epoch.pth')
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="workdirs/snapshots_folder_lol_v1_whole_decom_221019")

config = parser.parse_args()

print(config)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

if not os.path.exists(config.snapshots_folder):
    os.makedirs(config.snapshots_folder)

# Model Setting
model = DecomNet().cuda()

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


# L1_loss = CharbonnierLoss()
loss_fn = Decom_Loss()


ssim = SSIM()
psnr = PSNR()
ssim_high = 0
psnr_high = 0

model.train()
print('######## Start IAT Training #########')
for epoch in range(config.num_epochs):
    # adjust_learning_rate(optimizer, epoch)
    print('the epoch is:', epoch)
    for iteration, imgs in enumerate(train_loader):
        L_low, L_high = imgs[0].cuda(), imgs[1].cuda()

        # loss = L1_loss(enhance_img, high_img)
        optimizer.zero_grad()
        model.train()
        R_low, I_low = model(L_low)
        R_high, I_high = model(L_high)

        # loss = L1_loss(enhance_img, high_img)
        loss = loss_fn(R_low, R_high, I_low, I_high, L_low, L_high, hook=-1)
        loss.backward()
        
        optimizer.step()
        scheduler.step()

        if ((iteration + 1) % config.display_iter) == 0:
            print("Loss at iteration", iteration + 1, ":", loss.item())

    # Evaluation Model
    model.eval()
    PSNR_mean, SSIM_mean = my_validation(model, val_loader)

    with open(config.snapshots_folder + '/log.txt', 'a+') as f:
        f.write('epoch' + str(epoch) + ':' + 'the SSIM is' + str(SSIM_mean) + 'the PSNR is' + str(PSNR_mean) + '\n')

    if SSIM_mean > ssim_high:
        ssim_high = SSIM_mean
        print('the highest SSIM value is:', str(ssim_high))
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch" + '.pth'))

    f.close()
