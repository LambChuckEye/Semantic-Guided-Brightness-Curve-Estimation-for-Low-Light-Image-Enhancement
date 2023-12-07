import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import argparse
import numpy as np
from utils import PSNR, validation, LossNetwork
# from model.IAT_main import IAT
# from model.IAT_originZeroDCE import IAT
# from model.IAT_local_impl_zero_with_bias import IAT

# from model.IAT_local_U2Net import IAT
from model.mirnet_v2_arch import MIRNet_v2

from IQA_pytorch import SSIM, MS_SSIM
from data_loaders.lol_v1_whole import lowlight_loader_new
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=1)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--img_val_path', type=str, default='/home/wsz/workspace/Data/LOLdataset/eval15/low')
config = parser.parse_args()

print(config)
val_dataset = lowlight_loader_new(images_path=config.img_val_path, mode='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

model = MIRNet_v2().cuda()
s = "/home/wsz/workspace/Illumination-Adaptive-Transformer/IAT_enhance/enhancement_lol.pth"

# 加载模型 
checkpoint = torch.load(s)
# mir预训练模型读取
model.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",s)

model.eval()


ssim = SSIM()
psnr = PSNR()
ssim_list = []
psnr_list = []

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if config.save:
    result_path = config.img_val_path.replace('low', '0406mir')
    mkdir(result_path)

with torch.no_grad():
    for i, imgs in tqdm(enumerate(val_loader)):
        # print(i)
        low_img, high_img, name = imgs[0].cuda(), imgs[1].cuda(), str(imgs[2][0])
        print(name)
        #print(low_img.shape)
        enhanced_img = model(low_img)
        if config.save:
            torchvision.utils.save_image(enhanced_img, result_path + str(name) + '.jpg')

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        print("name: ",name,"SSIM: ",ssim_value,"PSNR: ", psnr_value)

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)


SSIM_mean = np.mean(ssim_list)
PSNR_mean = np.mean(psnr_list)
print('The SSIM Value is:', SSIM_mean)
print('The PSNR Value is:', PSNR_mean)
