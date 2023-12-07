import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import time
import os
import argparse
import numpy as np
from utils import PSNR, validation, LossNetwork
from model.IAT_main import IAT
# from model.IAT_originZeroDCE import IAT
# from model.IAT_local_impl_zero_with_bias import IAT

# from model.IAT_local_U2Net import IAT
from model.mirnet_v2_arch import MIRNet_v2
import zero_model

from IQA_pytorch import SSIM, MS_SSIM
from data_loaders.lol_v1_whole import lowlight_loader_new
from tqdm import tqdm


for i in tqdm((10,)):
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default=1)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--img_val_path', type=str, default= '/home/wsz/workspace/Data/video/dark'+str(i)+'/low/')
    config = parser.parse_args()

    print(config)
    val_dataset = lowlight_loader_new(images_path=config.img_val_path, mode='test')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    # model = MIRNet_v2().cuda()
    # s = "/home/wsz/workspace/Illumination-Adaptive-Transformer/IAT_enhance/enhancement_lol.pth"

    # # 加载模型 
    # checkpoint = torch.load(s)
    # # mir预训练模型读取
    # model.load_state_dict(checkpoint['params'])
    # print("===>Testing using weights: ",s)

    model = IAT().cuda()
    s = "/home/wsz/workspace/Illumination-Adaptive-Transformer/IAT_enhance/best_Epoch_lol.pth"
    model.load_state_dict(torch.load(s))
    model.eval()



    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    if config.save:
        result_path = config.img_val_path.replace('low', 'res-IAT-1')
        mkdir(result_path)
    start_time = time.time()
    flag_count = 1
    with torch.no_grad():
        for i, imgs in enumerate(val_loader):
            # print(i)
            low_img, high_img, name = imgs[0].cuda(), imgs[1].cuda(), str(imgs[2][0])
            #print(low_img.shape)
            _,enhanced_img,_  = model(low_img)
            if config.save:
                torchvision.utils.save_image(enhanced_img, result_path + str(name) + '.jpg')

            
            flag_count += 1
            
    end_time = time.time()
    run_time = end_time - start_time
    print(f"程序运行时间为：{run_time} 秒")
    print(f"平均帧处理时间为：{run_time/flag_count} 秒")

