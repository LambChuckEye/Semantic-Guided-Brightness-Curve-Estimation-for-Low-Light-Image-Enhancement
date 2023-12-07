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

from model.IAT_local_MIR2_pretrain import IAT

from IQA_pytorch import SSIM, MS_SSIM
from data_loaders.lol_v1_new import lowlight_loader_new
from tqdm import tqdm
import cv2  # type: ignore


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import numpy as np
import json
import os
from typing import Any, Dict, List

# SAM参数解析
def get_amg_kwargs(args):
    amg_kwargs = {
        "box_nms_thresh": args.box_nms_thresh
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

# 在mask中添加背景
def add_background_mask(masks):
    if len(masks) == 0:
        return masks
    bg = np.ones(masks[0]["segmentation"].shape)
    for mask in masks:
        bg -= mask["segmentation"]
    masks.append({"segmentation": bg.astype(bool)})
    return masks

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=1)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--img_val_path', type=str, default='/home/wsz/workspace/Data/LOLdataset/eval15/low/')
parser.add_argument(
    "--model_type",
    type=str,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    default='vit_h',
)
parser.add_argument(
    "--checkpoint",
    type=str,
    help="The path to the SAM checkpoint to use for mask generation.",
    default="/home/wsz/workspace/segment-anything-main/sam_vit_h_4b8939.pth",
)
parser.add_argument(
    "--convert_to_rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)
parser.add_argument(
    "--box_nms_thresh",
    type=float,
    default=0,
    help="The overlap threshold for excluding a duplicate mask.",
)


config = parser.parse_args()

print(config)
val_dataset = lowlight_loader_new(images_path=config.img_val_path, mode='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

model = IAT().cuda()
s = "/home/wsz/workspace/Illumination-Adaptive-Transformer/IAT_enhance/workdirs/snapshots_folder_lolv1_SAM_20230526/best_Epoch.pth"
model.load_state_dict(torch.load(s))
model.eval()


# SAM初始化
sam = sam_model_registry[config.model_type](checkpoint=config.checkpoint)
sam = sam.cuda()
output_mode = "coco_rle" if config.convert_to_rle else "binary_mask"
amg_kwargs = get_amg_kwargs(config)
generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, box_nms_thresh=config.box_nms_thresh)

ssim = SSIM()
psnr = PSNR()
ssim_list = []
psnr_list = []

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if config.save:
    result_path = config.img_val_path.replace('low', '0529')
    mkdir(result_path)

with torch.no_grad():
    for i, imgs in tqdm(enumerate(val_loader)):
        # print(i)
        low_img, high_img, name = imgs[0].cuda(), imgs[1].cuda(), str(imgs[2][0])
        print(name)
        

        # SAM分割
        
        # 生成掩码（低亮度)(SAM中输入的img shape 为 [h,w,c]，需要去掉batch同时交换c)
        
        masks = generator.generate(high_img[0].permute(1,2,0))
        # 生成掩码（高亮度）
        # masks = generator.generate(high_img)

        # 生成背景
        masks = add_background_mask(masks)
        
        enhanced_img = torch.zeros_like(low_img)
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            mask = torch.tensor(mask).cuda()
            mask = torch.unsqueeze(mask,0)
            mask = torch.unsqueeze(mask,0)
            mul, add ,enhanced_patch_img = model(low_img * mask)
            enhanced_img += enhanced_patch_img
        
        #print(low_img.shape)
        
        if config.save:
            torchvision.utils.save_image(enhanced_img, result_path + str(name) + '.png')

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        print("name: ",name,"SSIM: ",ssim_value,"PSNR: ", psnr_value)

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)


SSIM_mean = np.mean(ssim_list)
PSNR_mean = np.mean(psnr_list)
print('The SSIM Value is:', SSIM_mean)
print('The PSNR Value is:', PSNR_mean)


