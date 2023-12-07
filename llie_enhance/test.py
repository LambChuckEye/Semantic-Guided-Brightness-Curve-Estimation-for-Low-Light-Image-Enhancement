import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os

from model.decom_net import Decom_Net

if __name__ == "__main__":
    print(torch.__version__)
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    img = torch.Tensor(3, 3, 400, 600)
    net = Decom_Net()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    r,l = net(img)
    print(r.shape)
    print(l.shape)