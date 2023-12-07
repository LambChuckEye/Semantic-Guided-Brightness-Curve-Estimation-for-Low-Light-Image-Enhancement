import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
from utils import *
from model.u2net import U2NETP


# =============== model ===================
class Decom_Net(nn.Module):
    def __init__(self, in_dim=3):
        super(Decom_Net, self).__init__()
        self.u2net = U2NETP(in_dim,4)


    def forward(self, x):
        ref = self.u2net(x)
        
        # R, I
        return ref[:,:3,:,:],ref[:,3:,:,:],

# ================ other model ==============
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu', stride=1):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.conv_relu(x)
    
class ConvTranspose2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.deconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.deconv_relu(x)
 
class MaxPooling2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)

class Concat(nn.Module):
    def forward(self, x, y):
        _, _, xh, xw = x.size()
        _, _, yh, yw = y.size()
        diffY = xh - yh
        diffX = xw - yw
        y = F.pad(y, (diffX // 2, diffX - diffX//2, 
                      diffY // 2, diffY - diffY//2))
        return torch.cat((x, y), dim=1)
    
class DecomNet(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.conv_input = Conv2D(3, filters)
        # top path build Reflectance map
        self.maxpool_r1 = MaxPooling2D()
        self.conv_r1 = Conv2D(filters, filters*2)
        self.maxpool_r2 = MaxPooling2D()
        self.conv_r2 = Conv2D(filters*2, filters*4)
        self.deconv_r1 = ConvTranspose2D(filters*4, filters*2)
        self.concat_r1 = Concat()
        self.conv_r3 = Conv2D(filters*4, filters*2)
        self.deconv_r2 = ConvTranspose2D(filters*2, filters)
        self.concat_r2 = Concat()
        self.conv_r4 = Conv2D(filters*2, filters)
        self.conv_r5 = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self.R_out = nn.Sigmoid()
        # bottom path build Illumination map
        self.conv_i1 = Conv2D(filters, filters)
        self.concat_i1 = Concat()
        self.conv_i2 = nn.Conv2d(filters*2, 1, kernel_size=3, padding=1)
        self.I_out = nn.Sigmoid()

    def forward(self, x):
        conv_input = self.conv_input(x)
        # build Reflectance map
        maxpool_r1 = self.maxpool_r1(conv_input)
        conv_r1 = self.conv_r1(maxpool_r1)
        maxpool_r2 = self.maxpool_r2(conv_r1)
        conv_r2 = self.conv_r2(maxpool_r2)
        deconv_r1 = self.deconv_r1(conv_r2)
        concat_r1 = self.concat_r1(conv_r1, deconv_r1)
        conv_r3 = self.conv_r3(concat_r1)
        deconv_r2 = self.deconv_r2(conv_r3)
        concat_r2 = self.concat_r2(conv_input, deconv_r2)
        conv_r4 = self.conv_r4(concat_r2)
        conv_r5 = self.conv_r5(conv_r4)
        R_out = self.R_out(conv_r5)
        
        # build Illumination map
        conv_i1 = self.conv_i1(conv_input)
        concat_i1 = self.concat_i1(conv_r4, conv_i1)
        conv_i2 = self.conv_i2(concat_i1)
        I_out = self.I_out(conv_i2)

        return R_out, I_out
# ================ loss ====================


class Decom_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def reflectance_similarity(self, R_low, R_high):
        return torch.mean(torch.abs(R_low - R_high))
    
    def illumination_smoothness(self, I, L, name='low', hook=-1):
        # L_transpose = L.permute(0, 2, 3, 1)
        # L_gray_transpose = 0.299*L[:,:,:,0] + 0.587*L[:,:,:,1] + 0.114*L[:,:,:,2]
        # L_gray = L.permute(0, 3, 1, 2)
        L_gray = 0.299*L[:,0,:,:] + 0.587*L[:,1,:,:] + 0.114*L[:,2,:,:]
        L_gray = L_gray.unsqueeze(dim=1)
        I_gradient_x = gradient(I, "x")
        L_gradient_x = gradient(L_gray, "x")
        epsilon = 0.01*torch.ones_like(L_gradient_x)
        Denominator_x = torch.max(L_gradient_x, epsilon)
        x_loss = torch.abs(torch.div(I_gradient_x, Denominator_x))
        I_gradient_y = gradient(I, "y")
        L_gradient_y = gradient(L_gray, "y")
        Denominator_y = torch.max(L_gradient_y, epsilon)
        y_loss = torch.abs(torch.div(I_gradient_y, Denominator_y))
        mut_loss = torch.mean(x_loss + y_loss)
        if hook > -1:
            feature_map_hook(I, L_gray, epsilon, I_gradient_x+I_gradient_y, Denominator_x+Denominator_y, 
                            x_loss+y_loss, path=f'./images/samples-features/ilux_smooth_{name}_epoch{hook}.png')
        return mut_loss
    
    def mutual_consistency(self, I_low, I_high, hook=-1):
        low_gradient_x = gradient(I_low, "x")
        high_gradient_x = gradient(I_high, "x")
        M_gradient_x = low_gradient_x + high_gradient_x
        x_loss = M_gradient_x * torch.exp(-10 * M_gradient_x)
        low_gradient_y = gradient(I_low, "y")
        high_gradient_y = gradient(I_high, "y")
        M_gradient_y = low_gradient_y + high_gradient_y
        y_loss = M_gradient_y * torch.exp(-10 * M_gradient_y)
        mutual_loss = torch.mean(x_loss + y_loss) 
        if hook > -1:
            feature_map_hook(I_low, I_high, low_gradient_x+low_gradient_y, high_gradient_x+high_gradient_y, 
                    M_gradient_x + M_gradient_y, x_loss+ y_loss, path=f'./images/samples-features/mutual_consist_epoch{hook}.png')
        return mutual_loss

    def reconstruction_error(self, R_low, R_high, I_low_3, I_high_3, L_low, L_high):
        recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 -  L_low))
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - L_high))
        recon_loss_l2h = torch.mean(torch.abs(R_high * I_low_3 -  L_low))
        recon_loss_h2l = torch.mean(torch.abs(R_low * I_high_3 - L_high))
        return recon_loss_high + recon_loss_low + recon_loss_l2h + recon_loss_h2l

    def forward(self, R_low, R_high, I_low, I_high, L_low, L_high, hook=-1):
        I_low_3 = torch.cat([I_low, I_low, I_low], dim=1)
        I_high_3 = torch.cat([I_high, I_high, I_high], dim=1)
        #network output
        recon_loss = self.reconstruction_error(R_low, R_high, I_low_3, I_high_3, L_low, L_high)
        equal_R_loss = self.reflectance_similarity(R_low, R_high)
        i_mutual_loss = self.mutual_consistency(I_low, I_high, hook=hook)
        ilux_smooth_loss = self.illumination_smoothness(I_low, L_low, hook=hook) + \
                    self.illumination_smoothness(I_high, L_high, name='high', hook=hook) 

        decom_loss = recon_loss + 0.2*equal_R_loss + 0.2 * i_mutual_loss + 0.15 * ilux_smooth_loss

        return decom_loss


    
    
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    img = torch.Tensor(1, 3, 400, 600)
    net = Decom_Net()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    r,l = net(img)
    print(r.shape)
    print(l.shape)