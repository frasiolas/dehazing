import torch
import torch.nn as nn
from einops import rearrange
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import glob
import os
from turtle import forward
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from torch.utils.data import Dataset, DataLoader
from dataloader import  NHRESIDEITS, NHRESIDEITS_val, renormalize, RESIDE_SOTS
import numpy as np
import time
from timm import models as tmod
from torchsummary import summary
from kornia.contrib.vit_mobile import MobileViT
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from dehaze_lightweight import EncoderDecoder
import colorsys
import math
from math import exp
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(pred, gt):
    pred = pred.clamp(0, 1).cpu().detach().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)

    
def main() :
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transformss = transforms.ToPILImage()
    model = EncoderDecoder(batch_size=1)

    BatchSize = 1
   
    PATH_SAVE = r'C:\Users\vclab\Desktop\new\venv\lightweight_all_data'
    
    model.load_state_dict(torch.load(PATH_SAVE))
    test_dataset =RESIDE_SOTS()
    test_load = DataLoader(dataset=test_dataset, batch_size=BatchSize, shuffle=True)
    model.cuda()
    model.eval()
    ssims = []
    psnrs = []
    for i, data in enumerate(test_load, 0):
        outputs,inputs = data
        inputs = inputs.to(device)
        outputs = outputs.to(device)
        out = model.forward(inputs)
        out = out.to(device)
        outputs = F.interpolate(outputs, size=(480,620))
        out = F.interpolate(out, size=(480,620))
        ssim1 = ssim(out, outputs).item()
        psnr1 = psnr(out, outputs)
        ssims.append(ssim1)
        psnrs.append(psnr1)
    print(np.mean(ssims))
    print(np.mean(psnrs))

if __name__ == '__main__':
    main()  
