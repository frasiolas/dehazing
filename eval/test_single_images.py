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
from dehaze_big import EncoderDecoder
import cv2
import colorsys


def main() :
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transformss = transforms.ToPILImage()
    model = EncoderDecoder(batch_size=1)

    BatchSize = 2
   
    PATH_SAVE = r'C:\Users\vclab\Desktop\new\venv\big_all_data'
    model.load_state_dict(torch.load(PATH_SAVE))
    test_dataset =NHRESIDEITS_val()
    train_load = DataLoader(dataset=test_dataset, batch_size=BatchSize, shuffle=True)
    model.cuda()
    model.eval()


    for i, data in enumerate(train_load, 0):
        outputs,inputs = data
        inputs = inputs.to(device)
        outputs = outputs.to(device)
        out = model.forward(inputs)
        out = out.to(device)
        inputs = F.interpolate(inputs, size=(480,620))
        outputs = F.interpolate(outputs, size=(480,620))
        out = F.interpolate(out, size=(480,620))
        inputs = torch.squeeze(inputs[0], 0)
        outputs = torch.squeeze(outputs[0], 0)
        out = torch.squeeze(out[0], 0)

        inputs = transformss(inputs)                    
        outputs = transformss(outputs)
        out = transformss(out)

        inputs.show()
        outputs.show()
        out.show()
       
     

   


if __name__ == '__main__':
    main()  
