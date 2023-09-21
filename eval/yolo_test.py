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
from PIL import ImageEnhance
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
from dehaze_lightweight import EncoderDecoder
import cv2
import colorsys
from dataloader import renormalize


def lighten_image(image_path, output_path, brightness_factor):
    # Open the image
    image = Image.open(image_path)
    
    # Create an enhancer object with the brightness factor
    enhancer = ImageEnhance.Brightness(image)
    
    # Apply the brightness enhancement
    lightened_image = enhancer.enhance(brightness_factor)
    
    # Save the lightened image
    lightened_image.save(output_path)

# Example usage

brightness_factor = 2  # Increase this value to lighten the image more


def main() :
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transformss = transforms.ToPILImage()
    model = EncoderDecoder(batch_size=1)

    BatchSize = 1
   
    PATH_SAVE = r'C:\Users\vclab\Desktop\new\venv\lightweight_all_data'
    model.load_state_dict(torch.load(PATH_SAVE))
    model.cuda()

    model.eval()
    inputs = Image.open(r'C:\Users\vclab\Desktop\rescuer_images\frame_1006.png')
    x_transforms = transforms.Compose([
       transforms.Resize((448,448)),
       #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.2),
       transforms.ToTensor(),
       #transforms.Normalize(torch.Tensor(self.meanx), torch.Tensor(self.stdx))
       transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
    res = transforms.Resize((512,960))
    inputs = x_transforms(inputs)
    inputs = renormalize(inputs)
    inputs = torch.unsqueeze(inputs,0)
    inputs = inputs.to(device)
    out = model.forward(inputs)
    out = out.to(device)
    inputs = torch.squeeze(inputs, 0)
    inputs = transformss(inputs)
    inputs = res(inputs)
    out = torch.squeeze(out, 0)
    out = transformss(out)
    #out = res(out)
    #image_path = (r'C:\Users\vclab\Desktop\rescuer_images\light.jpg')
    #output_path = (r'C:\Users\vclab\Desktop\rescuer_images\light2.jpg')
    #lighten_image(image_path, output_path, brightness_factor)
    #inputs.show()
    #out.show()
    
    
  

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    oo = model(inputs)
    # Inference
    #results = model(out)
    #open = Image.open(output_path)
    results = model(out)
 

    # Results
    results.print()
    oo.show()
    results.show()  # or .show()
     
      
   


if __name__ == '__main__':
    main()  
