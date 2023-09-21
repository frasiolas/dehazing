from email.mime import image
import glob
import os
from statistics import mean
from sys import stderr
from turtle import forward
import torch
from PIL import Image, ImageOps, ImageCms
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NHRESIDEITS(Dataset):

  def __init__(self):
    
    
    self.nyu_depth = pd.read_csv(r'C:/Users/vclab/Desktop/new/venv/csv_files/everything.csv', sep = ',')
    self.meanx = [0.485, 0.456, 0.406]
    self.stdx = [0.229, 0.224, 0.225]

    self.n_samples = self.nyu_depth.shape[0]


  def __getitem__(self, index):
    path_x =  self.nyu_depth._get_value(index, 'clear', takeable=False)
    clear = Image.open(path_x)
    path_y = self.nyu_depth._get_value(index, 'hazy', takeable=False)
    haze = Image.open(path_y)
    if clear.mode == 'RGBA':
      clear = clear.convert('RGB')
    
    if haze.mode == 'RGBA':
      haze = haze.convert('RGB')


    x_transforms = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
       
    ])
     
    veritcal = transforms.RandomHorizontalFlip(p=1)
   
    if random.random() <0.5 :
     haze = veritcal(haze)
     clear = veritcal(clear)
    

    haze = x_transforms(haze)
    clear = x_transforms(clear)
    
    h = 460
    w = 640
    x1 = np.random.randint(0, h - 448)
    y1 = np.random.randint(0, w - 448)
    haze = torchvision.transforms.functional.crop(haze,x1, y1, 448, 448)
    clear = torchvision.transforms.functional.crop(clear,x1, y1,  448, 448)
    haze = renormalize(haze)
    clear = renormalize(clear)
  
    return  clear, haze

  def __len__(self):
    return self.n_samples

class RESIDE_SOTS(Dataset):

  def __init__(self):
    self.nyu_depth = pd.read_csv(r'C:/Users/vclab/Desktop/new/venv/csv_files/RESIDE_SOTS.csv', sep = ',')
    self.meanx = [0.485, 0.456, 0.406]
    self.stdx = [0.229, 0.224, 0.225]

    self.n_samples = self.nyu_depth.shape[0]


  def __getitem__(self, index):
    path_x =  self.nyu_depth._get_value(index, 'clear', takeable=False)
    clear = Image.open(path_x)
    path_y = self.nyu_depth._get_value(index, 'hazy', takeable=False)
    haze = Image.open(path_y)
    x_transforms = transforms.Compose([
       transforms.Resize((448,448)),
       #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.2),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    veritcal = transforms.RandomHorizontalFlip(p=1)
   
    #if random.random() <0.5 :
     #haze = veritcal(haze)
     #clear = veritcal(clear)

    y_transforms = transforms.Compose([
       transforms.Resize((448,448)),
       #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.2),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
  

    transformation = transforms.ToTensor()
    haze = x_transforms(haze)
    clear = y_transforms(clear)
    
    #clear = (clear - clear.min()) / (clear.max() - clear.min())
    #haze = (haze - haze.min()) / (haze.max() - haze.min())

    haze = renormalize(haze)
    clear = renormalize(clear)
  
    return  clear, haze

  def __len__(self):
    return self.n_samples


def renormalize(tensor):
    minFrom= tensor.min()
    maxFrom= tensor.max()
    minTo = 0
    maxTo=1
    return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))

