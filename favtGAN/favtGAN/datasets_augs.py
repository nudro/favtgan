import glob
import random
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class TempVector_PyTorch(object):
    def __init__(self, image, d):
        # image is a path for all images - fullpath
        self.image = image
        self.d = d
        
    def replace_with_dict2(self, ar, dic):
        # Extract out keys and values
        k = np.array(list(dic.keys()))
        v = np.array(list(dic.values()))
        # Get argsort indices
        sidx = k.argsort()
        ks = k[sidx]
        vs = v[sidx]
        return vs[np.searchsorted(ks,ar)]
        
    def make_pixel_vectors(self):
        img = np.array(self.image)
        img = img[:, :, 0] # Red channel - for thermal, they're all the same (dtype=uint8)
        temps = self.replace_with_dict2(img, self.d) 
        return temps


    
class MainDataset(Dataset):
    def __init__(self,root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_) # tensor, norm
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        self.T = np.linspace(24, 38, num=256) # 0 - 255 indices of temperatures in Celsius
        self.d = dict(enumerate((self.T).flatten(), 0)) # dictionary like {0: 24.0, 1: 24.054901960784314, etc.}

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h)) # PIL
        img_B = img.crop((w / 2, 0, w, h)) # PIL
        
        # no need to resize in the transforms, it's done here
        # need to do it here so that T_B will be 256 x 256
        newsize = (256, 256)
        img_A = img_A.resize(newsize, Image.BICUBIC)
        img_B = img_B.resize(newsize, Image.BICUBIC)
        
        # temps
        vectorizer = TempVector_PyTorch(img_B, self.d)
        img_B_temps = torch.Tensor(vectorizer.make_pixel_vectors())

        real_A = self.transform(img_A) # tensor
        real_B = self.transform(img_B) # tensor

        return {"A": real_A, 
                "B": real_B,
               "T_B": img_B_temps}

    def __len__(self):
        return len(self.files)
    
    

class AugDataset(Dataset):
    """
    You cannot pass a transforms here, otherwise it will do random transforms on 
    A or B, but not jointly. The only way to make pairwise, same transforms 
    is to use the aug_transform() fxn below.
    """
    def __init__(self,root, transforms_=None, mode="train"):
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        self.T = np.linspace(24, 38, num=256) # 0 - 255 indices of temperatures in Celsius
        self.d = dict(enumerate((self.T).flatten(), 0)) # dictionary like {0: 24.0, 1: 24.054901960784314, etc.}

    
    def aug_transform(self, A, B):
        # A and B are PIL images
        # Resize
        resize = transforms.Resize(size=(256, 256))
        A = resize(A)
        B = resize(B)

        # Random horizontal flipping
        if random.random() > 0.5:
            A = TF.hflip(A)
            B = TF.hflip(B)

        # Random vertical flipping
        if random.random() > 0.5:
            A = TF.vflip(A)
            B = TF.vflip(B)
            
        # here make B = B_flipped to stay as a PIL image
        # use the B_flipped geometry to get the temp vallues
        B_flipped = B

        # Transform to tensor
        A = TF.to_tensor(A)
        B = TF.to_tensor(B)
        
        # Normalize
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        A = normalize(A)
        B = normalize(B)
        
        return A, B, B_flipped

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h)) # PIL
        img_B = img.crop((w / 2, 0, w, h)) # PIL
        
        # augmented, but remains paired
        img_A_aug, img_B_aug, B_flipped = self.aug_transform(img_A, img_B)
        
        # temps have to be done on the PIL image not the tensor
        vectorizer = TempVector_PyTorch(B_flipped, self.d)
        img_B_aug_temps = torch.Tensor(vectorizer.make_pixel_vectors())
      

        return { "A": img_A_aug,
               "B": img_B_aug,
               "T_B": img_B_aug_temps}

    def __len__(self):
        return len(self.files)

    
    
class TestDataset(Dataset):
    """
    At test time, no augs.
    """
    def __init__(self,root, transforms_=None, mode="test"):
        self.transform = transforms.Compose(transforms_)  # resize, tensor, norm
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h)) # PIL
        img_B = img.crop((w / 2, 0, w, h)) # PIL

        real_A = self.transform(img_A) # tensor
        real_B = self.transform(img_B) # tensor

        return {"A": real_A, 
                "B": real_B}

    def __len__(self):
        return len(self.files)