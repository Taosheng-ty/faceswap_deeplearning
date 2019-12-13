from torch.utils.data import Dataset, DataLoader
import torch
import PIL
from PIL import ImageFile
import os
import sys
sys.path.append("..")
from utils import *


class cycle_data(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir,transform=None,dtype=torch.cuda.FloatTensor):
#         print("init being called")

        self.img= PIL.Image.open(img_dir)
        self.transform = transform
        self.dtype=dtype
    def __len__(self):
#         print("len called")
        return 100000

    def __getitem__(self, idx):
#         print(idx,"idx")
#         print("getitem idx",idx)
#         print(self.img.size)
#         sample=sample.type(self.dtype)
        sample=self.transform(self.img)
#         
        return sample
def remove(x):
    xx_clean=[]
    for i in x:
        if "ipy" not in i:
            xx_clean.append(i)
    return xx_clean
class cycle_data_withfolder(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir,transform=None,data=None,dtype=torch.cuda.FloatTensor):
#         print("init being called")
        self.img_dir=img_dir
        self.dirlist= os.listdir(img_dir)
        self.dirlist=remove(self.dirlist)
        self.len=len(self.dirlist)
        self.transform = transform
        self.dtype=dtype
    def __len__(self):
#         print("len called")
        return 100000

    def __getitem__(self, idx):
#         print(idx,"idx")
#         print("getitem idx",idx)
#         print(self.img.size)
        idx=idx%self.len
        img_dir=self.img_dir+self.dirlist[idx]
        self.img= PIL.Image.open(img_dir)
        if self.transform!=None:
            img=self.transform(self.img)
        sample={'rgb': img}
#         sample=sample.type(self.dtype)
        return sample

class cycle_data_withfolder_mask(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir,mask_dir,transform=None,data=None,dtype=torch.cuda.FloatTensor):
#         print("init being called")
        self.img_dir=img_dir
        self.data=data
        self.mask_dir=mask_dir
        self.dirlist= os.listdir(img_dir)
        self.dirlist_mask= os.listdir(mask_dir)
        self.len=len(self.dirlist)
        self.transform = transform
        self.dtype=dtype
    def __len__(self):
#         print("len called")
        return 100000

    def __getitem__(self, idx):
#         print(idx,"idx")
#         print("getitem idx",idx)
#         print(self.img.size)
        idx=idx%self.len
        img_dir=self.img_dir+self.dirlist[idx]
        mask_dir=self.mask_dir+self.dirlist_mask[idx]
        self.img= PIL.Image.open(img_dir)
        self.mask= PIL.Image.open(mask_dir)
        img,mask=self.transform(self.img,self.mask,self.data)
        mask=mask-torch.min(mask)
        sample={'rgb': img ,'mask':mask}
        
#         sample=sample.type(self.dtype)
        return sample

if __name__ == "__main__":
    monet="/home/taoyang/research/research_everyday/faceswap-GAN/faceA/rgb/"
    data_transforms=transform()
    dtype=torch.FloatTensor
    content=cycle_data_withfolder(monet,data_transforms["train"],dtype)
    
    loader_content = DataLoader(content,
                    batch_size=4,
                    num_workers=7,
                    shuffle=True)
    loader_style_iter=iter(loader_content)
    for i in range(10):
        img=next(loader_style_iter)
        print(img.shape)
    
    
    
    
    
    