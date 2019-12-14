import torch
import itertools
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import random
import torchvision.transforms.functional as TF
import PIL
import numpy as np
from scipy.misc import imread
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import ImageFile
import sys
sys.path.append("..")
from utils import *
def show_images(images):
    ##adapted from cs231n course code http://cs231n.github.io/
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return 
def preprocess(img, size=512):
    ##adapted from cs231n course code http://cs231n.github.io/
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)
def deprocess(img):
    ##adapted from cs231n course code http://cs231n.github.io/
    transform = T.Compose([
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in [0.229, 0.224, 0.225]]),
        T.Normalize(mean=[-m for m in [0.485, 0.456, 0.406]], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)
def rescale(x):
    ##adapted from cs231n course code http://cs231n.github.io/
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled
def train_transforms():
    data_transforms = {
    'train': T.Compose([
    # T.RandomResizedCrop(256),
    T.RandomHorizontalFlip(),
       T.Resize((256,256)),
    T.RandomAffine(10,shear=10),
    T.ColorJitter(0.2,0.2,0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': T.Compose([
    T.Resize(300),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),}
    return data_transforms
# dtype = torch.cuda.FloatTensor
def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
def my_segmentation_transforms(image, segmentation,data):
    if random.random() > 0:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        segmentation = TF.rotate(segmentation, angle)
        if random.random()>0.5:
            image = TF.hflip(image)
            segmentation = TF.hflip(segmentation)
        image = TF.resize(image, data["size"])
        segmentation = TF.resize(segmentation, data["size"])  
        affine= random.randint(-30, 30)
        image=TF.affine(image,affine,scale=1,translate=[0,0],shear=affine)
        segmentation=TF.affine(segmentation,affine,scale=1,translate=[0,0],shear=affine)
        image=TF.to_tensor(image)
        segmentation=TF.to_tensor(segmentation)
        image=TF.normalize(image,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        segmentation=TF.normalize(segmentation,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    return image, segmentation
def count_params(model):
    ##adapted from cs231n course code http://cs231n.github.io/
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count
if __name__ == "__main__":
    monet="/home/taoyang/research/research_everyday/faceswap-GAN/faceA/rgb/"
    data_transforms=train_transforms()
    dtype=torch.FloatTensor
    content=cycle_data_withfolder(monet,data_transforms["train"],dtype)
    loader_content = DataLoader(content,
                    batch_size=4,
                    num_workers=7,
                    shuffle=True)
    loader_style_iter=iter(loader_content)
    for i in range(10):
        img=next(loader_style_iter)
        img1=deprocess(img[0,:,:,:])
        print(img1)
        


