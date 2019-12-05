import torch
import itertools
import torch.nn as nn
from torch.nn import init
import torchvision
import functools
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import PIL
from PIL import ImageFile
import numpy as np
from scipy.misc import imread
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
from datetime import datetime
import sys

def bce_loss(input1, target,dtype=torch.cuda.FloatTensor):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input1.abs()
    loss = input1.clamp(min=0) - input1 * target + (1 + neg_abs.exp()).log()
    return loss.mean()
def discriminator_loss(logits_real, logits_fake,dtype=torch.cuda.FloatTensor):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    
    real_label=torch.ones_like(logits_real).type(dtype)
    fake_label=torch.zeros_like(logits_fake).type(dtype)
    real_loss=bce_loss(logits_real,real_label)
    fake_loss=bce_loss(logits_fake,fake_label)
    loss=real_loss+fake_loss
    return loss

def generator_loss(logits_fake,dtype=torch.cuda.FloatTensor):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    real_label=torch.ones_like(logits_fake).type(dtype)
    loss=bce_loss(logits_fake,real_label)

    return loss

def ls_discriminator_loss(scores_real, scores_fake,dtype=torch.cuda.FloatTensor):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    label_real=torch.ones_like(scores_real).type(dtype)
    label_fake=torch.zeros_like(scores_fake).type(dtype)
#     print(scores_real,label_real,scores_fake,"scores_real,label_real,scores_fake")
    loss=0.5*((scores_real-label_real)**2).mean()+0.5*((scores_fake-label_fake)**2).mean()
    
    
    return loss

def ls_generator_loss(scores_fake,dtype=torch.cuda.FloatTensor):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    label_fake=torch.ones_like(scores_fake).type(dtype)
    
    loss=0.5*((scores_fake-label_fake)**2).mean()
    
    return loss
def edge_loss(img):
    loss=tv_weight*(((img[:,:,0:-1,:]-img[:,:,1:,:])**2).sum()+((img[:,:,:,0:-1]-img[:,:,:,1:])**2).sum())
    return loss
def squzze_feature_loss(cnn,real,generated,data={"weights":[.1,1,10]}):
    set_requires_grad(cnn,False)
    real_f=extract_features(real,cnn)
    fake_f=extract_features(generated,cnn)
    weights=data["weights"]
    loss=0
    for i in range(2,-1,-1):
        loss+=weights[i]*((real_f[i]-fake_f[i])**2).mean()
    return  loss
if __name__ == "__main__":
    x=torch.rand((4,5))
    print(x,"x")
    y=torch.rand((4,5))
    print(y,"y")
    re=bce_loss(x,y)
    print(re,"bce_loss(x,y)")
    re_d=discriminator_loss(x,y)
    print(re_d,"discriminator_loss(x,y)")
    re_d=generator_loss(x)
    print(re_d,"generator_loss(x)")
    re_d=ls_discriminator_loss(x,y)
    print(re_d,"ls_discriminator_loss(x,y)")
    re_d=ls_generator_loss(x)
    print(re_d,"ls_generator_loss(x)")