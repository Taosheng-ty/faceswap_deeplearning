import torch
import itertools
import torch.nn as nn
from torch.nn import init
import torchvision
import functools
import torchvision.transforms as T
import torch.optim as optim
from torchsummary import summary
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
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True
from datetime import datetime
import sys
sys.path.append("../")
import sys

sys.path.append("/home/taoyang/research/Tao_lib/")
from Tao_lib.log import Logger
from utils import *
from utils import my_segmentation_transforms
from utils.loss import ls_generator_loss,ls_discriminator_loss
from utils.loss import *
dtype_float=torch.FloatTensor
# device0 = torch.device("cuda:0")
# device1 = torch.device("cuda:1")
# dtype = torch.cuda.HalfTensor

def cyclegan_ls_discriminator_loss(**data):
            real_A,real_B,D_A,D_B,G_A,G_B=data["real_A"],data[\
                 "real_B"],data["D_A"],data["D_B"],data["G_A"],data["G_B"]

            
            reconstructed_A=G_A(real_A)
            reconstructed_B=G_B(real_B)
            
            
            reconstructed_A_score=D_A(reconstructed_A.detach())
            reconstructed_B_score=D_B(reconstructed_B.detach()) 
            real_A_score=D_A(real_A)
            real_B_score=D_B(real_B)
            
            
            loss_DA=ls_discriminator_loss(real_A_score,reconstructed_A_score)
            loss_DB=ls_discriminator_loss(real_B_score,reconstructed_B_score)
            loss_D=(loss_DA+loss_DB)*data["discriminator_loss_weight"]
            return loss_D
        
        
def cyclegan_generator_loss(**data):
            real_A,real_B,D_A,D_B,G_A,G_B=data["real_A"],data[\
                 "real_B"],data["D_A"],data["D_B"],data["G_A"],data["G_B"]
            real_A_mask=data["real_A_mask"]
            real_B_mask=data["real_B_mask"]
            
            reconstructed_A=G_A(real_A)
            reconstructed_B=G_B(real_B)
            
            
            reconstructed_A_score=D_A(reconstructed_A)
            reconstructed_B_score=D_B(reconstructed_B)  
            cycle_A=G_A(G_B(real_A))
            cycle_B=G_B(G_A(real_A))
            loss_G=(ls_generator_loss(reconstructed_A_score)\
            +ls_generator_loss(reconstructed_B_score))*data["fooling_loss_weights"]+(L1(reconstructed_B,real_B)+L1(reconstructed_A,real_A))*data["reconstruction_loss_weights"]+(L1(cycle_A,real_A)+L1(cycle_B,real_B))*data["cycle_loss_weights"]
            
            
            
            if "edge" in data:
                
                    loss_G+=data["edge"]*(edge_loss(reconstructed_A)+edge_loss(reconstructed_B)+\
                             edge_loss(cycle_A)+edge_loss(cycle_B))
                    
                    
            if "cnn" in data:
                cnn=data["cnn"]
                loss_G_pl=squzze_feature_loss(cnn,real_A,reconstructed_A)
                loss_G_pl+=squzze_feature_loss(cnn,real_B,reconstructed_B)
                loss_G_pl+=squzze_feature_loss(cnn,real_B,cycle_B)
                loss_G_pl+=squzze_feature_loss(cnn,real_A,cycle_A)
                loss_G=loss_G_pl*data["pl"]+loss_G
            if "mask_A" in data:
                real_B=real_B*real_B_mask
                real_A=real_A*real_A_mask
                reconstructed_A=reconstructed_A*real_A_mask
                reconstructed_B=reconstructed_B*real_B_mask
                reconstructed_A_score=D_A(reconstructed_A)
                reconstructed_B_score=D_B(reconstructed_B)  
                cycle_A=cycle_A*real_A_mask
                cycle_B=cycle_B*real_B_mask
                loss_G+=(L1(reconstructed_B,real_B)+L1(reconstructed_A,real_A))*data["reconstruction_loss_weights"]+\
                (L1(cycle_A,real_A)+L1(cycle_B,real_B))*data["cycle_loss_weights"]

               
            return loss_G



def run_a_cyclegan(G_A,D_A, G_B,D_B,G_solver, D_solver, discriminator_loss, generator_loss,data={}, show_every=250, 
              batch_size=128, noise_size=96, num_epochs=100):
    """
    Train a GAN!
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    mean_running=0
    iter_count = 0
    print(data)
    dtype=data["dtype"]
    nn=0
    save_iter=1000
#     loader_style_iter=iter(loader_content)       
    loader_content=data["A"]
    loader_style=data["B"]
    other_Loader=data["C"]
#     log_dir=Logger(data["ckpt_path"]+"/")
#     sys.stdout=log_dir
    log_dir=data["ckpt_path"]+"/img/"
    ckpt_log_dir=data["ckpt_path"]+"/ckpt/"
    if not os.path.exists(ckpt_log_dir):
        os.makedirs(ckpt_log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    lossG_CPU_=1000
    train_Discriminator_flag=True
    cnn = torchvision.models.squeezenet1_1(pretrained=True).features
    cnn=cnn.type(dtype)
    set_requires_grad(cnn,False)
    for epoch in range(num_epochs):
        for i, (real_A_dict, real_B_dict,real_C_dict) in enumerate(zip(loader_style, loader_content,other_Loader)):
            nn=nn+1
#             print(real_A.keys(),real_A["rgb"].shape,real_A["mask"].shape)
            real_A=real_A_dict["rgb"]
            real_A=real_A.type(dtype)
        
            real_A_mask=real_A_dict["mask"].type(dtype)
            real_A_mask=real_A_mask.type(dtype)
            
            real_B=real_B_dict["rgb"]
            real_B=real_B.type(dtype)
            
            real_B_mask=real_B_dict["mask"]
            real_B_mask=real_B_mask.type(dtype)
            
            real_C=real_C_dict["rgb"]
            real_C=real_C.type(dtype)
            if train_Discriminator_flag:
                set_requires_grad(G_A,False)
                set_requires_grad(G_B,False)
                set_requires_grad([D_A,D_B],True) 
#                 print(data.keys(),"any key why none typebefoe???")
                data.update({"real_A":real_A,"real_B":real_B,"real_A_mask":real_A_mask\
                             ,"real_B_mask":real_B_mask,\
                             "D_A":D_A,"D_B":D_B,"G_A":G_A,"G_B":G_B})
#                 print(data.keys(),"any key why none type???")
                loss_D=cyclegan_ls_discriminator_loss(**data)
                D_solver.zero_grad()
                loss_D.backward() 
                D_solver.step()
            
            
            
            set_requires_grad([G_A,G_B],True)
            set_requires_grad([D_A,D_B],False) 
            data.update({"real_A":real_A,"real_B":real_B,"D_A":D_A,"D_B":D_B,"G_A":G_A,"G_B":G_B})
           
            data["cnn"]=cnn

            loss_G=cyclegan_generator_loss(**data)
            G_solver.zero_grad()
            loss_G.backward()
            G_solver.step()
            
            lossG_CPU=loss_G.cpu()
            if lossG_CPU/lossG_CPU_<0.99:
                lossG_CPU_=lossG_CPU
                train_Discriminator_flag=True
            else:
                train_Discriminator_flag=False
#             lossD_CPU_=lossD_CPU
            if i%save_iter==0:
                img_logdir=log_dir+"itration_"+str(nn)
                if not os.path.exists(img_logdir):
                    os.makedirs(img_logdir)
                reconstructed_A=G_A(real_A)
                reconstructed_B=G_B(real_B)
                generated_B_from_A=G_B(real_A)
                generated_A_from_B=G_A(real_B)
                generated_A_from_C=G_A(real_C)
                generated_B_from_C=G_B(real_C)    
                cycle_A=G_A(G_B(real_A))
                cycle_B=G_B(G_A(real_B))
                print(loss_D.cpu(),"loss of discriminator")
                print(loss_G.cpu(),"loss of generator")
                plt.figure()
#                 print(real_A.size())
                imgs = real_A[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"/real_A.png")
#                 plt.show()
                
                plt.close()
                
                plt.figure()
#                 print(real_A.size())
                imgs = reconstructed_A[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"/reconstructed_A.png")
#                 plt.show()
                plt.close()
                
                plt.figure()
#                 print(real_A.size())
                imgs = real_B[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"/real_B.png")
#                 plt.show()
            
                plt.close()
  

                plt.figure()
#                 print(real_A.size())
                imgs = reconstructed_B[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"/reconstructed_B.png")
#                 plt.show()
                plt.close()
    
                plt.figure()
#                 print(real_A.size())

                imgs = real_C[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"/realc.png")
#                 plt.show()
                plt.close()
                plt.figure()
#                 print(real_A.size())

                imgs =generated_A_from_C[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"/generated_A_from_C.png")
#                 plt.show()
                plt.close()
    
                imgs =generated_B_from_C[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"/generated_B_from_C.png")
#                 plt.show()
                plt.close()

                plt.figure()
#                 print(real_A.size())
                imgs = generated_B_from_A[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"/generated_B_from_A.png")
#                 plt.show()
            
                plt.close()
    
                plt.figure()
#                 print(real_A.size())
                imgs = generated_A_from_B[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"/generated_A_from_B.png")
#                 plt.show()
            
                plt.close()   

                plt.figure()
#                 print(real_A.size())
                imgs = cycle_A[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"/cycle_A.png")
#                 plt.show()
            
                plt.close()         
                plt.figure()
#                 print(real_A.size())
                imgs = cycle_B[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"/cycle_B.png")
#                 plt.show()
            
                plt.close()         
                ckpt=ckpt_log_dir+"/"
                if not os.path.exists(ckpt):
                        os.makedirs(ckpt)
                torch.save({
                    'G_A_state_dict': G_A.state_dict(),
                    'G_B_state_dict': G_B.state_dict(),
                    'D_A_state_dict': D_A.state_dict(),
                    'D_B_state_dict': D_B.state_dict(),
            },ckpt+str(nn)+".ckpt")
          

def cycle_gan(**data):
    dtype = data["type"]
    if "full_conv" in data:
        encoder=ResnetEncoder_total_conv(**data)
        encoder.type(data["type"])
        summary(encoder,input_size=(3,data["size"],data["size"]))
        decoder_A=ResnetDecoder_total_conv(**data)
        decoder_A.type(data["type"])
        summary(decoder_A,input_size=(1024,1,1))
        decoder_B=ResnetDecoder_total_conv(**data)
        D_A=build_dc_classifier(**data).type(dtype)
        summary(D_A,input_size=(3,data["size"],data["size"]))
    elif "shaoanlu" in data:
        encoder=ResnetEncoder_shaoanlu(**data)
        encoder.type(data["type"])
        summary(encoder,input_size=(3,data["size"],data["size"]))
        decoder_A=ResnetDecoder_shaoanlu(**data)
        decoder_A.type(data["type"])

        decoder_B=ResnetDecoder_shaoanlu(**data)
        D_A=build_dc_classifier(**data).type(dtype)

                
    else:
        encoder=ResnetEncoder_full(**data)
        decoder_A=ResnetDecoder_full(**data)
        decoder_B=ResnetDecoder_full(**data)
        D_A=build_dc_classifier(**data).type(dtype)
        
    G_A = build_dc_generator(encoder,decoder_A).type(dtype)
    print("G_A graph")
    summary(G_A,input_size=(3,data["size"],data["size"]))
    G_B = build_dc_generator(encoder,decoder_B).type(dtype)
    G_A.apply(initialize_weights)
    G_B.apply(initialize_weights)
    
    D_A.apply(initialize_weights)
    print("D_A graph")
    summary(D_A,input_size=(3,data["size"],data["size"]))
    D_B=build_dc_classifier(**data).type(dtype)
    D_B.apply(initialize_weights)
    G_solver = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder_A.parameters(), decoder_B.parameters()) ,lr=1e-3,betas=[0.5,0.999])
    D_solver = torch.optim.Adam(itertools.chain(D_B.parameters(), D_A.parameters()),lr=1e-3,betas=[0.5,0.999])
    return G_A,D_A, G_B,D_B,  G_solver, D_solver

if __name__ == "__main__":
#     log_dir=Logger(data["ckpt_path"]+"/")
#     sys.stdout=log_dir
    parse=faceswapping_parser()
    parse=parse.parse_args()
    print(parse)
    data=json.load(open(parse.json_file))
   
    os.environ["CUDA_VISIBLE_DEVICES"] = str(data["GPU"])
    data.update({"input_nc":3, "output_nc":3, "ngf":128, "norm_layer":nn.BatchNorm2d, "use_dropout":False, "n_blocks":2, "padding_type":'reflect'})

    data["type"]=torch.cuda.FloatTensor
    data["shaoanlu"]=False
    date_a = datetime.now() 
    folder_name=str(date_a).replace(" ","-")
    log_dir=data["log"]+folder_name
    data["ckpt_path"]=log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout=Logger(log_dir+"/")
    G_A,D_A, G_B,D_B,  G_solver, D_solver=cycle_gan(**data)
    dtype = torch.cuda.FloatTensor
    data["dtype"]=torch.cuda.FloatTensor
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    data_transforms = {
    'train': T.Compose([
#     T.RandomResizedCrop(256),
#        T.RandomRotation(10),
#     T.RandomHorizontalFlip(),
       T.Resize((data["size"],data["size"])),
#     T.RandomAffine(10,shear=10),
#     T.ColorJitter(0.2,0.2,0.2),
    T.ToTensor()
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': T.Compose([
    T.Resize(300),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),}

    # monet="/raid/taoyang/research/research_everyday/repository/pytorch-CycleGAN-and-pix2pix/datasets/monet2photo/trainA/"
    # photo="/raid/taoyang/research/research_everyday/repository/pytorch-CycleGAN-and-pix2pix/datasets/monet2photo/trainB/"
    rgb_A=data["rgb_A"]
    rgb_C=data["rgb_C"]
    rgb_B=data["rgb_B"]
    my_transforms=my_segmentation_transforms
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    mask_A=data["mask_A"]
    mask_B=data["mask_B"]
    A=cycle_data_withfolder_mask(rgb_A,mask_A,my_transforms,data)
    B=cycle_data_withfolder_mask(rgb_B,mask_B,my_transforms,data)
    C_=cycle_data_withfolder(rgb_C,data_transforms["train"])
#     data={"size":64}
#     single=cycle_data_withfolder_mask(A,B,my_segmentation_transforms,data)
    loader_A = DataLoader(A,
                        batch_size=data["batch_size"],
                        num_workers=7,
                        shuffle=True)
    # loader_content = DataLoader(content,
    #                     batch_size=8,
    #                     num_workers=7,
    #                     shuffle=True)
    loader_B= DataLoader(B,
                        batch_size=data["batch_size"],
                        num_workers=7,
                        shuffle=True)
    # style=cycle_data('styles/starry_night.jpg',data_transforms["train"])
    loader_C = DataLoader(C_,
                        batch_size=data["batch_size"],
                        num_workers=7,
                        shuffle=True)
    L1=torch.nn.L1Loss()

#     data={}
    
    data["A"]=loader_A
    data["B"]=loader_B
    data["C"]=loader_C
#     date_a = datetime.now() 
#     folder_name=str(date_a).replace(" ","-")
#     log_dir="../../logs/"+folder_name
    data["ckpt_path"]=log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    import sys
    sys.path.append("/home/taoyang/research/Tao_lib/")
    ls_discriminator_loss=ls_discriminator_loss
    ls_generator_loss=ls_generator_loss
    run_a_cyclegan(G_A,D_A, G_B,D_B,  G_solver, D_solver, ls_discriminator_loss, ls_generator_loss,data=data, num_epochs=50)