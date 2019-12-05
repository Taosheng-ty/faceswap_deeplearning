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
sys.path.append("../")
import sys
sys.path.append("/home/taoyang/research/Tao_lib/")
from Tao_lib.log import Logger

from utils import *
from utils.loss import ls_generator_loss,ls_discriminator_loss

dtype_float=torch.FloatTensor
# device0 = torch.device("cuda:0")
# device1 = torch.device("cuda:1")
# dtype = torch.cuda.HalfTensor






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
    nn=0
    save_iter=200
#     loader_style_iter=iter(loader_content)
    
        
    loader_content=data["A"]
    loader_style=data["B"]
    other_Loader=data["C"]
    log_dir=Logger(data["ckpt_path"]+"/")
    sys.stdout=log_dir
    log_dir=data["ckpt_path"]+"/img/"
    ckpt_log_dir=data["ckpt_path"]+"/ckpt/"
    if not os.path.exists(ckpt_log_dir):
        os.makedirs(ckpt_log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    for epoch in range(num_epochs):
        for i, (real_A, real_B,real_C) in enumerate(zip(loader_style, loader_content,other_Loader)):
            nn=nn+1
#             print(real_B.size())
#             print(real_A.size())
            real_A=real_A.type(dtype)
            real_B=real_B.type(dtype)
            real_C=real_C.type(dtype)
            set_requires_grad(G_A,False)
            set_requires_grad(G_B,False)
            set_requires_grad([D_A,D_B],True)
            real_B=real_B.type(dtype)
#         logits_real = D(2* (real_data - 0.5)).type(dtype)
            fake_B=G_A(real_A)
#             print(fake_B.size(),"b size")
            fake_A=G_B(real_B)
            rec_A=G_B(fake_B)
            rec_B=G_A(fake_A)
            idt_A=G_B(real_A)
            idt_B=G_A(real_B)
            rec_AA=G_A(idt_A)
            rec_BB=G_B(idt_B)
            idt_A_Ascore=D_A(idt_A.detach())
#             rec_A_score=D_A(idt_A.detach())
            idt_A_Bscore=D_B(idt_A.detach())
            idt_B_Ascore=D_A(idt_B.detach())
#             rec_A_score=D_A(idt_A.detach())
            idt_B_Bscore=D_B(idt_B.detach())
            real_B_score=D_A(real_B)
            fake_B_score=D_A(fake_B.detach())
            real_A_score=D_B(real_A)
            real_AA_score=D_A(real_A)
            real_BB_score=D_B(real_B)
#             print(real_A_score.cpu().size())
            fake_A_score=D_B(fake_A.detach())
            
#             loss_DA=ls_discriminator_loss(real_B_score,fake_B_score)+ls_discriminator_loss(idt_A_Ascore,idt_A_Bscore)
            
# #             print("segem")
#             loss_DB=ls_discriminator_loss(real_A_score,fake_A_score)+ls_discriminator_loss(idt_B_Bscore,idt_B_Ascore)
#             loss_D=(loss_DA+loss_DB)*2

            loss_DA=ls_discriminator_loss(real_AA_score,fake_B_score)
            
#             print("segem")
            loss_DB=ls_discriminator_loss(real_BB_score,fake_A_score)
            loss_D=(loss_DA+loss_DB)*2




            ##here I should decide what kinds of loss matters and how to calculate the loss.

            D_solver.zero_grad()
#             DB_solver.zero_grad()
#             loss_D.backward(retain_graph=True) 
            loss_D.backward() 
            D_solver.step()
#             DB_solver.zero_grad()
#             DA_solver.step()
            set_requires_grad([G_A,G_B],True)
            set_requires_grad([D_A,D_B],False)
            fake_A=G_B(real_B)
            fake_B=G_A(real_A)
            idt_A=G_B(real_A)
            idt_B=G_A(real_B)
            rec_B=G_A(fake_A)
            rec_A=G_B(fake_B)
            fake_B_score=D_A(fake_B)
            fake_A_score=D_B(fake_A)
            loss_G=ls_generator_loss(fake_B_score)+ls_generator_loss(fake_A_score)+(L1(fake_B,real_A)+L1(fake_A,real_B))*10+5*(L1(rec_AA,real_A)+L1(rec_BB,real_B))

            G_solver.zero_grad()
            loss_G.backward()

            G_solver.step()
#             mean_running=loss_D+mean_running
            if i%save_iter==0:
#                 print(real_B_score.cpu(),fake_B_score.cpu())
                print(loss_D.cpu(),"loss of discriminator")
                print(loss_G.cpu(),"loss of generator")
                plt.figure()
#                 print(real_A.size())
                imgs = real_A[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"real_A.png")
#                 plt.show()
                
                plt.close()
                
                plt.figure()
#                 print(real_A.size())
                imgs = fake_B[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"fake_B.png")
#                 plt.show()
                plt.close()
                
                plt.figure()
#                 print(real_A.size())
                imgs = real_B[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"real_B.png")
#                 plt.show()
            
                plt.close()
                
                plt.figure()
#                 print(real_A.size())
                imgs = fake_A[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"fake_A.png")
#                 plt.show()
                plt.close()
                G_A_C=G_A(real_C)
                G_B_C=G_B(real_C)
                plt.figure()
#                 print(real_A.size())

                imgs = real_C[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"realc.png")
#                 plt.show()
                plt.close()
                plt.figure()
#                 print(real_A.size())

                imgs = G_A_C[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"G_A_C.png")
#                 plt.show()
                plt.close()
    
                imgs = G_B_C[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"G_B_C.png")
#                 plt.show()
                plt.close()
       
    
    
    
    
    
    
                plt.figure()
#                 print(real_A.size())
                imgs = idt_A[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"idt_A.png")
#                 plt.show()
            
                plt.close()
    
                plt.figure()
#                 print(real_A.size())
                imgs = idt_B[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"idt_B.png")
#                 plt.show()
            
                plt.close()   

                plt.figure()
#                 print(real_A.size())
                imgs = rec_BB[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"rec_BB.png")
#                 plt.show()
            
                plt.close()         
                plt.figure()
#                 print(real_A.size())
                imgs = rec_AA[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"rec_AA.png")
#                 plt.show()
            
                plt.close()         
                
                plt.figure()
#                 print(real_A.size())
                imgs = rec_B[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"rec_B.png")
#                 plt.show()
            
                plt.close()         
                plt.figure()
#                 print(real_A.size())
                imgs = rec_A[0,:,:,:].cpu()
                imgs=imgs.type(dtype_float)
                img_gene=deprocess(imgs)
                plt.imshow(img_gene)
                plt.savefig(log_dir+"itration_"+str(nn)+"rec_A.png")
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
          

def cycle_gan():
    dtype = torch.cuda.FloatTensor
    
    encoder=ResnetEncoder_full(3,3,n_blocks=2,ngf=32)
    decoder_A=ResnetDecoder_full(3,3,n_blocks=2,ngf=32)
    decoder_B=ResnetDecoder_full(3,3,n_blocks=2,ngf=32)
    G_A = build_dc_generator(encoder,decoder_A).type(dtype)
    G_B = build_dc_generator(encoder,decoder_B).type(dtype)
    G_A.apply(initialize_weights)
    G_B.apply(initialize_weights)
    D_A=build_dc_classifier().type(dtype)
    D_A.apply(initialize_weights)
    D_B=build_dc_classifier().type(dtype)
    D_B.apply(initialize_weights)
    G_solver = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder_A.parameters(), decoder_B.parameters()) ,lr=1e-3,betas=[0.5,0.999])
    D_solver = torch.optim.Adam(itertools.chain(D_B.parameters(), D_A.parameters()),lr=1e-3,betas=[0.5,0.999])
    return G_A,D_A, G_B,D_B,  G_solver, D_solver
if __name__ == "__main__":
    
    parse=faceswapping_parser()
    print(parse)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    G_A,D_A, G_B,D_B,  G_solver, D_solver=cycle_gan()
    dtype = torch.cuda.FloatTensor
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    data_transforms = {
    'train': T.Compose([
    # T.RandomResizedCrop(256),
    T.RandomHorizontalFlip(),
       T.Resize((128,128)),
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

    # monet="/raid/taoyang/research/research_everyday/repository/pytorch-CycleGAN-and-pix2pix/datasets/monet2photo/trainA/"
    # photo="/raid/taoyang/research/research_everyday/repository/pytorch-CycleGAN-and-pix2pix/datasets/monet2photo/trainB/"
    monet="/home/taoyang/research/research_everyday/faceswap-GAN/faceA/rgb/"
    photo="/home/taoyang/research/research_everyday/faceswap-GAN/faceB/rgb/"
    other="/home/taoyang/research/datasets/lfw/Ari_Fleischer/"
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    content=cycle_data_withfolder(monet,data_transforms["train"],dtype)
    other_=cycle_data_withfolder(other,data_transforms["train"],dtype)
    style=cycle_data_withfolder(photo,data_transforms["train"],dtype)

    


# encoder=build_encoder()
# decoder_B=build_decoder()
# decoder_A=build_decoder()
# G_A = build_dc_generator(encoder,decoder_A).type(dtype)
# G_A = ResnetGenerator(3,3,n_blocks=2).type(dtype)
# G_A.apply(initialize_weights)
# D_A=build_dc_classifier().type(dtype)
# D_A.apply(initialize_weights)

# # G_B = build_dc_generator(encoder,decoder_B).type(dtype)
# # G_B =ResnetGenerator(3,3,n_blocks=2).type(dtype)
# # G_B.apply(initialize_weights)
# D_B=build_dc_classifier().type(dtype)
# D_B.apply(initialize_weights)
# GA_solver, DA_solver,GB_solver,DB_solver=get_optimizer(G_A),get_optimizer(D_A),get_optimizer(G_B),get_optimizer(D_B)
# itertools.chain(encoder.parameters(), decoder_A.parameters(), decoder_B.parameters())
# G_solver = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder_A.parameters(), decoder_B.parameters()) ,lr=1e-3,betas=[0.5,0.999])
# G_solver = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()) ,lr=1e-3,betas=[0.5,0.999])
# D_solver = torch.optim.Adam(itertools.chain(D_B.parameters(), D_A.parameters()),lr=1e-3,betas=[0.5,0.999])

# content=cycle_data('styles/tubingen.jpg',data_transforms["train"])
    loader_content = DataLoader(content,
                        batch_size=4,
                        num_workers=7,
                        shuffle=True)
    # loader_content = DataLoader(content,
    #                     batch_size=8,
    #                     num_workers=7,
    #                     shuffle=True)
    other_Loader= DataLoader(other_,
                        batch_size=4,
                        num_workers=7,
                        shuffle=True)
    # style=cycle_data('styles/starry_night.jpg',data_transforms["train"])
    loader_style = DataLoader(style,
                        batch_size=4,
                        num_workers=7,
                        shuffle=True)
    L1=torch.nn.L1Loss()
    date_a = datetime.now() 
    data={}
    
    data["A"]=loader_content
    data["B"]=loader_style
    data["C"]=other_Loader
    folder_name=str(date_a).replace(" ","-")
    log_dir="../../logs/"+folder_name
    data["ckpt_path"]=log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    import sys
    sys.path.append("/home/taoyang/research/Tao_lib/")
#     from Tao_lib.log import Logger
#     log=Logger(folder="logs/"+folder_name+"/")
#     sys.stdout=log
    ls_discriminator_loss=ls_discriminator_loss
    ls_generator_loss=ls_generator_loss
    run_a_cyclegan(G_A,D_A, G_B,D_B,  G_solver, D_solver, ls_discriminator_loss, ls_generator_loss,data=data, num_epochs=50)