import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchvision import models
import os
import numpy as np
from torchsummary import summary
def set_requires_grad( nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.
    
    Input:
    - model: A PyTorch model that we want to optimize.
    
    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    optimizer=optim.Adam(model.parameters(),lr=1e-3,betas=[0.5,0.999])
    return optimizer
def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250, 
              batch_size=128, noise_size=96, num_epochs=10):
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
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.type(dtype)
#             print(real_data.size())
            logits_real = D(2* (real_data - 0.5)).type(dtype)

            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()        
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
                imgs_numpy = fake_images.data.cpu().numpy()
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
def build_dc_classifier(**data):
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    """
    size=64
    input_nc=3
    if "size" in data:
        
        size=data["size"]
        input_nc=data["input_nc"]
    N_pow=int(np.log2(size))
    n_firstc=16
    model=[]
    current_size=size
    for  i in range(N_pow//2):
        n_firstc=n_firstc*2
        model+=[nn.Conv2d(input_nc,n_firstc,kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(n_firstc),\
            nn.LeakyReLU(),\
            nn.MaxPool2d(2,2)]
        input_nc=n_firstc
        
        current_size=int(current_size/4)
    model+=[Flatten()]  
    model+=[nn.Linear(current_size*current_size*n_firstc,64,True)]
    model+=[nn.LeakyReLU()]
    model+=[nn.Linear(64,1,True)]         
        
    return nn.Sequential(*model)

def _get_padding(size, kernel_size, stride, dilation):
    padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) //2
    return padding
def build_encoder():
        return nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),  # b, 16, 10, 10
#          Print_net(),
            nn.ReLU(True),
#           Print_net(),
            nn.AvgPool2d(2, stride=2),  # b, 16, 5, 5
#         Print_net(),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
#          Print_net(),
            nn.ReLU(True),
#           Print_net(),
            nn.AvgPool2d(2, stride=2),  # b, 16, 5, 5
#         Print_net(),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
#          Print_net(),
            nn.ReLU(True),
#         Print_net(),
            nn.AvgPool2d(2, stride=2),  # b, 8, 2, 2
#             Print_net(),
            nn.Conv2d(8, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
#          Print_net(),
            nn.ReLU(True),
#         Print_net(),
            nn.AvgPool2d(2, stride=2),  # b, 8, 2, 2
            nn.Conv2d(8, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
#          Print_net(),
            nn.ReLU(True),
#         Print_net(),
            nn.AvgPool2d(2, stride=2)  # b, 8, 2, 2
            
        )

    
def build_decoder():    
    return  nn.Sequential(
#             Print_net(),
            nn.ConvTranspose2d(8, 8, 3, stride=2,padding=1),  # b, 16, 5, 5
#             Print_net(),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 3, stride=2,padding=0),  # b, 16, 5, 5
#             Print_net(),
            nn.ReLU(True),
#             Print_net(),
            nn.ConvTranspose2d(16, 8, 5, stride=2, padding=0),  # b, 8, 15, 15
#             Print_net(),
            nn.ReLU(True),
#             Print_net(),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
#             Print_net(),
            nn.Tanh()
        )
    
def build_dc_generator(encoder,decoder):
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described above.
    """
    return  nn.Sequential(encoder,decoder)
class Print_net(nn.Module):
    def __init__(self):
        super(Print_net, self).__init__()

    def forward(self,x):
        print(x.size())
        return x
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

    
class ResnetDecoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model=[]
        n_downsampling = 2
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(True)]

#         mult = 2 ** n_downsampling
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock_upsampling(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(3)]
        model += [nn.ConvTranspose2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

     
    
class ResnetBlock_upsampling(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock_upsampling, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 1
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.ConvTranspose2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 1
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.ConvTranspose2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

    
    
    
    
class ResnetEncoder_full(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, **data):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type=data["input_nc"],data["output_nc"],\
        data["ngf"],data["norm_layer"],data["use_dropout"],data["n_blocks"],data["padding_type"]
        assert(n_blocks >= 0)
        super( ResnetEncoder_full, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        size=data["size"]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            model += [Self_Attn( ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            model += [Self_Attn(ngf * mult)]
        model+=[nn.Conv2d(ngf * mult, 32, kernel_size=1, padding=0, bias=use_bias)]
        model+=[nn.Flatten()]
        model+=[nn.Linear(size*size//(mult)//(mult)*32,256)]
        model+=[nn.Linear(256,size*size//(2**n_downsampling)//(mult)*ngf)]
        model+=[Reshape(-1,ngf,size//(mult),size//(mult))]
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)    
class ResnetDecoder_full(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, **data):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type=data["input_nc"],data["output_nc"],\
        data["ngf"],data["norm_layer"],data["use_dropout"],data["n_blocks"],data["padding_type"]
        assert(n_blocks >= 0)
        super(ResnetDecoder_full, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model=[]
        size=data["size"]
        n_downsampling = 2
        mult = 2 ** n_downsampling
        model+=[nn.Flatten()]
        model+=[nn.Linear(size*size//(mult)//(mult)*ngf,256)]
        model+=[nn.Linear(256,size*size//(mult)//(mult)*32)]
        
        model+=[Reshape(-1,32,size//(mult),size//(mult))]
        model+=[nn.Conv2d( 32, ngf * mult,kernel_size=1, padding=0, bias=use_bias)]
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(True)]

#         mult = 2 ** n_downsampling
        
        for i in range(n_blocks):       # add ResNet blocks
            model += [Self_Attn( ngf * mult)]
            model += [ResnetBlock_upsampling(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [Self_Attn( int(ngf * mult))]
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            
#         model += [nn.ReflectionPad2d(3)]
        model += [nn.ConvTranspose2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    
class ResnetEncoder_total_conv(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, **data):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type=data["input_nc"],data["output_nc"],\
        data["ngf"],data["norm_layer"],data["use_dropout"],data["n_blocks"],data["padding_type"]
#         data={"input_nc":3, "output_nc":512, "ngf":16, "norm_layer":nn.BatchNorm2d, "use_dropout":False, "n_blocks":6, "padding_type":'reflect',"size":128}
        size=128
        if "size" in data:           
            size=data["size"]
        n_downsampling=int(np.log2(size))
        assert(n_blocks >= 0)
        super( ResnetEncoder_total_conv, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d        
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        n_dim_reduction=0
        for i in range(n_downsampling-1):  # add downsampling layers
            mult = 2 ** i
            if n_dim_reduction<n_downsampling-1:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            n_dim_reduction+=1
            if n_dim_reduction<n_downsampling-1:
                model += [nn.MaxPool2d(2)]
            n_dim_reduction+=1
            if n_dim_reduction<n_downsampling-1:
                model += [ResnetBlock(ngf * mult* 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        model += [nn.MaxPool2d(2)]    
#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):       # add ResNet blocks
#             model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
#         model+=[nn.Conv2d(ngf * mult, 32, kernel_size=1, padding=0, bias=use_bias)]
#         model+=[nn.Flatten()]
#         model+=[nn.Linear(size*size//(mult)//(mult)*32,256)]
#         model+=[nn.Linear(256,size*size//(2**n_downsampling)//(mult)*ngf)]
#         model+=[Reshape(-1,ngf,size//(mult),size//(mult))]
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)      
    
    
    
    
    
class ResnetDecoder_total_conv(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, **data):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type=data["input_nc"],data["output_nc"],\
        data["ngf"],data["norm_layer"],data["use_dropout"],data["n_blocks"],data["padding_type"]
#         data={"input_nc":3, "output_nc":512, "ngf":16, "norm_layer":nn.BatchNorm2d, "use_dropout":False, "n_blocks":6, "padding_type":'reflect',"size":128}
        size=128
        if "size" in data:           
            size=data["size"]

        n_upampling=int(np.log2(size))
        assert(n_blocks >= 0)
        super(ResnetDecoder_total_conv, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model=[]
        mult = 2 ** n_upampling
#         model+=[nn.Flatten()]
#         model+=[nn.Linear(size*size//(mult)//(mult)*ngf,256)]
#         model+=[nn.Linear(256,size*size//(mult)//(mult)*ngf)]
        
#         model+=[Reshape(-1,ngf,size//(mult),size//(mult))]
#         model+=[nn.Conv2d( 32, ngf * mult,kernel_size=1, padding=0, bias=use_bias)]
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(True)]

#         mult = 2 ** n_downsampling
        
#         for i in range(n_blocks):       # add ResNet blocks

#             model += [ResnetBlock_upsampling(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        k_block=int(2**(n_upampling//2))
        shape_in=ngf * k_block*2
        for i in range(n_upampling):  # add upsampling layers
            shape_in=int(shape_in/2)
            mult = 2 ** (k_block - i)
            
            model += [nn.ConvTranspose2d(shape_in, int(shape_in/ 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(shape_in/ 2)),
                      nn.ReLU(True)]
            
#             k_block=int(k_block/2)
            model += [ResnetBlock_upsampling(int(shape_in/ 2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
                                             #         model += [nn.ReflectionPad2d(3)]
            
        model += [nn.ConvTranspose2d(int(shape_in/ 2), output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    
    
    

    
    
    
    
    
    
    
class ResnetEncoder_shaoanlu(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, **data):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type=data["input_nc"],data["output_nc"],\
        data["ngf"],data["norm_layer"],data["use_dropout"],data["n_blocks"],data["padding_type"]
        latent=256
        size=128
        if "size" in data:           
            size=data["size"]
        assert(n_blocks >= 0)
        super(ResnetEncoder_shaoanlu, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
#         model = [nn.ReflectionPad2d(3),
#                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]
        
        n_downsampling = 3
        ngf=64
        input_=3
        mult=1
        activa_map=1
        model=[]
        model += [nn.Conv2d(input_, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult),
                          nn.ReLU(True)]        

        for i in range(0,n_downsampling-1):  # add downsampling layers
            mult = 2 ** i
            input_=ngf * mult
            activa_map*=2    
            model += [nn.Conv2d(input_, ngf * mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult*2),
                          nn.ReLU(True)]
            
        while (size/activa_map>4):
            input_=input_*2
            model += [Self_Attn(input_,)]
            activa_map*=2 
            
            model += [nn.Conv2d(input_, input_*2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(input_*2),
                          nn.ReLU(True)]   
        
        input_=input_*2
        print(input_)
        model+=[nn.Flatten()]
        model+=[nn.Linear(4*4*input_,latent)]
        model+=[nn.Linear(latent,4*4*1024)]
        model+=[Reshape(-1,1024,4,4)]
#         model += [nn.Conv2d(1024, input_*2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(True)]         
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)    
    
    
    
class ResnetDecoder_shaoanlu(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, **data):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        
        input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type=data["input_nc"],data["output_nc"],\
        data["ngf"],data["norm_layer"],data["use_dropout"],data["n_blocks"],data["padding_type"]
        assert(n_blocks >= 0)
        super(ResnetDecoder_shaoanlu, self).__init__()
        n_downsampling = 3
        ngf=64
        input_=3
        mult=1
        activa_map=1
        model=[]
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model=[]
        size=128
        latent=256
        if "size" in data:           
            size=data["size"]
        n_atten=int(np.log2(size//4))
        n_upsampling = 3
        input_=int(ngf*2**n_atten)
        mult = 2 ** n_upsampling
        size_current=4
        model+=[nn.Flatten()]
        model+=[nn.Linear(4*4*1024,latent)]
        model+=[nn.Linear(latent,input_*size_current*size_current)]
        
        model+=[Reshape(-1,input_,size_current,size_current)]
        
        while size_current<=size//mult:
            model += [nn.ReLU(True),norm_layer(int(input_)),
                      nn.ConvTranspose2d(input_, int(input_ / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),Self_Attn(int(input_ / 2))] 
            size_current*=2
            input_=int(input_/2)
#             model += [Self_Attn(input_,)]
#         input_=int(input_/2)
        for i in range(n_upsampling-1):
            model += [norm_layer(int(input_ )),nn.ReLU(True),nn.ConvTranspose2d(input_, int(input_ / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      
                      ] 
            input_=int(input_/2)
        model += [norm_layer(int(input_ )),nn.ReLU(True)]
        model += [nn.ConvTranspose2d(input_, 3,
                                         kernel_size=3, stride=1,
                                         padding=1, output_padding=0,
                                         bias=use_bias)]

        model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    
    
    
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation=nn.ReLU()):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out    
    
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)      
 
if __name__ == "__main__":
    dtype=torch.cuda.FloatTensor
    encoder=ResnetEncoder_full(3,3,n_blocks=2,ngf=32)
    encoder.type(dtype)
    summary(encoder,input_size=(3,128,128))