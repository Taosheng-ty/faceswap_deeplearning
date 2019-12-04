from datetime import datetime
import torch
import os
import sys
sys.path.append("../")
from torch.utils.data import Dataset, DataLoader
from nets import *
from utils import *
import torchvision
import torchvision.transforms as T
def transform():
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
    return data_transforms


def show_results(data):
    model=torch.load(data["weights_path"])
    dtype=data["type"]
    encoder=ResnetEncoder_full(3,3,n_blocks=2,ngf=32)
    decoder_A=ResnetDecoder_full(3,3,n_blocks=2,ngf=32)
    decoder_B=ResnetDecoder_full(3,3,n_blocks=2,ngf=32)
    G_A = build_dc_generator(encoder,decoder_A).type(dtype)
    G_B = build_dc_generator(encoder,decoder_B).type(dtype)
    G_A.load_state_dict(model["G_A_state_dict"])
    G_B.load_state_dict(model["G_B_state_dict"])
    data_transforms=transform()
    img=cycle_data_withfolder(data["img_path"],data_transforms["train"],dtype)
    img_loader= DataLoader(img,
                    batch_size=2,
                    num_workers=1,
                    shuffle=True)
    img_loader_iter=iter(img_loader)
    img=next(img_loader_iter)
    img=img.type(dtype)
    real=img[0,:,:,:].cpu()
    img_A=G_A(img)
    img_B=G_B(img)
    date_a = datetime.now() 
    folder_name=str(date_a).replace(" ","-")
    log_dir=data["log_path"]+folder_name+"/img/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    for i in range(5):
        img=next(img_loader_iter)
        img=img.type(dtype)
        real=img[0,:,:,:].cpu()        
        plt.figure()
    #                 print(real_A.size())
        imgs = real
        img_gene=deprocess(imgs)
        plt.imshow(img_gene)
        plt.title("real.png")
        plt.savefig(log_dir+str(i)+"real.png")
    #     plt.show()

        plt.figure()
    #                 print(real_A.size())
        imgs = img_A.cpu()
        img_gene=deprocess(imgs[0,:,:,:])
        plt.imshow(img_gene)
        plt.title("img_A.png")
        plt.savefig(log_dir+str(i)+"img_A.png")
    #     plt.show()

        plt.figure()
    #                 print(real_A.size())
        imgs = img_B.cpu()
        img_gene=deprocess(imgs[0,:,:,:])
        plt.imshow(img_gene)
        plt.title("img_B.png")
        plt.savefig(log_dir+str(i)+"img_B.png")
    cmd="cd "+log_dir+" \n zip -r "+"../img.zip  "+"*"
    print(cmd)
    os.system(cmd)    
    #     plt.show()
if __name__ == "__main__":
    data={"weights_path":"../../faceswap_deeplearning/nets/logs/2019-12-03-18:00:37.490229/ckpt/93201.ckpt",
     "type":torch.cuda.FloatTensor,
      "img_path":"../../img/", 
      "log_path":"../../logs/test/"}
    print(data)
    show_results(data)