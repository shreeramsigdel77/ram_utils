import torch
from torch import tensor
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataloader
from torchvision import transforms as T
from torchvision.transforms.functional import pad, to_pil_image
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomRotation, Resize, ToPILImage
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import cv2
import numpy as np
import albumentations as A
from torchsummary import summary

from PIL import Image
from natsort import natsorted
from torch.utils.data import dataset
import matplotlib.pyplot as plt
# import shutil

from datetime import datetime
from autoencoder_networkLayer import autoencoder
from pyutilities import save_ckp, load_ckp,pil_img_to_tensor,create_dir,preview_img,image_np_array,tensor_to_pil
from tqdm import tqdm as tq
#day month year hour month sec




# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

torch.backends.cudnn.benchmark = True


#write on tensorboard 
from torch.utils.tensorboard import SummaryWriter
tens_writer = SummaryWriter(log_dir='log_output3')

#Writer will output to ./runs/ directory by default.

if not os.path.exists('./cust_dc_img'):
    os.mkdir('./cust_dc_img')



      
        

class CustomDataSet(dataset.Dataset):
    def __init__(self,main_dir,transform) -> None:
        self.main_dir = main_dir
        self.transform = transform
        if os.path.isdir(main_dir):
            all_imgs = os.listdir(main_dir)
        else:
            self.main_dir,all_imgs=os.path.split(main_dir)
            all_imgs =[all_imgs]
            # print(main_dir)
        self.total_imgs = natsorted(all_imgs)
    
    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self,idx):
        file_name = self.total_imgs[idx]
        img_loc =os.path.join(self.main_dir,self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return file_name, tensor_image

data_type = ["good_test","color","cut","thread","metal_contamination","hole"] 

train_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/data_set/carpet/train/good"
# test_dir = f"/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/patch_images/{data_type[0]}"

test_dir = f"//home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/patch_images/good/"
# test_dir =f"/home/nabu/Desktop/aug_img"
# output dir 
# output_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/patch_inference_img"
output_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/dropout_aug_inference/aug_inference"

#output dir for different test folders
output_dir = os.path.join(output_dir,test_dir.split('/')[-1])
create_dir(output_dir)

ckp_path = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/model_dir/checkpoint_499.pth"


learning_rate = 1e-3

# learning_rate = 1e-2


#parameters 
params = {
    'batch_size' : 1,
    'shuffle': False,
    'num_workers': 1
}


img_preprocess = T.Compose([
    # T.RandomResizedCrop()
    # T.RandomCrop((64,64)),
    # T.Resize((800,1024)),
    T.RandomHorizontalFlip(p=0),
    T.RandomVerticalFlip(p=0),
    T.RandomRotation(0),
    T.ToTensor(),  #converts data from 0-255 to 0-1
    # T.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),  #converts into range of -1 to 0 to 1
    
])

#note
    # T.Resize: PIL image in, PIL image out.
    # T.ToTensor: PIL image in, PyTorch tensor out.
    # T.Normalize: PyTorch tensor in, PyTorch tensor out.

reverse_preprocess = T.Compose([
    #inverse of normalize if you have normalized 
    # T.Normalize(
    # mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    # std=[1/0.229, 1/0.224, 1/0.225]
    # ), 
    T.ToPILImage(),
    np.array,
])







start_epoch=0
model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
if os.path.exists(ckp_path):
    model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)

dir_name = test_dir

for j in tq(sorted(os.listdir(dir_name))):
    # print(dir_name)
    output_dir1 = os.path.join(output_dir,j)
    create_dir(output_dir1)
    
    path_sub_dir = os.path.join(dir_name,str(j))
    for i in sorted(os.listdir(path_sub_dir)):        
        each_img_path = os.path.join(path_sub_dir,i)
        cust_load_img = CustomDataSet(main_dir=each_img_path,transform=img_preprocess)
        inferencedataloader = DataLoader(cust_load_img,**params)
        
        for data in inferencedataloader:
            filename, img= data  
            # filename = filename[0]        
            img = img.data.cuda().detach()
            
            # ===================forward=====================
            output = model(img) 
            # output=output.data.cuda().detach() 
            
            # print("Output Type", type(output))   
            # print(output)
           
            # output = np.array(tensor_to_pil(output[0]))
            # print(output)
            # output = cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
            # cv2.imshow("Output Window",output)
            
            # cv2.waitKey(0)
            # exit()
            # ===================log========================
            # val_loss = criterion(output, img)
            # ===================backward====================
            #write loss function on tensorboard
            # print('Inference Loss [{:4f}]'.format(val_loss))
            inference_img =output.data.cpu().detach() 
            # print(filename)
            preview_img(inference_img,filename,output_dir1,"Inference Preview")

       



              
  
       


