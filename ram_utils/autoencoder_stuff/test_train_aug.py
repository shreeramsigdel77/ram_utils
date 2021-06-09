from matplotlib.colors import Normalize
from albumentations.core.composition import OneOf
from numpy.core.fromnumeric import shape
import torch
from torch import tensor
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataloader
from torchvision import transforms as T
from torchvision.transforms.functional import pad, to_pil_image
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomRotation, Resize, ToPILImage, ToTensor
from torchvision.utils import save_image
from torchvision import transforms


import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from natsort import natsorted
from torch.utils.data import dataset
# import shutil
from tqdm import tqdm 

from autoencoder_networkLayer import autoencoder
#write on tensorboard 

from torch.utils.tensorboard import SummaryWriter

from pyutilities import save_ckp, load_ckp,create_dir,preview_img,tensor_to_pil

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True



from albumentations.pytorch.transforms import ToTensor as py_ToTensor
normalize_dict = {
     "mean":[0.485, 0.456, 0.406],
    "std":[0.229, 0.224, 0.225]
}

albu_transform = A.Compose([
    OneOf([
        
        A.CoarseDropout(always_apply=False, p=0.7, max_holes=1, max_height=64, max_width=64, min_holes=1, min_height=64, min_width=64, fill_value=(0, 0, 0), mask_fill_value=None),
    ],p=0.7),
    py_ToTensor(),
    
])


#inv normalization
inv_normalize = T.Compose([
    T.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
    )
])


re_normalize = T.Compose([
    T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
])



img_preprocess = T.Compose([
   
    T.Resize((256,256)),
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
        # image = Image.open(img_loc).convert("RGB")
        image = Image.open(img_loc)
        # img=cv2.imread(img_loc)
        tensor_image = self.transform(image)
        # print("cv2values",img)
        # batch_t = torch.unsqueeze(tensor_image,0)
        return file_name, tensor_image
#logdir
tensorboard_writer = SummaryWriter(log_dir='log_output3')

train_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/data_set/carpet/train/good"


inference_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/inference_result"
train_output = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/cust_dc_img"
val_output = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/train_valid"


model_dir = os.path.join("/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train","model_dir")
create_dir(model_dir)


ckp_path = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/model_dir/checkpoint_499.pth"




num_epochs = 3000
learning_rate = 1e-3

# learning_rate = 1e-2






#parameters 
params = {
    'batch_size' : 1,
    'shuffle': True,
    'num_workers': 1
}


my_dataset = CustomDataSet(main_dir=train_dir,transform=img_preprocess)



#split data train,test
train,valid = dataset.random_split(my_dataset,[230,50])




# train_loader = DataLoader(my_dataset,**params)

trainloader = DataLoader(train,**params)
valloader = DataLoader(valid,**params)

print("Batach Sized Training Dataset :",len(trainloader))
print("Batch Sized Validation Dataset :",len(valloader))



#view
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,  # extra comma    
    def forward(self, x):
        return x.view(*self.shape)


epoch_checkpoint=1

start_epoch=0
model = autoencoder().cuda()
criterion = nn.MSELoss()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
if os.path.exists(ckp_path):
    model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)



for epoch in tqdm(range(start_epoch,num_epochs)):
    total_loss = 0
    for data in trainloader:
        filename,img= data 

        aug_img = img.clone()
        # print(aug_img.shape)
        #convert torch.tensor to img
        cvimg = tensor_to_pil(aug_img[0])
        #converts back to torch.tensor with the albumentation effects
        cvimg = albu_transform(image=cvimg)

        cvaug_img = cvimg['image']

        
        #adding one more dimension and converting from 3D to 4D
        cvaug_img = cvaug_img[None,:,:,:]


        #preview iamge in tensorboard     
        cvaug_img = cvaug_img.data.cuda().detach()
        # cvaug_img = Variable(cvaug_img).cuda()

        img =img.data.cuda().detach()
        # tens_img = tensor_to_pil(img[0])
        
        # ===================forward=====================    
        output = model(cvaug_img)
        # print("Raw",img)
        # print("Aug",cvaug_img)
        # print("Output",output) 
        
        import cv2
        loss = criterion(output, img)
        tensflow_output_img =output.data.cuda().detach()        
        
        # ===================backward====================
        #write loss function on tensorboard
        tensorboard_writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    #Image and graphs for tensorboard
    grid = torchvision.utils.make_grid(img)
    tensorboard_writer.add_image('Images',grid,epoch)
    tensorboard_writer.add_graph(model,img)
    
    grid2 = torchvision.utils.make_grid(cvaug_img)
    tensorboard_writer.add_image('Aug Effects',grid2,epoch)
    tensorboard_writer.add_graph(model,cvaug_img.float())

    grid1 = torchvision.utils.make_grid(tensflow_output_img)
    tensorboard_writer.add_image('Output Images',grid1,epoch)
    tensorboard_writer.add_graph(model,tensflow_output_img)
    
    total_loss += loss.data
    
    #############################################################
    #############################################################
    #############################################################
    total_val_loss = 0
    for val_data in valloader:
        val_filename,val_img= val_data
        val_img = val_img.data.cuda().detach()
        # val_img = Variable(val_img).cuda()
        # print("Val_fileName", val_filename)
        # ===================forward=====================
        validation_output = model(val_img)
      
        validation_output = validation_output.data.cuda().detach()
      
        val_loss = criterion(validation_output, val_img)
        # print("val_loss",val_loss)

        # ===================backward====================
        #write loss function on tensorboard
        tensorboard_writer.add_scalar("Val_Loss/train", val_loss, epoch)

    grid3 = torchvision.utils.make_grid(validation_output)
    tensorboard_writer.add_image('Validation Images',grid3,epoch)
    tensorboard_writer.add_graph(model,validation_output) 
    # ===================log========================
    total_val_loss += val_loss.data
    #############################################################
    #############################################################
    #############################################################



    print('Epoch [{}/{}], Training loss:{:.4f}, Validation Loss:{:.4f}'
          .format(epoch+1, num_epochs, total_loss,total_val_loss,))


    if (epoch+1) % epoch_checkpoint == 0:  
        pic = output.data.cpu().detach()
        preview_img(pic,filename,os.path.join(train_output,str(epoch)),"Output_Img")
        
        #val image samples        
        pic1 = validation_output.data.cpu().detach()
        preview_img(pic1,val_filename,os.path.join(val_output,str(epoch)),"Val_Output_IMG")

        #save models
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, model_dir,epoch)
        if epoch+1 == num_epochs:
            save_ckp(checkpoint, model_dir,"final")

#flush make sure all the pending events have been written to disk
# tb.close()
tensorboard_writer.flush()
tensorboard_writer.close()


