from ram_utils.pytorch_utils.pytorch_utils import CustomDataSet, save_ckp,load_ckp,preview_img,use_cuda,write_log_summary

import torch

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms


import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image
from natsort import natsorted
from torch.utils.data import dataset
import matplotlib.pyplot as plt

from datetime import datetime

now = datetime.now()
#day month year hour month sec

dt_string = now.strftime("%d-%m-%Y %H:%M:%S.%f")



inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255]
)

re_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.255]
)
# inv_tensor = inv_normalize(tensor)






# #save checkpoint
# def save_ckp(state, checkpoint_dir, epoch):
#     f_path = os.path.join(checkpoint_dir , f'checkpoint_{epoch}.pth')
#     torch.save(state, f_path)
#     # if is_best:
#     #     best_fpath = best_model_dir / 'best_model.pth'
#     #     shutil.copyfile(f_path, best_fpath)

# #load checkpoint
# def load_ckp(checkpoint_fpath, model, optimizer):
#     checkpoint = torch.load(checkpoint_fpath)
#     # state_dict = torch.load('./cust_conv_autoencoder.pth')
#     # model.load_state_dict(state_dict)
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     return model, optimizer, checkpoint['epoch']

# # CUDA for PyTorch
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")

device = use_cuda()
torch.backends.cudnn.benchmark = True


# #write on tensorboard 
# from torch.utils.tensorboard import SummaryWriter
# tens_writer = SummaryWriter(log_dir='log_output')

#Writer will output to ./runs/ directory by default.

if not os.path.exists('./cust_dc_img'):
    os.mkdir('./cust_dc_img')



def cust_imshow(img):
    #torch.tensor data
    # print(img[0])
    img = img / 2 + 0.5    # unnormalize
    # print(type(img))
    npimg = img.numpy()   #converts images to numpy
    print("preview with imshow methode")
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #convert from tensor image
    plt.show()

def create_dir(dir_name):
    os.makedirs(dir_name) if not os.path.exists(dir_name) else print("Directory Exist...\n Continue Trainning..")

# def preview_img(pic, dir_path):
#     create_dir(dir_path)
#     for count,i in enumerate (pic):
#         # print(i)
#         # print(i.shape)

#         #new approach
#         untransform = transforms.Compose([
#             transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
#             transforms.ToPILImage,
#         ])
#         # print(type(i))
#         # cust_imshow(i)
#         im_numpy = inv_normalize(i)
#         im_ram = np.transpose(im_numpy,(1,2,0))
#         im_ram = im_ram.numpy()
#         # print(im_ram.shape)
#         cv2.imwrite(os.path.join(dir_path,f"{dt_string}.png"),im_ram*255)
#         cv2.imshow("preview", im_ram)
#         cv2.waitKey(5000)
#         cv2.destroyAllWindows()

#         # plt.imshow(im_ram)
#         # plt.show()
#         # im_ram = untransform(i[0])
#         # print("im_ram_type",type(im_ram))
        
        
#         # plt.imshow(im_ram)
        
#         # plt.tight_layout(pad=0,h_pad=0,w_pad=0,rect=(0,0,0,0))
        
#         # plt.axis('off')
        
#         # plt.savefig(os.path.join(dir_path,f"{dt_string}.png"),pad_inches = 0)
        



# #  print(i.shape)
# class CustomDataSet(dataset.Dataset):
#     def __init__(self,main_dir,transform) -> None:
#         self.main_dir = main_dir
#         self.transform = transform
#         all_imgs = os.listdir(main_dir)
#         self.total_imgs = natsorted(all_imgs)
    
#     def __len__(self):
#         return len(self.total_imgs)

#     def __getitem__(self,idx):
#         img_loc =os.path.join(self.main_dir,self.total_imgs[idx])
#         # image = Image.open(img_loc).convert("RGB")
        
#         image = Image.open(img_loc)
#         # image = cv2.imread(img_loc)
#         # image[0].show("Preview")
#         tensor_image = self.transform(image)
        
#         # print("Getitem",type(tensor_image))       
#         # img = tensor_image
#         # img = inv_normalize(img)    # unnormalize
#         # print(type(img))
#         # npimg = img.numpy()
#         # print("preview with gett method")
#         # plt.imshow(np.transpose(npimg, (1, 2, 0)))
#         # plt.show()


#         return tensor_image

# def to_img(x):
#     x = 0.5 * (x + 1)
#     # x = x.clamp(0, 1)
#     x = x.view(x.size(0), 3, 400, 400)
#     return x



def to_img(x):
    cust_imshow(x[0])
    x = 0.5 * (x + 1)
    # x = x / 2 + 0.5  #unnormalize
   
    print(x[0])
    print(type(x[0]))
    print((x[0].shape))
    # x = x.clamp(0, 1)
    from torchvision.utils import save_image
    save_image(x[0],"img1.png")
   
    # x = x.view(x.size(0), 3, 400, 400)
    return x


train_dir = "/home/pasonatech/workspace/deepNN_py/data_set/carpet/train/good"
test_dir = "/home/pasonatech/workspace/deepNN_py/data_set/carpet/test/good"
ano_validation = "/home/pasonatech/workspace/deepNN_py/data_set/anomoly_val_data"



inference_dir = "/home/pasonatech/workspace/deepNN_py/carpet_script/bw_convert_train/inference_result"
train_output = "/home/pasonatech/workspace/deepNN_py/carpet_script/bw_convert_train/cust_dc_img"
val_output = "/home/pasonatech/workspace/deepNN_py/carpet_script/bw_convert_train/train_valid"

ano_val_output = "/home/pasonatech/workspace/deepNN_py/carpet_script/bw_convert_train/anomaly_val_train"
model_dir = os.path.join("/home/pasonatech/workspace/deepNN_py/carpet_script/bw_convert_train","model_dir")
create_dir(model_dir)


color_test_dir = "/home/pasonatech/workspace/deepNN_py/data_set/carpet/test/color"
# train_dir = inference_dir

# ckp_path = "/home/pasonatech/workspace/deepNN_py/carpet_script/model_dir/checkpoint_2999.pth"

ckp_path = ""



num_epochs = 10
# batch_size = 128
learning_rate = 1e-3


img_transform = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.RandomHorizontalFlip(p=0),
    transforms.RandomVerticalFlip(p=0),
    transforms.RandomRotation(0),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
    
    
])

albu_transform = A.Compose([
    A.OneOf([
        A.CoarseDropout(always_apply=False, p=1.0, max_holes=100, max_height=100, max_width=12, min_holes=8, min_height=8, min_width=8),
        # A.GaussNoise(always_apply=False, p=1.0, var_limit=(0.001, 0.005)),
    ],p=1),
    ToTensorV2(),
    # A.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
    # A.Resize(1024,1024),
        
])




#calling my dataset

#parameters 
params = {
    'batch_size' : 1,
    'shuffle': True,
    'num_workers': 6
}


my_dataset = CustomDataSet(img_path_or_dir=train_dir,img_transform=img_transform)


# print(albuloader)
#about 50 ano validation images
ano_val = CustomDataSet(img_path_or_dir=ano_validation,img_transform=img_transform)


test_ano_color = CustomDataSet(img_path_or_dir=color_test_dir,img_transform=img_transform)

testloader = DataLoader(test_ano_color,**params)
#split data train,test
train,valid = dataset.random_split(my_dataset,[230,50])




# train_loader = DataLoader(my_dataset,**params)

trainloader = DataLoader(train,**params)
valloader = DataLoader(valid,**params)
anovalloader = DataLoader(ano_val, **params)

print("Batach Sized Training Dataset :",len(trainloader))
print("Batch Sized Validation Dataset :",len(valloader))

# get some random training images
# dataiter = iter(trainloader)
# images = dataiter.next()



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(                                                   # 3,  1024, 1024
            nn.Conv2d(in_channels = 3, out_channels= 8, kernel_size = 3, padding=1),   #b 8 1024 1024
            nn.ReLU(True),
            
            nn.Conv2d(in_channels = 8, out_channels= 16, kernel_size = 3, padding=1),  # in 8  out 16  
            nn.ReLU(True),
            
            nn.Conv2d(in_channels = 16, out_channels= 8, kernel_size = 2, padding=1),  # in 16  out 8
            nn.ReLU(True),
            
            nn.Conv2d(in_channels = 8, out_channels= 4, kernel_size = 2, padding=1),  # in 8 out 4  
            nn.ReLU(True),
            
            #flatten 
            # nn.Flatten(),
            # nn.Linear(in_features = np.product([16,256,256]),out_features=1000)
            
           


            #dense layer
        )
        self.decoder = nn.Sequential(
            #dense layer
            # nn.Linear(in_features=1000,out_features=np.product([16,512,512])),
            # View([-1,16,512,512]),
           
            nn.ConvTranspose2d(4, 8, 2,padding=1),  # 16 8
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 2, padding=1),  # 16 8
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, padding=1),  # 16 8
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=1,padding=1),   # 8 3 
            nn.Tanh()
            #nn.Sigmoid()

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




epoch_checkpoint=2

start_epoch=0
model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
if os.path.exists(ckp_path):
    model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)

from skimage.util import random_noise

# import torchvision.transforms.functional as F



for epoch in tqdm(range(start_epoch,num_epochs)):
    total_loss = 0
    for data in trainloader:
        # img, _ = data
        filename,img= data   

        # print(type(img))
        # print(img.shape)

        # print("Image type",type(img))
        aug_img = img.clone()
        aug_img = inv_normalize(aug_img)    # unnormalize
        # print(type(img))
        npimg = aug_img.numpy()
        cvimg = np.transpose(npimg[0], (1, 2, 0))
        cv2.imwrite("drop-original.png",cvimg*255)
        cvimg = albu_transform(image=cvimg)
        
        cvaug_img = cvimg['image']


        # preview augmentation implemented
                
        # cvaug_img = cvaug_img.numpy()
        # cvaug_img = np.transpose(cvaug_img, (1, 2, 0))
        # cv2.imwrite("dropout.png",cvaug_img*255)
        # cv2.imshow("opencv",cvaug_img)       
        # print("preview with gett method")
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()

        
        #normalize the cvaug_img
        cvaug_img = re_normalize(cvaug_img)

        #adding one more dimension and converting from 3D to 4D
        cvaug_img = cvaug_img[None,:,:,:]

        cvaug_img = Variable(cvaug_img).cuda()
        img = Variable(img).cuda()
        # ===================forward=====================
        # output = model(img)

        #adding some noise 
        # print(type(gauss_img))
        # print(gauss_img[0])
        # print(img[0])
        # print("cvaug_img",cvaug_img.shape)
        # print("img",img.shape)
        output = model(cvaug_img)
        
        # print("output",output.shape)
        
        loss = criterion(output, img)
        # ===================backward====================
        #write loss function on tensorboard
        write_log_summary(title="Loss/train",loss=loss,epoch=epoch)
        # tens_writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    total_loss += loss.data
    
    #############################################################
    #############################################################
    #############################################################
    total_val_loss = 0
    for val_data in valloader:
        val_filename,val_img= val_data
        val_img = Variable(val_img).cuda()
        # ===================forward=====================
        validation_output = model(val_img)
        val_loss = criterion(validation_output, val_img)
        # ===================backward====================
        #write loss function on tensorboard
        # tens_writer.add_scalar("Val_Loss/train", val_loss, epoch)
        write_log_summary(title="Val_Loss/train",loss=val_loss,epoch=epoch)
        
    # ===================log========================
    total_val_loss += val_loss.data
    #############################################################
    #############################################################
    #############################################################
    total_ano_val_loss = 0
    for ano_val_data in anovalloader:
        ano_val_filename,ano_val_img= ano_val_data
        ano_val_img = Variable(ano_val_img).cuda()
        # ===================forward=====================
        ano_validation_output = model(ano_val_img)
        ano_val_loss = criterion(ano_validation_output, ano_val_img)
        # ===================backward====================
        #write loss function on tensorboard
        write_log_summary(title="Anomaly Val_Loss/train",loss=ano_val_loss,epoch=epoch)
        # tens_writer.add_scalar("Anomaly Val_Loss/train", ano_val_loss, epoch)
        
    # ===================log========================
    total_ano_val_loss += ano_val_loss.data
    



    print('Epoch [{}/{}], Training loss:{:.4f}, Validation Loss:{:.4f}, Anomaly Val Loss:{:.4f}'
          .format(epoch+1, num_epochs, total_loss,total_val_loss,total_ano_val_loss))


    if (epoch+1) % epoch_checkpoint == 0:  
        pic = output.data.cpu().detach()
        preview_img(pic,filename,os.path.join(train_output,str(epoch)))
        
        #val image samples        
        pic1 = validation_output.data.cpu().detach()
        preview_img(pic1,val_filename,os.path.join(val_output,str(epoch)))


        #ano val image samples        
        pic2 = ano_validation_output.data.cpu().detach()
        preview_img(pic2,ano_val_filename,os.path.join(ano_val_output,str(epoch)))

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
# tens_writer.flush()





