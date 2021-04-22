from ram_utils.pytorch_utils.pytorch_utils import CustomDataSet,preview_img,save_ckp,load_ckp,use_cuda


import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataloader
from torchvision import transforms
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

import matplotlib.pyplot as plt

# inv_normalize = transforms.Normalize(
#     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#     std=[1/0.229, 1/0.224, 1/0.255]
# )
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

# CUDA for PyTorch

device = use_cuda

torch.backends.cudnn.benchmark = True


#write on tensorboard 
# from torch.utils.tensorboard import SummaryWriter
# tens_writer = SummaryWriter(log_dir='log_output')

#Writer will output to ./runs/ directory by default.

if not os.path.exists('./cust_dc_img'):
    os.mkdir('./cust_dc_img')



def cust_imshow(img):
    #torch.tensor data
    print(img[0])
    img = img / 2 + 0.5    # unnormalize
    print(type(img))
    npimg = img.numpy()   #converts images to numpy
    print("preview with imshow methode")
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #convert from tensor image
    plt.show()

def create_dir(dir_name):
    os.makedirs(dir_name) if not os.path.exists(dir_name) else print("Directory Exist...\n Continue Trainning..")

# def preview_img(pic,filename:str, dir_path:str=None):
#     create_dir(dir_path)
#     # now = datetime.now()
#     # dt_string = now.strftime("%d-%m-%Y %H:%M")
#     # dir_path =os.path.join(dir_path,dt_string)
#     # create_dir(dir_path)
#     for filename,i in zip (filename,pic):
#         im_numpy = inv_normalize(i)
#         im_ram = np.transpose(im_numpy,(1,2,0))
#         im_ram = im_ram.numpy()
        
#         # cv2.imshow("inside preview",im_ram) 
#         # cv2.waitKey(0) 
#         # print(im_ram.shape)
#         if dir_name is not None:
#             cv2.imwrite(os.path.join(dir_path,f"{filename}"),np.round(im_ram*255))
        

#         # im_ram= cv2.cvtColor(im_ram,cv2.COLOR_BGR2RGB) 
#         # plt.imshow(im_ram) 
#         # plt.gca().set_axis_off()
#         # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#         #             hspace = 0, wspace = 0)
#         # plt.margins(0,0)
#         # plt.gca().xaxis.set_major_locator(plt.NullLocator())
#         # plt.gca().yaxis.set_major_locator(plt.NullLocator())
#         # plt.savefig(os.path.join(dir_path,f"mat{filename[0]}"), bbox_inches = 'tight',
#         #     pad_inches = 0)        
        
#         # plt.savefig(os.path.join(dir_path,f"{count}.png"),pad_inches = 0)
        

#  print(i.shape)
# class CustomDataSet(dataset.Dataset):
#     def __init__(self,main_dir,transform) -> None:
#         self.main_dir = main_dir
#         self.transform = transform
#         if os.path.isdir(main_dir):
#             all_imgs = os.listdir(main_dir)
#         else:
#             self.main_dir,all_imgs=os.path.split(main_dir)
#             all_imgs =[all_imgs]
#             # print(main_dir)
#         self.total_imgs = natsorted(all_imgs)
    
#     def __len__(self):
#         return len(self.total_imgs)

#     def __getitem__(self,idx):
#         file_name = self.total_imgs[idx]
#         img_loc =os.path.join(self.main_dir,self.total_imgs[idx])
#         # image = Image.open(img_loc).convert("RGB")
#         image = Image.open(img_loc)
#         img=cv2.imread(img_loc)
#         tensor_image = self.transform(image)
#         print("cv2values",img)
#         # batch_t = torch.unsqueeze(tensor_image,0)
#         return file_name, tensor_image



train_dir = "/home/pasonatech/workspace/deepNN_py/data_set/carpet/train/good"
test_dir = "/home/pasonatech/workspace/deepNN_py/data_set/carpet/test/good"

#for validation 48 img
ano_validation = "/home/pasonatech/workspace/deepNN_py/data_set/anomoly_val_data"

#for test validation 50 img
test_train_data= "/home/pasonatech/workspace/deepNN_py/data_set/test_train_samples"


inference_dir = "/home/pasonatech/workspace/deepNN_py/carpet_script/inference_result"


inference_test_data = "/home/pasonatech/workspace/deepNN_py/carpet_script/bw_convert_train/inference_result_aug/"


drop_noise = "/home/pasonatech/workspace/deepNN_py/carpet_script/bw_convert_train/drop_gnoise"


train_output = "/home/pasonatech/workspace/deepNN_py/carpet_script/cust_dc_img"
val_output = "/home/pasonatech/workspace/deepNN_py/carpet_script/train_valid"
model_dir = os.path.join("/home/pasonatech/workspace/deepNN_py/carpet_script","model_dir")
create_dir(model_dir)


color_test_dir = "/home/pasonatech/workspace/deepNN_py/data_set/carpet/test/color"

#cut
# color_test_dir="/home/pasonatech/workspace/deepNN_py/data_set/test/hole"

#metal
# color_test_dir="/home/pasonatech/workspace/deepNN_py/data_set/test/thread"


good_test_dir = "/home/pasonatech/workspace/deepNN_py/data_set/test/good"
# train_dir = inference_dir

# ckp_path = "/home/pasonatech/workspace/deepNN_py/carpet_script/bw_convert_train/model_dir/checkpoint_799.pth"

ckp_path = "/home/pasonatech/workspace/deepNN_py/carpet_script/bw_convert_train/aug_backup4-21/model_dir/checkpoint_799.pth"

learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.RandomHorizontalFlip(p=0),
    transforms.RandomVerticalFlip(p=0),
    transforms.RandomRotation(0),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ],inplace=True),
    
    
])


#parameters 
params = {
    'batch_size' : 1,
    'shuffle': False,
    'num_workers': 0
}




class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(                                                   # 3,  1024, 1024
            nn.Conv2d(in_channels = 3, out_channels= 8, kernel_size = 3, padding=1),   #b 8 1024 1024
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size = 2, stride=1,padding=1),
            nn.Conv2d(in_channels = 8, out_channels= 16, kernel_size = 3, padding=1),  # in 16  
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size = 2, stride=2),  # 16 * 512 512
            nn.Conv2d(in_channels = 16, out_channels= 8, kernel_size = 2, padding=1),  # in 16  
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size = 2, stride=1),  # 16 *256
            nn.Conv2d(in_channels = 8, out_channels= 4, kernel_size = 2, padding=1),  # in 16  
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size = 2, stride=2),  # 16 *256
            #flatten 
            # nn.Flatten(),
            # nn.Linear(in_features = np.product([16,256,256]),out_features=1000)
            
           


            #dense layer
        )
        self.decoder = nn.Sequential(
            #dense layer
            # nn.Linear(in_features=1000,out_features=np.product([16,512,512])),
            # View([-1,16,512,512]),
           
            nn.ConvTranspose2d(4, 8, 2,padding=1),  #４ 8
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


start_epoch=0
model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
if os.path.exists(ckp_path):
    model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)


summary(model,(3,1024,1024))
#inference
save_all = []

#load one image at a time:
dir_name =good_test_dir

for i in sorted(os.listdir(dir_name)):
    each_img_path = os.path.join(dir_name,i)
    cust_load_img = CustomDataSet(img_path_or_dir=each_img_path,img_transform=img_transform)
    inferencedataloader = DataLoader(cust_load_img,**params)
    all_inf_loss = []
    for data in inferencedataloader:
        
        filename, img= data
        # print(type(img))
        
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)      

        # ===================log========================
        val_loss = criterion(output, img)
        # ===================backward====================
        #write loss function on tensorboard
        print('Inference Loss [{:4f}]'.format(val_loss))
        all_inf_loss.append(val_loss)

        pic = output.cpu().detach()
        preview_img(pic,filename,os.path.join(inference_test_data,"good_test_data"))
    
    # print(all_inf_loss)




