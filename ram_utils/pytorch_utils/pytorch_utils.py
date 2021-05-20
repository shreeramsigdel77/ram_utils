from distutils import filelist
import os
import cv2
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import dataset
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

def write_log_summary(title:str,loss, epoch,log_dir_path:str ="log_output"):
    """[summary
    
    e.g write_log_summary("Train_Loss/train", train_loss, epoch)]
    ]

    Args:
        title (str): [Tensorflow graph title )]
        loss ([type]): [description]
        epoch ([type]): [description]
    """
    
    tens_writer = SummaryWriter(log_dir=log_dir_path)
    tens_writer.add_scalar(title, loss, epoch)


def use_cuda():
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda:0" if use_cuda else "cpu")


class CustomDataSet(dataset.Dataset):
    """[Pytorch custom dataset reader: 
    Provide Directory URL for training and for Testing individual image path to reduce the memory full error
    ]

    Args:
        dataset ([type]): [description]

        returns file name and tensor_image in list 
        Sample of img_transform
        img_transform = transforms.Compose([
            transforms.Resize((1024,1024)),
            transforms.RandomHorizontalFlip(p=0),
            transforms.RandomVerticalFlip(p=0),
            transforms.RandomRotation(0),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ],inplace=True),
        ])

        returns file_name and tensor_image in list

    """
    
    def __init__(self,img_path_or_dir,img_transform) -> None:
        self.main_dir = img_path_or_dir
        self.transform = img_transform
        if os.path.isdir(self.main_dir):
            all_imgs = os.listdir(self.main_dir)
        else:
            self.main_dir,all_imgs=os.path.split(self.main_dir)
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
        tensor_image = self.transform(image)
        return file_name, tensor_image



#save checkpoint
def save_ckp(state, checkpoint_dir, epoch):
    """[To save the weight file trained with pytorch]


    Args:
        state ([type]): [description]
        checkpoint_dir ([type]): [Path to save directory]
        epoch ([type]): [current epoch]
    """
    f_path = os.path.join(checkpoint_dir , f'checkpoint_{epoch}.pth')
    torch.save(state, f_path)
    # if is_best:
    #     best_fpath = best_model_dir / 'best_model.pth'
    #     shutil.copyfile(f_path, best_fpath)

#load checkpoint
def load_ckp(checkpoint_fpath, model, optimizer):
    """[Can be used to continue the training or for inferencing hte results]

    Args:
        checkpoint_fpath ([type]): [Path to weight file]
        model ([type]): [description]
        optimizer ([type]): [description]

    Returns:
        [type]: [model, optimizer and epoch when file was stored]
    """
    checkpoint = torch.load(checkpoint_fpath)
    # state_dict = torch.load('./cust_conv_autoencoder.pth')
    # model.load_state_dict(state_dict)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def create_dir(dir_name):
    os.makedirs(dir_name) if not os.path.exists(dir_name) else print("Directory Exist...\n Continue Trainning..")

def preview_img(pic,filename:str, dir_path:str=None):
    """[summary]

    Args:
        pic ([type]): [listo fo tensor detached images]
        filename (str): [list of image index]
        dir_path (str): [path to save the images]
    """
    create_dir(dir_path)

    #revert normalize
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    for filename,i in zip (filename,pic):
        im_numpy = inv_normalize(i)
        im_ram = np.transpose(im_numpy,(1,2,0))
        im_ram = im_ram.numpy()  
        if dir_path is not None:      
            cv2.imwrite(os.path.join(dir_path,f"{filename}"),np.round(im_ram*255))
        
        #if you are saving with matplot requires to convert the color channel to rgb
        # im_ram= cv2.cvtColor(im_ram,cv2.COLOR_BGR2RGB) 
        # plt.imshow(im_ram) 
        # plt.gca().set_axis_off()
        # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        #             hspace = 0, wspace = 0)
        # plt.margins(0,0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.savefig(os.path.join(dir_path,f"mat{filename[0]}"), bbox_inches = 'tight',
        #     pad_inches = 0)        
        
        # plt.savefig(os.path.join(dir_path,f"{count}.png"),pad_inches = 0)



