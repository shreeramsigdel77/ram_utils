import os
from albumentations.pytorch.transforms import img_to_tensor
import cv2
import torch
import numpy as np
from torchvision import transforms as T
from datetime import datetime



img_preprocess = T.Compose([
    # T.RandomResizedCrop()
    T.RandomCrop((64,64)),
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



#save checkpoint
def save_ckp(state, checkpoint_dir, epoch):
    f_path = os.path.join(checkpoint_dir , f'checkpoint_{epoch}.pth')
    torch.save(state, f_path)

#load checkpoint
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']




def tensor_to_pil(tensor_image:torch.Tensor):
    """[Converts torch.Tensor to numpy.ndarray]

    Args:
        tensor_image (torch.Tensor): [Torch Tensor image]

    Returns:
        [numpy.ndarray]: [returns in RGB format]
    """
    rev_tens_img = reverse_preprocess(tensor_image)
    revert_tensor_image = cv2.cvtColor(rev_tens_img, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Reverse Tensor",revert_tensor_image)
    # cv2.waitKey(1000)
    # cv2.destroyWindow("Reverse Tensor")
    return rev_tens_img

def pil_img_to_tensor(img:np.ndarray):
    """[Convert np.ndarray Image to torch.Tensor]

    Args:
        img (np.ndarray): [Image ]

    Returns:
        [torch.Tensor]: [returns processed image]
    """
    img_tensor = T.Compose([
    T.ToTensor(), 
    ])
    processed_img = img_to_tensor(img)
    return processed_img


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name) 

def preview_img(pic:torch.tensor,filenames:list, dir_path:str, win_name:str="Preview"):
    create_dir(dir_path)
    for count,(i,filename) in enumerate (zip(pic,filenames)):
        #day month year hour month sec
        pil_img_rgb=T.ToPILImage()(i.squeeze_(0))
        pil_img_rgb.save(os.path.join(dir_path,filename))
        # im_rgb = tensor_to_pil(i)
        # im_bgr = cv2.cvtColor(im_rgb,cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(dir_path,filename),im_bgr)
        # cv2.imshow(win_name, im_bgr)
        # cv2.waitKey(5000)
        # cv2.destroyAllWindows()

       



def image_np_array(image_dir):
    """[Converts list of image from a directory to numpy array]

    Args:
        image_dir ([string]): [image directory path]

    Returns:
        [numpy.ndarray]: [list of images from a folder]
    """
    img_list = []
    for i in (os.listdir(image_dir)):
        img = cv2.imread(os.path.join(image_dir,i))
        img_list.append(img)
    #convert list to numpy array
    img_list = np.asarray(img_list)
    return img_list
#  print(i.shape)
