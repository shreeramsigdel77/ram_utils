from os.path import splitext
import cv2
import os
from tqdm import tqdm as tq

def crop_img(img_path:str,total_number:int,output_dir:str):
    img =cv2.imread(img_path,)
    img = cv2.resize(img, (256,256))
    h,w,c =img.shape
    crop_height = h/total_number
    crop_width = w/total_number
    
    count = 0
    for i in range(total_number):
        for j in range(total_number):
            
            img1 = img[int(i*crop_width):int((i+1)*crop_width),int(j*crop_height):int((j+1)*crop_height)]

            # img1 = img[int(j*crop_width):int(i*crop_height),int((j+1)*crop_width):int((i+1)*crop_height)]
            # print(img1)
            cv2.imwrite(os.path.join(output_dir,f"{count}.png"),img1)
            count+=1
            



#Images list
#train_good
# input_img_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/data_set/carpet/train/good"

dir_name = ["color","cut","good","thread","hole","metal_contamination"]
for i in dir_name:
    input_img_dir = f"/home/pasonatech/ram_backup/workspace/deepNN_py/data_set/test/{i}"

    output_img_dir = "/home/pasonatech/workplace_new/ram_utils/ram_utils/autoencoder_stuff/patch_images/"


    #create different folder for each image
    img_list = os.listdir(input_img_dir)
    img_list = sorted(img_list)

    output_img_dir = os.path.join(output_img_dir,input_img_dir.split('/')[-1])
    # print(img_list)
    for i  in tq(img_list):
        img_fullpath = os.path.join(input_img_dir,i)
        # print(img_fullpath)
        # generates patch images from one single image
        
        h,t = splitext(i)
        # print(output_img_dir)
        
        new_dir = os.path.join(output_img_dir,str(h))
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print("New directory created")
        crop_img(img_fullpath,4,os.path.join(output_img_dir,str(h)))  #4 256/4 = 64