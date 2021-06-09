import os
from os.path import splitext
import cv2
import natsort
import numpy as np


def patch_count_writer(img_path:str,total_number:int,save_filename:str=None):
    img =cv2.imread(img_path,)
    file_name = img_path.split('/')[-1]
    print(file_name)
    h,w,c=img.shape
    test_img = img.copy()
    crop_height = h/total_number
    crop_width = w/total_number
    count = 0
    # print(score[0])
    # exit()
    
    for i in range(total_number):
        for j in range(total_number):   
            color = (36,255,12)  
            test_img = cv2.rectangle(test_img,(int(j*crop_width),int(i*crop_height),int((j+1)*crop_width),int((i+1)*crop_height)),(36,255,12),1)
            cv2.putText(test_img, f"{count}", (int(j*crop_width)+35, int(i*crop_height)+55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            count+=1
    cv2.imwrite(save_filename,test_img)
    


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name) 


input_folder = "/home/nabu/workspace/pytorch_env/deepNN_py/data_set/test"
output_folder = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/img_patch_position"

for data_type in os.listdir(input_folder):

    data_type_dir = os.path.join(input_folder,data_type)
    output_data_type_dir = os.path.join(output_folder,data_type)
    create_dir(output_data_type_dir)
    #list file names inside dir
    for image_name in os.listdir(data_type_dir):
        # print(image_name)
        file_path = os.path.join(data_type_dir,image_name)
        out_filename = os.path.join(output_data_type_dir,image_name)
        patch_count_writer(img_path=file_path,total_number=16,save_filename=out_filename)