import os
import numpy as np
import cv2
from tqdm import tqdm as tq
from os.path import splitext


class Ram_Patch_Utils:
    def __init__(self) -> None:
        pass

    def ram_patch_img(self,img_path:str="",total_number:int=16,output_dir:str="output_patched_image")-> None:
        """[Create multiple images in patch from a single images,
         patch is created from left to right with file name starting from 0 eg: 0.png, 1.png .....
         Works best with square sized image]

        Args:
            img_path (str): [Image path ]
            total_number (int): [Number of patch images to be created]
            output_dir (str): [Output directory to save patch images]
        """

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img =cv2.imread(img_path,)
        h,w,c =img.shape
        crop_height = h/total_number
        crop_width = w/total_number
        
        count = 0
        for i in range(total_number):
            for j in range(total_number):
                img1 = img[int(i*crop_width):int((i+1)*crop_width),int(j*crop_height):int((j+1)*crop_height)]
                cv2.imwrite(os.path.join(output_dir,f"{count}.png"),img1)
                count+=1
                

    def ram_patch_merge_img(self,input_dir:str="output_patched_image",output_dir:str="output_patch_merge_img",patch_height:int=1024,patch_width:int=1024):
        
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # print(input_dir)
        img_list = os.listdir(input_dir)
        # file_extension = img_list[0].split('.')[-1]
        # print(input_dir)
        # print(img_list)
        # print(len(img_list))
        img1 = cv2.imread(os.path.join(input_dir,img_list[0]))
        h,w,c =img1.shape
        vert_merge_no =int(patch_height/h)
        hori_merge_no =int(patch_width/w)
       
        count=0
        for i in range(vert_merge_no):    
            for j in range(hori_merge_no):
                img = cv2.imread(os.path.join(input_dir,f"{count}.png"))
                hor_concat = img if j == 0 else cv2.hconcat([hor_concat,img])
                count+=1
            vert_concat = hor_concat if i == 0 else cv2.vconcat([vert_concat,hor_concat])

        # vert_concat = cv2.cvtColor(vert_concat,cv2.COLOR_BGR2RGB)
        # cv2.imshow("preview",vert_concat)
        # cv2.imwrite(f"{output_dir}/example.{file_extension}",vert_concat)
        cv2.imwrite(output_dir,vert_concat)
        # cv2.waitKey(2000)
        

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name) 

#use case
patch_img = Ram_Patch_Utils()
# patch_img.ram_patch_img("/home/nabu/workspace/ram_utils/ram_utils/drop-original.png",16)
# input_img_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/patch_inference_img/color/000"
# input_img_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/patch_inference_img/"
# output_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/inference_img_patched"

input_img_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/patch_img_differences"
output_dir="/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/difference_img_patched"

input_img_dir = "/home/pasonatech/workplace_new/ram_utils/ram_utils/autoencoder_stuff/merge_infer_dropout_section"
output_dir = "/home/pasonatech/workplace_new/ram_utils/ram_utils/autoencoder_stuff/patch_infer_diff_back_256"

list_datatype_directory = sorted(os.listdir(input_img_dir))
for dir_type in list_datatype_directory:
    # print(dir_type) #datatype folder name
    input_dir_type = os.path.join(input_img_dir,dir_type)

    #create output folder
    datawise_output_dir = os.path.join(output_dir,dir_type)
    # print(datawise_output_dir)
    create_dir(datawise_output_dir)

    #list patch images folder
    list_dir_type_number = sorted(os.listdir(input_dir_type))

    for each_patch_dir in list_dir_type_number:
        #input patch folder
        each_dir = os.path.join(input_dir_type,each_patch_dir)
        #output file name
        output_filename = f"{each_patch_dir}.png"
        output_save_dir = os.path.join(datawise_output_dir,output_filename)

        # print(output_save_dir)
        # exit()
        # print(each_dir)
        patch_img.ram_patch_merge_img(input_dir=each_dir,output_dir=output_save_dir,patch_height=256,patch_width=256)
    # print(output_save_dir)