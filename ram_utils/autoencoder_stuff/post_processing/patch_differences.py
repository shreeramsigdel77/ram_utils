import cv2
import os
import numpy as np


def subtract_image(img1,img2,output,option:str="None"):
    
    os.path.isfile(img1)
    os.path.isfile(img2)
    img1 = cv2.imread(img1,2)
    img2 = cv2.imread(img2,2)
    # diff = cv2.subtract(img1, img2)
    diff = cv2.absdiff(img2, img1)
    if option == 'grayscale':
    #   invert color
        diff = cv2.bitwise_not(diff) # OR
    # print(diff)

    #threshhold
    # diff[diff>100]=0
    diff[diff>20]=255
   
    result = not np.any(diff) #returns false when diff is all zero and with not it will inverse to true
    print(output)
    cv2.imwrite(output,diff)
    # cv2.imshow("Preview_result", diff)
    # cv2.waitKey(0)
    # cv2.imshow("Preview1", diff)
    # cv2.waitKey(0)
    # return diff
    # print("Done")
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name) 


input_data = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/patch_images"
input_inference_data = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/patch_inference_img/"
output_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/patch_img_differences"

input_data = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/aug_inference/aug_img/aug_input_data"
input_inference_data = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/aug_inference/aug_img/aug_inference_data"
output_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/aug_inference/aug_img"


list_datatype_directory = sorted(os.listdir(input_data))
for dir_type in list_datatype_directory:
    # print(dir_type) #datatype folder name
    input_dir_type = os.path.join(input_data,dir_type)
    input_dir_type2 = os.path.join(input_inference_data,dir_type)

    #create output folder
    datawise_output_dir = os.path.join(output_dir,dir_type)
    # print(datawise_output_dir)
    create_dir(datawise_output_dir)

    #list patch images folder
    list_dir_type_number = sorted(os.listdir(input_dir_type))
    list_dir_type_number2 = sorted(os.listdir(input_dir_type2))

    for each_patch_dir in list_dir_type_number:
        #input patch folder
        # print(each_patch_dir)
        
        each_dir = os.path.join(input_dir_type,each_patch_dir)
        each_dir2 = os.path.join(input_dir_type2,each_patch_dir)
        print(each_dir2)
        #output file name
        output_dir_name = os.path.join(datawise_output_dir,each_patch_dir)
        # print(output_dir_name)
        # output_save_dir = os.path.join(datawise_output_dir,)
        create_dir(output_dir_name)
        for i in os.listdir(each_dir):
            # print(os.path.join(each_dir2,i))
            # print(os.path.join(each_dir,i))
            # print(os.path.join(output_dir_name,i))
            subtract_image(os.path.join(each_dir2,i),os.path.join(each_dir,i),os.path.join(output_dir_name,i),)
