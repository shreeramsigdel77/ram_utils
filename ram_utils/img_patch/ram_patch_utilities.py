import os
import numpy as np
import cv2
from tqdm import tqdm as tq
from os.path import splitext


class Ram_Patch_Utils:
    def __init__(self) -> None:
        pass

    def ram_patch_img(img_path:str="",total_number:int=4,output_dir:str="./output_patched_image")-> None:
        """[Creates multiple images in patch from a single images,
         patch is created from left to right with file name starting from 0 eg: 0.png, 1.png .....
         Works best with square sized image]

        Args:
            img_path (str): [Image path ]
            total_number (int): [Number of patch images to be created]
            output_dir (str): [Output directory to save patch images]
        """
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
                

    def ram_patch_merge_img():
        pass