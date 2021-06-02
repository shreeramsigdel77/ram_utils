from __future__ import print_function
import cv2 
import numpy as np
import argparse
import os
src = None
erosion_size = 0
max_elem = 2
max_kernel_size = 21
title_trackbar_element_shape = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_erosion_window = 'Erosion Demo'
title_dilation_window = 'Dilation Demo'
def main(image,filename,good_img):
    img = cv2.imread(image)  
    img_good = cv2.imread(os.path.join(good_img,filename))
    kernel1 = np.ones((5,5), np.uint8)  
    kernel2 = np.ones((7,7), np.uint8) 
    kernel3 = np.ones((16,16), np.uint8) 
    img_erosion = cv2.erode(img, kernel1, iterations=1)  
    img_dilation = cv2.dilate(img_erosion, kernel2, iterations=1)  
    img_erosion = cv2.erode(img_dilation, kernel2, iterations=1)  
    img_dilation = cv2.dilate(img_erosion, kernel3, iterations=1) 

    #finding contours from image
    # find the contours from the thresholded image

    img_gray = cv2.cvtColor(img_dilation,cv2.COLOR_BGR2GRAY)
    #apply binary thresholding
    ret,thresh =cv2.threshold(img_gray,150,255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    img_contour = img_dilation.copy()

    #createempty mask
    mask = np.zeros(img_contour.shape[:2],dtype=img_contour.dtype)
    # create contours with area
    for c in contours:
        if cv2.contourArea(c)>400:
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(mask,[c],0,(255),2)
            cv2.drawContours(img_good,[c],0,(255),2)

    #applying the mask to the original image
    result = cv2.bitwise_and(img_good,img_good,mask=mask)

    cv2.imshow(f'{filename}', img_good) 
    cv2.waitKey(0)  
    cv2.destroyAllWindows()


if __name__ == "__main__":
    datatype = ["color","thread","hole","cut","metal_contamination","good_test"]
    picked_dir = datatype[1]
    good_img_dir = f"/home/nabu/workspace/pytorch_env/deepNN_py/data_set/test/{picked_dir}"
    input_dir = f"/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/path_calcualtion/difference_img_patched/{picked_dir}/"
    
    if input_dir is None:
        print("Could not open or find the image:",input_dir)
        exit(0)
    img_list = os.listdir(input_dir)
    for i in img_list:
        input_img_path = os.path.join(input_dir,i)
        main(input_img_path,i,good_img_dir)