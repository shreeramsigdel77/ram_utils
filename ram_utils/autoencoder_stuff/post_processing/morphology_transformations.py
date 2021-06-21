
import cv2 
import numpy as np
import argparse
import os
from tqdm import tqdm 
src = None
erosion_size = 0
max_elem = 2
max_kernel_size = 21
title_trackbar_element_shape = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_erosion_window = 'Erosion Demo'
title_dilation_window = 'Dilation Demo'

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name) 

def main(image,filename,good_img,output):
    img = cv2.imread(image)  
    img_good = cv2.imread(os.path.join(good_img,filename))
    img_good = cv2.resize(img_good,(256,256))
    # print("img", img.shape)
    # print("img", img_good.shape)
    # exit()
    #
    # 
    # 
    # kernel1 = np.ones((3,3), np.uint8)
    # #

    kernel1 = np.ones((5,5), np.uint8)  
    kernel2 = np.ones((7,7), np.uint8) 
    kernel2e = np.ones((6,6), np.uint8) 
    kernel3 = np.ones((8,8), np.uint8) 
    # img_erosion = cv2.erode(img, kernel1, iterations=1)  
    # img_dilation = cv2.dilate(img_erosion, kernel2, iterations=1)  
    # img_erosion = cv2.erode(img_dilation, kernel2e, iterations=1)  
    # img_dilation = cv2.dilate(img_erosion, kernel3, iterations=1) 

    img_erosion = cv2.erode(img, kernel1, iterations=1)  
    img_dilation = cv2.dilate(img_erosion, kernel2, iterations=1)
    img_dilation = cv2.dilate(img_dilation, kernel3, iterations=1)   
    img_erosion = cv2.erode(img_dilation, kernel2e, iterations=1)  
    
    # img_dilation = cv2.dilate(img, kernel2, iterations=1)
    # img_erosion = cv2.erode(img_dilation, kernel1, iterations=1)  
    # img_erosion = cv2.erode(img_erosion, kernel2e, iterations=1) 
    # img_dilation = cv2.dilate(img_erosion, kernel3, iterations=1)   
    



    #finding contours from image
    # find the contours from the thresholded image
    img_gray = cv2.cvtColor(img_erosion,cv2.COLOR_BGR2GRAY)

    # img_gray = cv2.cvtColor(img_dilation,cv2.COLOR_BGR2GRAY)
    #apply binary thresholding
    
    ret,thresh =cv2.threshold(img_gray,15,255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    img_contour = img_dilation.copy()
    img_contour_all = img_dilation.copy()

    #createempty mask
    mask = np.zeros(img_contour.shape[:2],dtype=img_contour.dtype)
    # create contours with area
    # img_contour[img_contour>25] = 255
    for c in contours:
        if cv2.contourArea(c)>100:
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(mask,[c],0,(255),2)
            cv2.drawContours(img_good,[c],0,(255),2)
            cv2.drawContours(img_contour,[c],0,(255),2)

        if cv2.contourArea(c)>0:
            x,y,w,h = cv2.boundingRect(c)
            # cv2.drawContours(mask,[c],0,(255),2)
            # cv2.drawContours(img_good,[c],0,(255),2)
            cv2.drawContours(img_contour_all,[c],0,(255),2)

    #applying the mask to the original image
    # result = cv2.bitwise_and(img_good,img_good,mask=mask)
    # cv2.imwrite(os.path.join(output,filename),img_good)

    #####test
    # img_dilation = cv2.dilate(img, kernel1, iterations=1)

    # cv2.imwrite("original.png",img)
    # cv2.imwrite("erode.png",img_erosion)
    # cv2.imwrite("dilation.png",img_dilation)
    # cv2.imwrite("contour.png",img_contour)
    # cv2.imwrite("contour_all.png",img_contour_all)
    # cv2.imwrite("img_goodcontour.png",img_good)
    
    # cv2.imshow(f'{filename}img_good', img_good)
    # cv2.imshow(f'{filename}contours_all', img_contour_all) #shows all contours
    # cv2.imshow(f'{filename}contours', img_contour)
    # cv2.imshow(f'{filename}dilate', img_dilation)
    # cv2.imshow(f'{filename}erode', img_erosion) 
    print("output",output)

    cv2.imwrite(os.path.join(output,filename), img_good)

    # cv2.imwrite(os.path.join(f"{output}contours",filename), img_contour)
    # cv2.imshow(f'{filename}', img)
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()
    # exit()


if __name__ == "__main__":
    datatype = ["color","thread","hole","cut","metal_contamination","good_test"]
    # picked_dir = datatype[1]

    # datatype = ["good"]
    output = "/home/pasonatech/workplace_new/ram_utils/ram_utils/autoencoder_stuff/morphological_inference_result"
    for picked_dir in tqdm(datatype):
        if picked_dir == "good":
            good_img_dir = "/home/nabu/workspace/pytorch_env/deepNN_py/data_set/carpet/train/good"
        else:
            good_img_dir = f"/home/pasonatech/ram_backup/workspace/deepNN_py/data_set/carpet/test/{picked_dir}"
        input_dir = f"/home/pasonatech/workplace_new/ram_utils/ram_utils/autoencoder_stuff/patch_infer_diff_back_256/{picked_dir}/"
        
        output_dir = os.path.join(output,picked_dir)
        create_dir(output_dir)
        if input_dir is None:
            print("Could not open or find the image:",input_dir)
            exit(0)
        img_list = os.listdir(input_dir)
        for i in tqdm(img_list):
           
            input_img_path = os.path.join(input_dir,i)
            print(input_img_path)
            print(good_img_dir)
            main(input_img_path,i,good_img_dir,output_dir)