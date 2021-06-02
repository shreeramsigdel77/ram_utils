
#output diir for cropped img
output_dir_input = "/home/nabu/workspace/ram_utils/ram_utils/img_simalirity_check/parts_img_in"
output_dir_output = "/home/nabu/workspace/ram_utils/ram_utils/img_simalirity_check/parts_img_out"
output_rgb = "/home/nabu/workspace/ram_utils/ram_utils/img_simalirity_check/parts_img_rgb"
#color
input_img = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/inference_result_aug/anomaly_val_data/color001.png"
img_inference = "/home/nabu/workspace/pytorch_env/deepNN_py/data_set/anomoly_val_data/color001.png"


#holes
input_img = "//home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/inference_result_aug/anomaly_val_data/hole014.png"
img_inference = "/home/nabu/workspace/pytorch_env/deepNN_py/data_set/anomoly_val_data/hole014.png"

#metal
# input_img = "//home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/inference_result_aug/anomaly_val_data/metal000.png"
# img_inference = "/home/nabu/workspace/pytorch_env/deepNN_py/data_set/anomoly_val_data/metal000.png"

#thread
# input_img = "//home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/inference_result_aug/anomaly_val_data/th011.png"
# img_inference = "/home/nabu/workspace/pytorch_env/deepNN_py/data_set/anomoly_val_data/th011.png"


# #test data 
# input_img = "/home/nabu/workspace/pytorch_env/deepNN_py/data_set/test/good/000.png"
# img_inference = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/inference_result_aug/good_test_data/000.png"



# input_img = "/home/nabu/workspace/ram_utils/ram_utils/drop-original.png"
# img_inference = "/home/nabu/workspace/ram_utils/ram_utils/dropout.png"
dir_img_name = "000"
img_number = "166"
input_img = f"/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/patch_images/color/{dir_img_name}/{img_number}.png"
img_inference = f"/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/patch_inference_img/color/{dir_img_name}/{img_number}.png"


import cv2
import numpy as np
import os

img = cv2.imread(input_img)
print(img)
cv2.imshow("preview",img)
cv2.waitKey(0)
cv2.destroyWindow("preview")


def subtract_image(img1,img2,option:str="None"):
    
    os.path.isfile(img1)
    os.path.isfile(img2)
    img1 = cv2.imread(input_img,2)
    img2 = cv2.imread(img_inference,2)
    # diff = cv2.subtract(img1, img2)
    diff = cv2.absdiff(img2, img1)
    if option == 'grayscale':
    #   invert color
        diff = cv2.bitwise_not(diff) # OR
    print(diff)

    #threshhold
    # diff[diff>100]=0
    diff[diff>20]=255
   
    result = not np.any(diff) #returns false when diff is all zero and with not it will inverse to true
    
    cv2.imwrite("difference1.png",diff)
    cv2.imshow("Preview_result", diff)
    cv2.waitKey(0)
    # cv2.imshow("Preview1", diff)
    # cv2.waitKey(0)
    # return diff
    print("Done")

subtract_image(input_img,img_inference,)
exit()

def crop_img(img_path:str,total_number:int,output_dir:str):
    img =cv2.imread(img_path,)
    # h,w,c = img.shape
    #split image
    # b,g,r =cv2.split(img)
    h,w,c =img.shape
    crop_height = h/total_number
    crop_width = w/total_number
    img_crop_list =[]
    bgr_color_list =[]
    count = 0
    for i in range(total_number):
        for j in range(total_number):
            # crop_box=(j*crop_width,i*crop_height,(j+1)*crop_width,(i+1)*crop_height)
            print(int(i*crop_height))
            print(j*crop_width)
            print((i+1)*crop_height)
            print((j+1)*crop_width)
            # img1 = img[int(j*crop_width):int((j+1)*crop_width),int(i*crop_height):int((i+1)*crop_height),:]

            img1 = img[int(i*crop_width):int((i+1)*crop_width),int(j*crop_height):int((j+1)*crop_height)]
            # img1 = img[int(j*crop_width):int((j+1)*crop_width),int(i*crop_height):int((i+1)*crop_height)]
            # img1 =img.crop(crop_box)
            # print(img1)
            bgr_color_list.append(sum(sum(img1)))
            print(sum(sum(img1)))
            print(img1.shape)
            cv2.imwrite(os.path.join(output_dir,f"{count}.png"),img1)
            count+=1
            cv2.imshow("preview",img1)
            # cv2.waitKey(0)
            img_crop_list.append(img1)
    print("Count",count)
    # print(bgr_color_list)
    return bgr_color_list

def crop_img_channel(img_path:str,total_number:int,output_dir:str):
    img =cv2.imread(img_path,)
    # h,w,c = img.shape
    #split image
    b,g,r =cv2.split(img)
    h,w =b.shape
    test_img = img.copy()
    test_img_score = img.copy()
    cv2.imshow("preview_blue_channel",b)
    cv2.waitKey(5000)
    crop_height = h/total_number
    crop_width = w/total_number
    img_crop_list =[]
    bgr_color_list =[]
    count = 0
    temp_input_img = "/home/nabu/workspace/pytorch_env/deepNN_py/carpet_script/bw_convert_train/inference_result_aug/anomaly_val_data/hole014.png"
    temp_img = cv2.imread(temp_input_img)
    temp_img_copy = temp_img.copy()
    b1,g1,r1 = cv2.split(temp_img)
    for i in range(total_number):
        for j in range(total_number):
            # crop_box=(j*crop_width,i*crop_height,(j+1)*crop_width,(i+1)*crop_height)
            # print(int(i*crop_height))
            # print(j*crop_width)
            # print((i+1)*crop_height)
            # print((j+1)*crop_width)
            # img1 = img[int(j*crop_width):int((j+1)*crop_width),int(i*crop_height):int((i+1)*crop_height),:]
            # print(j,i)
            # exit()
            #draw rect
            # test_img = cv2.rectangle(test_img,(int(j*crop_width),int((j+1)*crop_width)),(int(i*crop_height),int((i+1)*crop_height)),(0,255,0),3)

            
            test_img = cv2.rectangle(test_img,(int(j*crop_width),int(i*crop_height),int((j+1)*crop_width),int((i+1)*crop_height)),(0,255,0),1)

            test_img_score = cv2.rectangle(test_img_score,(int(j*crop_width),int(i*crop_height),int((j+1)*crop_width),int((i+1)*crop_height)),(0,255,0),1)
            temp_img_copy = cv2.rectangle(temp_img_copy,(int(j*crop_width),int(i*crop_height),int((j+1)*crop_width),int((i+1)*crop_height)),(0,255,0),1)


            # test_img = cv2.rectangle(test_img, (x, y), (x + 64, y + 64), (36,255,12), 1)
            cv2.putText(test_img, f"{count}", (int(j*crop_width)+5, int(i*crop_height)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)




            img1 = r[int(i*crop_width):int((i+1)*crop_width),int(j*crop_height):int((j+1)*crop_height)]

            # img1 = b[int(j*crop_width):int((j+1)*crop_width),int(i*crop_height):int((i+1)*crop_height)]
            # img1 =img.crop(crop_box)
            print(img1)
            bgr_color_list.append(sum(sum(img1)))
            print(sum(sum(img1)))
            if sum(sum(img1))>3500:
                cv2.putText(test_img_score, f"{sum(sum(img1))}", (int(j*crop_width)+5, int(i*crop_height)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)
                cv2.putText(temp_img_copy, f"{sum(sum(img1))}", (int(j*crop_width)+5, int(i*crop_height)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)
                

            print(img1.shape)
            img1 = cv2.putText(img1, f'{count}', (0,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(output_dir,f"{count}.png"),img1)
            count+=1
            cv2.imshow("preview",img1)
            # cv2.waitKey(0)
            img_crop_list.append(img1)
    cv2.imwrite(f"bbox-img.png",test_img)
    cv2.imwrite(f"bbox-img_score.png",test_img_score)
    cv2.imwrite(f"temp-bbox-img_score.png",temp_img_copy)
    print("Count",count)
    # print(bgr_color_list)
    return bgr_color_list

# inp1 =crop_img(img_path=input_img, total_number =16,output_dir= output_dir_input)     
# out1 =crop_img(img_path=img_inference, total_number =16,output_dir= output_dir_output)


inp1 =crop_img_channel(img_path=input_img, total_number =16,output_dir= output_dir_input)     
out1 =crop_img_channel(img_path=img_inference, total_number =16,output_dir= output_dir_output)


#difference image
diff_img = "/home/nabu/workspace/ram_utils/ram_utils/img_simalirity_check/difference1.png"
diff_out = "/home/nabu/workspace/ram_utils/ram_utils/img_simalirity_check/parts_img_diff"
out_diff =crop_img(img_path=input_img, total_number =16,output_dir= output_rgb)

print(out_diff)

print(out_diff[163])

for i in range(len(out_diff)):
    if out_diff[i]<2000:
        out_diff[i] = 0
    
print(out_diff)

# print(np.subtract(input_sum,output_sum))

# from ram_utils.matplot_handler.hist_create import plot_hist

from ram_utils.matplot_handler.hist_create import plot_hist
plot_hist(out_diff,[],"","",100, "Color Space 64X64 sum","Color Space 64X64 sum","Frequency",True,True)
# plot_hist(inp1,out1,"Input","Output",100, "Grayscal 64X64 sum display","Sum of 64X64 grayscale value","Frequency",True,True)