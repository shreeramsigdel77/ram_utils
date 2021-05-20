import json
import os
import glob
import cv2


def best_weightfile(metrics_json_file: str,det_weight_dir: str,no_weight_file:int):
    #load metrics and find best ap from the list:
    a = 0
    file_data = []
    tmp_val = []
    tmp_wt_name = []
    for line in open(metrics_json_file, 'r'):
        file_data.append(json.loads(line))
    
    #load pth files from a folder
    wt_file_list = glob.glob(os.path.join(det_weight_dir, '*.pth'))
    wt_file_list.sort()
    for filename in wt_file_list:
        file_head, file_tail = os.path.split(filename)
        name_head,file_tail = file_tail.split('.')
        h, t = name_head.split('_')
        if t.isnumeric():
            tmp_wt_name.append(int(t))

    #ends change log
    for item in file_data:
        keylists = item.keys()
        if 'bbox/AP50' in keylists:
            if (item['iteration'] + 1) % (tmp_wt_name[0] + 1) == 0:
                tmp_val.append( [item['iteration'],item['bbox/AP50']])  
    
    tmp_val = sorted(tmp_val, key=lambda x: x[0],reverse= True) #sorting once to achieve the updated iteration when ap is same
    sort_ap_list = sorted(tmp_val, key=lambda x: x[1],reverse= True) #sorting with bbox/50  
    tmp_itr = sort_ap_list[0][0] # iteration with max ap from the list
    #removes models,
    csv_log_value = sort_ap_list
    list_items = len(sort_ap_list)
    if no_weight_file is not -1:
        min_no_wtfile = no_weight_file if len(sort_ap_list)>no_weight_file else len(sort_ap_list)
        for i in sort_ap_list[min_no_wtfile:]:  #removes weigth files except the highest 1(Increase the value of 1 to store the number of weight files)
            tmp_rm_itr = i[0]
            csv_log_value.remove(i)
            
            rm_model_name = "model_" + '{:0>7}'.format(str(tmp_rm_itr)) + ".pth"
            rm_file_pth = det_weight_dir+'/'+rm_model_name
            if os.path.exists(rm_file_pth):
                os.remove(rm_file_pth)
            else:
                print("The file does not exist") 
    bst_model = "model_" + '{:0>7}'.format(str(tmp_itr)) + ".pth"

    bst_model_pth = det_weight_dir + '/' + bst_model

    return bst_model_pth, csv_log_value


def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2):
    """Visualizes a single bounding box on the image"""
    BOX_COLOR = (255, 0, 0) # Red
    TEXT_COLOR = (255, 255, 255) # White
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def draw_bboxtovisualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(img)
    return img

