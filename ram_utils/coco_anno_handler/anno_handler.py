import json
import os



def images_path_from_json(test_json:str,test_img_dir:str):
    """[Reads contents from a json file to generate a list of test image path and the respective id]

    Args:
        test_json (str): [test json file]
        test_img_dir (str): [test image directory path]

    Returns:
        [type]: [list of images path and id]
    """
    with open(test_json, "r") as json_file:
        data = json.load(json_file)
    test_img_path_list = []
    for items in data['images']:
        test_img_path_list.append([(os.path.join(test_img_dir,items['file_name'])),items['id']])
    json_file.close()
    return test_img_path_list


def annotations_updater(file_path:str)->None:
    """[Start annotation id from 1 so, that it follows the coco api Ap calculation format. It loads the file and changes the annotations from 0 ]

    Args:
        file_path (str): [Path to coco json file. ]
    """
    with open(file_path, "r") as jsonFile:
        data = json.load(jsonFile)
    
    if data["annotations"][0]["id"] is 0:
        for items in data["annotations"]:
            items['id'] =items['id']+1
        
        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile)