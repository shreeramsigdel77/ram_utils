import cv2
import numpy as np
import random
from PIL import Image
from ram_utils.checker.check_utils import *
class ImageHandler:
    def load_img(image_path: str)->np.ndarray:
        """[Opencv read function]

        Args:
            image_path (str): [description]

        Returns:
            np.ndarray: [description]
        """
        is_file(image_path)
        try:
            return cv2.imread(image_path)
        except IOError as io:
            print(io)

    def show_img(image:np.ndarray, win_name:str = "Preview", wait_val:int = 500 )-> None:
        """[image preview]

        Args:
            image (np.ndarray): [description]
            win_name (str, optional): [description]. Defaults to "Preview".
            wait_val (int, optional): [description]. Defaults to 500.
        """
        try:
            cv2.imshow(win_name, image)
            cv2.waitKey(wait_val)
            cv2.destroyAllWindows()
        except IOError as io:
            print(io)

    def save_img(image:np.ndarray, filename:str="0.png")->None:
        """[saves images]

        Args:
            image ([]): [Image to be saved]
            filename ([string]): [name of a image]
        """
        try:
            cv2.imwrite(filename, image)
        except IOError as io:
            print(io)

    def convert_PIL_to_numpy(image, format="BGR"):
        """
        Convert PIL image to numpy array of target format.
        Args:
            image (PIL.Image): a PIL image
            format (str): the format of output image
        Returns:
            (np.ndarray): also see `read_image`
        """
        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format in ["BGR", "YUV-BT.601"]:
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)        # handle formats not supported by PIL
        elif format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        elif format == "YUV-BT.601":
            image = image / 255.0
            image = np.dot(image, np.array(_M_RGB2YUV).T)

        return image

    

def get_random_url(folder):
    choose_random = random.choice(os.listdir(folder))
    return Image.open(os.path.join(folder, choose_random))

def add_background(image:np.ndarray= None,background_img_path:str = None)->np.ndarray:
    background = get_random_url(folder =background_img_path)        # resize the image
    size = image.size[:2]
    background = background.resize(size,Image.ANTIALIAS)
    background.paste(image, (0, 0), image)
    return background

def filter_images(dir_path:str)->list:
    """[creates a list of image path from a given directory]

    Args:
        dir_path (str): [Path to a folder which contains images]

    Returns:
        list: [every image path as a list]
    """
    
    file_extensions = ['png', 'jpg', 'gif', 'jpeg']    # Add image formats here

    img_path_list = []
    for files in os.listdir(dir_path):
        if any(extensions in files for extensions in file_extensions):
            img_path_list.append(os.path.join(dir_path,files))

    return img_path_list

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
    