import cv2
from ram_utils.checker.check_utils import *
class ImageHandler:
    def load_img(self, image_path: str):
        is_file(image_path)
        try:
            return cv2.imread(image_path)
        except IOError as io:
            print(io)

    def show_img(self, image)-> None:
        try:
            cv2.imshow("Preview", image)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
        except IOError as io:
            print(io)

    def save_img(self, image, filename:str)->None:
        """[saves images]

        Args:
            image ([]): [Image to be saved]
            filename ([string]): [name of a image]
        """
        try:
            cv2.imwrite(filename, image)
        except IOError as io:
            print(io)