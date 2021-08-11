import os
import cv2
import argparse

def videoToImages(video_path:str, output_dir:str):
    
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        success, image = cap.read()
        count = 0
        while (cap.isOpened()):
            success, image = cap.read()
            if not success:
                print("Extraction Completed")
                break
            #save frames as png
            cv2.imwrite(os.path.join(output_dir,f"frame_{count}.png"),image)
            # exit on escape key
            if cv2.waitKey(10)==27:
                break
            count += 1
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Convert Video to Images")
    parser.add_argument("--videoPathIn", help="Path to video",required=True,type=str)
    parser.add_argument("--imagesPathOut", help="Path to save images", default="./output_images", type= str)
    option = parser.parse_args()

    print(option.imagesPathOut)
    # print(option)
    if not os.path.exists(option.imagesPathOut):
        os.mkdir(option.imagesPathOut)
    
    videoToImages(option.videoPathIn,option.imagesPathOut)