import cv2

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import histogram, label


def histo_bins(bins,binwidth):
    return np.arange(min(bins),max(bins)+binwidth,binwidth)
   

def plot_hist(data1:list=[],data2:list=[],label1:str=None,label2:str=None,binwidth:float=1.0,title:str=None,xaxis_label:str="X-axis",yaxis_label:str="Frequecy",preview:bool=True,save_histo:bool=False):
    data1 = list(np.around(np.array(data1),7))
    data2 = list(np.around(np.array(data2),7))
    #data1
    counts1, bins1 = np.histogram(data1)
    plt.hist(bins1[:-1], bins=histo_bins(bins1[:-1],binwidth), weights=counts1, label=label1,alpha=0.5)
    #data2
    counts, bins = np.histogram(data2)
    plt.hist(bins[:-1], bins=histo_bins(bins[:-1],binwidth), weights=counts, label= label2, alpha=0.5)
    # plt.axvline(x=0.838,color="r", label="Threshhold")
    # plt.axvline(x=0.925,color="r", label="Threshhold")
    plt.title(f"{title}")
    plt.legend()
    plt.xlabel(f'{xaxis_label}')
    plt.ylabel(f'{yaxis_label}')
    if save_histo:
        plt.savefig("histogram.png")
    if preview:
        plt.show()



def rgb_hist(img,win):
    color = ('b','g','r')
    plt.figure(win)
    for channel,col in enumerate(color):
        histr = cv2.calcHist(images=[img],channels=[channel],mask=None,histSize=[256],ranges=[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
  
    plt.title('Histogram for color scale picture')
    plt.show()

def rgb_hist_two_image(noanomaly,anomaly,win="title", label1="label1",label2 = "label2"):
    color = ('b','g','r')
    plt.figure(win)
    # for channel,col in enumerate(color):
    #     histr = cv2.calcHist(images=[noanomaly],channels=[channel],mask=None,histSize=[256],ranges=[0,256])
    #     plt.plot(histr,color = 'g', label="Noanomaly")
    #     histr1 = cv2.calcHist(images=[anomaly],channels=[channel],mask=None,histSize=[256],ranges=[0,256])
    #     plt.plot(histr1,color = 'b', label="Anomaly")
    #     plt.legend()
    #     plt.title(f'{col}')
    #     plt.xlim([0,256])
    #     plt.show()

    channel = 2
    col = "r"
    histr = cv2.calcHist(images=[noanomaly],channels=[channel],mask=None,histSize=[256],ranges=[0,256])
    plt.plot(histr,color = 'g', label=label1)
    histr1 = cv2.calcHist(images=[anomaly],channels=[channel],mask=None,histSize=[256],ranges=[0,256])
    plt.plot(histr1,color = 'b', label=label2)
    plt.legend()
    plt.title(f'{col}')
    plt.xlim([0,256])
    plt.show()


    # plt.legend()
    # plt.title('Histogram for color scale picture')
    # plt.savefig(f"{win}.png")
    # plt.show()

def gray_hist(img,win):
    plt.figure(win)

    histr = cv2.calcHist(images=[img],channels=[0],mask=None,histSize=[256],ranges=[0,256])
    plt.plot(histr,color = "#D3D3D3")
    plt.xlim([0,256])
  
    plt.title('Histogram for color scale picture')
    plt.savefig(f"{win}.png")
    plt.show()

def gray_hist_two_image(noanomaly,anomaly,win):
    plt.figure(win)
    
    histr = cv2.calcHist(images=[noanomaly],channels=[0],mask=None,histSize=[256],ranges=[0,256],)
    plt.plot(histr,color = "b",label="Not an anomaly")

    histr1 = cv2.calcHist(images=[anomaly],channels=[0],mask=None,histSize=[256],ranges=[0,256], )
    plt.plot(histr1,color = "r",label="Anomaly")


    plt.xlim([0,256])
    plt.legend()
    plt.title('Histogram for grayscale picture')
    plt.savefig(f"{win}.png")
    plt.show()


