from PIL import Image
import matplotlib.pyplot as plt
import os


def generate_bargraph(csv_path:str, fig_save_path:str=None, img_name:str="result_bar_diagram.png"):
    """[function to generate bar diagram for a coco api]

    Args:
        csv_path (str): [path to a csv file]
        fig_save_path (str, optional): [save path]. Defaults to None so saves to.
    """
    df = pd.read_csv(csv_path)
    df.iloc[:,:-2].plot(kind="bar")

    axes = plt.gca()
    axes.yaxis.grid()
    plt.xticks([*range(0,df.shape[0])],df.iloc[:,-1],rotation="horizontal")
    plt.legend(bbox_to_anchor=(1,1), fontsize=10)
    plt.xlabel("AP and AR", labelpad=20, fontsize= 20)
    plt.ylabel("Score", labelpad=20, fontsize= 20)

    plt.gcf().set_size_inches(18.5,10.5)

    plt.subplots_adjust(right=0.75)
    plt.gcf().savefig(os.path.join(fig_save_path,img_name),dpi=100)

    # plt.show()

def generate_filtered_bargraph(csv_path:str, fig_save_path:str=None, img_name:str="filtered_bar_diagram.png"):
    """[function to generate bar diagram for a coco api]

    Args:
        csv_path (str): [path to a csv file]
        fig_save_path (str, optional): [save path]. Defaults to None so saves to.
    """
    df = pd.read_csv(csv_path)
    df.iloc[:,:3].plot(kind="bar")

    axes = plt.gca()
    axes.yaxis.grid()
    plt.xticks([*range(0,df.shape[0])],df.iloc[:,-1],rotation="horizontal")
    plt.legend(bbox_to_anchor=(1,1), fontsize=10)
    plt.xlabel("AP", labelpad=20, fontsize= 20)
    plt.ylabel("Score", labelpad=20, fontsize= 20)

    plt.gcf().set_size_inches(18.5,10.5)

    plt.subplots_adjust(right=0.75)
    plt.gcf().savefig(os.path.join(fig_save_path,img_name),dpi=100)

    # plt.show()