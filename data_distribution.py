# Analytical data distribution
# code by Yinglong.du 02/28/2023


#
import cv2
import os
import tqdm
from data_tools import read_common, write2file
import matplotlib.pyplot as plt
import numpy as np


def Analytical_data_distribution(image_path, label_path, tags, label_format="bbox"):
    #   At first, we need to know is how distribution the resolution of the picture is.
    print("Processing Analytical_data_distribution.........")
    if image_path[-1] != '/':
        image_path += "/"
    images = os.listdir(image_path)
    h_w_ratio = dict()      # Image aspect ratio h/w
    Area_ratio = dict()     # Area ratio of each tag
    distribution = dict()   # image's distribution
    result_path = image_path + "../result/"
    data_state = list()
    data_state.append("all_images:"+str(len(images)))
    data_state.append("all_labels:" + "")
    data_state.append("images_with_out_label:" + "")
    data_state.append("all_bbox:" + "\n")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for i in tags:
        Area_ratio[i] = dict()
    for i in tqdm.tqdm(images):
        img = cv2.imread(image_path+i) if i[:-1] == 'g' else ""
        if img == "": continue
        h, w = img.shape[0], img.shape[1]
        ratio = round(h/w, 1)
        if h_w_ratio.get(str(ratio)):
            h_w_ratio[str(ratio)] += 1
        else:
            h_w_ratio[str(ratio)] = 1
        max_h_w = max(h, w)
        d_w = (max_h_w // 32) * 32
        d_ratio = str(d_w)+"*"+str(d_w)
        if distribution.get(str(d_ratio)):
            distribution[str(d_ratio)] += 1
        else:
            distribution[str(d_ratio)] = 1
        label = i[:-3] + ".txt"
        try:
            bboxs = read_common(label_path + label, label_format)
        except:
            bboxs = ""
        if bboxs:
            # caculate 面积比
            for bbox in bboxs:
                a_ratio = round(bbox[3] * bbox[4] / (h*w), 1)
                if Area_ratio[str(bbox[1])].get(str(a_ratio)):
                    Area_ratio[str(bbox[1])][(str(a_ratio))] += 1
                else:
                    Area_ratio[str(bbox[1])][(str(a_ratio))] = 1

        #write_h_w_ratio
        x_data = list(h_w_ratio.keys())
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        for i in x_data:
            plt.bar(x_data[i], h_w_ratio[i])
        plt.title("Image aspect ratio h/w")
        plt.xlabel("h/w ratio")
        plt.ylabel("quantity")
        plt.gcf().set_size_inches(32, 32)  # get current figure
        plt.savefig(result_path+"Image aspect ratio h/w.png", dpi=300)
        plt.close()

        # write image's distribution
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        x_data = list(distribution.keys())
        for i in x_data:
            plt.bar(x_data[i], distribution[i])
        plt.title("image's distribution")
        plt.xlabel("distribution")
        plt.ylabel("quantity")
        plt.gcf().set_size_inches(32, 32)  # get current figure
        plt.savefig(result_path+"image's distribution.png", dpi=300)
        plt.close()

        #  Area ratio of each tag
        Ads = Area_ratio.keys()
        for Ad in Ads:
            plt.rcParams["font.sans-serif"] = ["SimHei"]
            plt.rcParams["axes.unicode_minus"] = False
            x_data = list(Area_ratio[Ad].keys())
            for i in x_data:
                plt.bar(x_data[i], Area_ratio[Ad][i])
            plt.title("Area ratio of"+Ad)
            plt.xlabel("Area_ratio")
            plt.ylabel("quantity")
            plt.gcf().set_size_inches(32, 32)  # get current figure
            plt.savefig(result_path+"Area ratio of"+Ad+".png", dpi=300)
            plt.close()
    write2file(data_state, result_path + "Info.txt", add=True)
    print("Analytical_data_distribution is done, Have a nice day!")
