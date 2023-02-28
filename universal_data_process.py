# Universal data processing program
# develop by YingLong Du 02/27/2023


# First bbox2yolo

import os
import tqdm
import cv2
import datetime
from data_tools import divide_data, read_coco, bbox_to_yolo, write2file


# date
def universal_bbox2yolo(data_path, label_dict=None, train_test_val_ratio=[]):
    if data_path[-1] != '/':
        data_path += "/"
    image_path = data_path + "images/"
    txts_path = data_path + "txts/"

    img_list = os.listdir(image_path)  # get image_list
    total = len(img_list)
    print("Number of all pictures:", total)
    source = data_path + "labels/"
    if not os.path.exists(source):
        os.mkdir(source)

    if label_dict:
        num_bbox = 0
        num_image = 0
        negative_sample = 0
        classes = label_dict
        print("The label format is being converted.....")
        for image in tqdm.tqdm(img_list):
            label = image.replace("jpg", 'txt')  # gets the label corresponding to the current image
            try:
                bboxs = read_coco(txts_path + label)
            except:
                # negative_sample
                f = open(source + label, 'w')
                num_image += 1
                negative_sample += 1
                f.write("\n")
                f.close()
                continue
            img = image_path + image
            img = cv2.imread(img)
            height, width = img.shape[0], img.shape[1]

            content = []
            for bbox in bboxs:
                tittle = bbox[0]
                bbox = bbox[1:]
                try:
                    bbox_y = bbox_to_yolo(bbox, w=width, h=height)
                    str_bbox_y = str(classes[tittle]) + " " + str(bbox_y[0]) + " " + str(bbox_y[1]) + " " + str(
                        bbox_y[2]) + " " + str(bbox_y[3]) + "\n"
                    content.append(str_bbox_y)
                except:
                    print(label, tittle)

            f = open(source + label, 'w')
            num_image += 1
            num_bbox += len(content)
            for i in content:
                f.write(i)
            f.close()
        current_time = datetime.datetime.now()
        write_content = ["convert Time: " + str(current_time),
                         "label_dict: " + str(label_dict),
                         "number of images: " + str(num_image),
                         "number of bbox: " + str(num_bbox),
                         "number of negative sample: " + str(negative_sample) + "\n"]
        write2file(write_content, data_path + "Info.txt", add=False)

    if train_test_val_ratio:
        print("Dividing the data set.....")
        total, train, test, val = divide_data(data_path, data_path, train_test_val_ratio)
        current_time = datetime.datetime.now()
        write_content = ["Divided Time: " + str(current_time),
                         "Dataset_Total: " + str(total),
                         "number of train: " + str(train),
                         "number of test: " + str(test),
                         "number of val: " + str(val) + "\n"]
        write2file(write_content, data_path + "Info.txt", add=True)
    print("Universal data process is done, Have a nice day!")
