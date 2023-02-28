import cv2
import os
import json
import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
def read_common(file_name, format = "bbox"):
    data = []
    try:
        file = open(file_name,'r',encoding='utf-8')
        file_date = file.readlines()
    except:
        file = open(file_name,'r',encoding='gbk')
        file_date = file.readlines()

    for row in file_date:
        try:
            tmp_list = row.split(' ')
            tmp_list[-1] = tmp_list[-1].replace('\n', '')
            if format == "bbox":
                tmp_list = [tmp_list[0], int(tmp_list[1]), int(tmp_list[2]), int(tmp_list[3]), int(tmp_list[4])]
            data.append(tmp_list)
        except:
            print(tmp_list)
    return data

def read_coco(file_name, format = "bbox"):
    data = []
    try:
        file = open(file_name,'r',encoding='utf-8')
        file_date = file.readlines()
    except:
        file = open(file_name,'r',encoding='gbk')
        file_date = file.readlines()

    for row in file_date:
        try:
            tmp_list = row.split(' ')
            tmp_list[-1] = tmp_list[-1].replace('\n', '')
            if format == "bbox":
                tmp_list = [tmp_list[0], int(tmp_list[1]), int(tmp_list[2]), int(tmp_list[3]), int(tmp_list[4])]
            data.append(tmp_list)
        except:
            print(tmp_list)
    return data

def load_txt(filename,encoding = 'utf-8'):
    data = []
    try:
        f = open(filename,'r',encoding=encoding)
        for i in f.readlines():
            if i != '\n':
                data.append(i)
    except:
        f = open(filename,'r',encoding='gbk')
        for i in f.readlines():
            if i != '\n':
                data.append(i)
    return data

def bbox_to_rect(bbox, color):
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2], height=bbox[3],
        fill=False, edgecolor=color, linewidth=2)

def bbox_to_yolo(bbox,w,h):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
    x = abs((x1 + x2) / (2*w))
    y = abs((y1 + y2) / (2*h))
    w = abs((x2 - x1) / w)
    h = abs((y2 - y1) / h)
    return[x,y,w,h]

def yolo_to_bbox(yolo,w,h):
    x_, y_, w_, h_ = yolo[0], yolo[1], yolo[2], yolo[3]
    x1 = w * x_ - 0.5 * w * w_
    x2 = w * x_ + 0.5 * w * w_ 
    y1 = h * y_ - 0.5 * h * h_
    y2 = h * y_ + 0.5 * h * h_
    return [x1 , y1, x2-x1, y2-y1]

# img:图片路径, label:标签路径,yolo=False:是否为yolo数据集格式,save=False:save的路径,show=False:True展示照片,classes=False:类别字典,contain=False,包含
def show_picture(img, label,yolo=False,save=False,show=False,classes=False,contain=False,only_class=False):
    img = cv2.imread(img)
    bboxs = read_common(label)
    if len(bboxs) <= 5:
        bboxs = [bboxs]
    h,w = img.shape[0],  img.shape[1]
    titles = [bbox[0] for bbox in bboxs]
    if contain:
        if contain not in titles:
            return False
    for bbox in bboxs:
        title = bbox[0]
        if classes:
            title = classes[title]
        bbox=bbox[1:]
        if only_class:
            if title not in only_class:
                continue
        if yolo:
            bbox = yolo_to_bbox(bbox,w=w,h=h)
        cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),2)
        cv2.putText(img, title,(bbox[0],bbox[1]-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    if save:
        cv2.imwrite(save,img)
    if show:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    return True

#统计每个图像的长宽分布
def bbox2yolo(image_path,label_path,classes=False):
    total = 0
    images = os.listdir(image_path)
    labels = os.listdir(label_path)
    for img in images:
        total+=1
        if total%100 == 0:
            print(total,end=" ")
        label_name = os.path.splitext(img)[0]+".txt" #取得标签名
        if label_name not in labels: continue
        img = cv2.imread(image_path+img)
        bboxs = read_common(label_path+label_name)
        h,w = img.shape[0],  img.shape[1]
        f = open('/data1/10cls/duyinglong/tools/labels/'+label_name,'w')
        for bbox in bboxs:
            title = bbox[0]
            bbox=bbox[1:]
            bbox = bbox_to_yolo(bbox,w=w,h=h)
            if classes:
                title = classes[title]
            f.write(str(title)+" "+str(bbox[0])+" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3])+'\n')
        f.close()

def convert(img_size, box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    # 转换并归一化
    center_x = (x1 + x2) * 0.5 / img_size[0]
    center_y = (y1 + y2) * 0.5 / img_size[1]
    w = abs((x2 - x1)) * 1.0 / img_size[0]
    h = abs((y2 - y1)) * 1.0 / img_size[1]

    return (center_x, center_y, w, h)



def decode_json(save_path, json_folder_path, json_name, classes):
    txt_name = save_path + json_name[0:-5] + '.txt'
    # txt保存位置

    
    json_path = os.path.join(json_folder_path, json_name)
    data = json.load(open(json_path, 'r'))

    img_w = 1280
    img_h = 720
    with open(txt_name, 'w') as txt_file:  # te files
        for i in data['labels']:
            if i['box2d']:  # 仅适用矩形框标注
                x1 = float(i['box2d']['x1'])
                y1 = float(i['box2d']['y1'])
                x2 = float(i['box2d']['x2'])
                y2 = float(i['box2d']['y2'])
                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                    continue
                else:
                    bb = (x1, y1, x2, y2)
                    bbox = convert((img_w, img_h), bb)
                
                cls = i['category']  # 得到当前label的类别
                
                # 转换成训练模式读取的标签
                cls_id = classes.index(cls)  # 位于定义类别索引位置
                
                # 保存
                txt_file.write(str(cls_id) + ' ' +" ".join([str(a) for a in bbox])+"\n")  # 生成格式0 cx,cy,w,h
                
def divide_data(root:str,save_dir,ratio:list,add = False):
    
    train_path = root + "labels/"
    images = root + 'images/'
    train_list = os.listdir(train_path) # get label list
    total = len(train_list) 
    print("len_of_images:", total)
    print("divide ratio",ratio)
    train_k =int(total * ratio[0] / 10)
    test_k = int(total * ratio[1] / 10)
    val_k = int(total * ratio[2] / 10)
    #train = [images+i for i in train]
    train = []
    test = []
    val = []
    for i in train_list:
        t = images+i.replace("txt","jpg")
        train.append(t)   
    len_of_data = len(train)
    if test_k:
        i_list = random.choices(range(len_of_data), k=test_k) #
        i_list.sort(reverse=True)
        for i in i_list:
            test.append(train.pop(i))
    len_of_data = len(train)
    if val_k:
        i_list = random.choices(range(len_of_data), k=val_k) #
        i_list.sort(reverse=True)
        for i in i_list:
            val.append(train.pop(i))
    print("train:",len(train))
    print("test:",len(test))
    print("val:",len(val))
    write2file(train, save_dir+"train.txt", add)
    write2file(test, save_dir+"test.txt", add)
    write2file(val, save_dir+"val.txt", add)
    return total,len(train),len(test),len(val)

def write2file(list, dir, add = False):
    if add:
        f = open(dir,'a')
    else:
        f = open(dir,'w')
    for i in list:
        f.write(i+"\n")
    f.close
            
       


