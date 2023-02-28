from universal_data_process import universal_bbox2yolo
from data_distribution import Analytical_data_distribution


#data_path,label_dict,train_test_val_ratio

data_path = "/data1/10cls/duyinglong/tricycle_0219/"
label_dict = {"头肩":0,"遮挡头肩":1}
train_test_val_ratio = [8,2,0]
universal_bbox2yolo(data_path=data_path,label_dict=label_dict,train_test_val_ratio=train_test_val_ratio)