# coding:utf-8

import matplotlib.pyplot as plt
from draw import box_to_rect
from data_loader import train_data,rgb_mean,rgb_std,class_names
from dataloader import *
myloader = MyLoader("image2/*.pick", w=256, h=256, s=4, b=2, c=2)
from utils import *
from model import *
from mxnet import cpu
"""
train mode:
set the params depend on your case.
then train the model and save the model's weight. 
"""
params = {"class_num": 2,
          "class_name":class_names,
          "epoch": 5,
          "layer_num": 6,
          "s": 4,
          "b": 2,
          "verbose": True
          }


yolo = train(params, train_data)
save(yolo, pre="yolo")
"""
predict mode:
set the same params as same as train mode.
then load the model's weight from yolo.params and predict.   
"""
# yolo = Yolo(6, 2,class_name=class_names, s=4, b=2)
# yolo.load_params("yolo.params", ctx=cpu(0))
#
# data,img = process_image("1.jpg",data_shape=256,rgb_mean=rgb_mean,rgb_std=rgb_std)
#
# class_name,c_list,boxs = predict(yolo,data)
#
# for i in range(len(class_name)):
#     text = class_name[i]
#     score = c_list[i]
#     box = boxs[i]
#     plt.imshow(img.asnumpy())
#     rect = box_to_rect(box, 'red', 2.5)
#     plt.gca().add_patch(rect)
#     plt.gca().text(box[0], box[1], '{:s} {:.2f}'.format(text, score),
#                    bbox=dict(facecolor="red", alpha=0.5), fontsize=9, color='white')
# plt.show()
