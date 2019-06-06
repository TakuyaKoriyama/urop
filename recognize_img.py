from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
from finetune_helper_function import train_model, set_parameter_requires_grad, initialize_model
from sklearn.externals import joblib

def read_img(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def vis_img(img):
    plt.imshow(img), plt.axis("off")
    plt.show()

def transform(img): ###訓練した時と同じような前処理をするcodeをopencvで書きたい←transformsモジュールをよく見る
    img = cv2.resize(img, (224,224))
    return img

def recognize(model_ft, img):
    labels = model_ft(img)
    return labels

model_ft = joblib.load("ant_classifier.pkl")

label_to_name = ["ants", "bees"]

img_path = "./img/test_img/ants.jpg"
img = read_img(img_path=img_path)
img = transform(img)

vis_img(img=img)  
img = torch.tensor(img) ###tensorのsizeがモデルの期待する形と会ってない←バッチサイズも含めて4ベクトルだから、次元を増やす必要がある。
label = recognize(model_ft=model_ft, img=img)

print(label)
print(label_to_name[label])



