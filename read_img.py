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


def read_img(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return 

def vis_img(img):
    plt.imshow(image_rgb), plt.axis("off")
    plt.show()

def transform(img, input_size):
    img = cv2.resize(img, (input_size, input_size))
    return img


def recognize(model_ft, img):
    output = model_ft(img)
    return output
    


if __name__ == '__main__':
    
    ####finetune
    label_to_name = ["dog", "cat", "rabbit"]
    model_name = "alexnet"
    num_classes = 3
    feature_extract = True
    data_dir = "./data/animals"
    bacth_size = 2
    num_epochs = 10


    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    print("Initializing Datasets and Dataloaders...")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model_ft = model_ft.to(device)


    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)


    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))







    #test
    img_path = "./data/dog.jpg"
    img = read_img(path=img_path)
    img = transform(img, input_size)

    vis_img(img=img)  # check vis

    label = recognize(model_ft=model_ft, img=img)

    print(label)
    print(label_to_name[label])



