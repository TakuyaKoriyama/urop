import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from finetune_helper_function import train_model, set_parameter_requires_grad, initialize_model
import argparse
import pickle

parser = argparse.ArgumentParser(description="recognizer")


parser.add_argument("pretrained_model_name", type = str, default='alexnet', 
    choices=["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"])
parser.add_argument("data_name", type=str, help="example: 'ants_bees'")
parser.add_argument("class_name", type=str, nargs="*", help="example: ants bees")
parser.add_argument("-f_e", "--feature_extract", action="store_true", help="feature extract will be activated.")

parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=15)
parser.add_argument("--lr", type=float, default = 0.001)
parser.add_argument("--momentum", type=float, default=0.9)


args = parser.parse_args()


data_name = args.data_name ##'ants_bees'
label_to_name = args.class_name ###["ants", "bees"]
data_path = "./data/" + data_name  ### "./data/ants_bees"
label_to_name_path = data_path + "/label_to_name"
weight_path = data_path + "/weight.pt"
img_path = data_path + "/img"


pretrained_model_name = args.pretrained_model_name
num_classes = len(label_to_name)
feature_extract = args.feature_extract
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
momentum = args.momentum


model, input_size = initialize_model(pretrained_model_name, num_classes, feature_extract, use_pretrained=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
params_to_update = model.parameters()
optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)
criterion = nn.CrossEntropyLoss()




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
image_datasets = {x: datasets.ImageFolder(os.path.join(img_path, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}




print("Params to learn:")

if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)






model, _ = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, is_inception=(pretrained_model_name=="inception"))

torch.save(model.state_dict(), weight_path) 


f = open(label_to_name_path, 'wb')
pickle.dump(label_to_name, f)



