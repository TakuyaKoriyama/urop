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


parser.add_argument("--pretrained_model_name", type = str, default='alexnet', 
    choices=["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"])
parser.add_argument("--new_model_name", type=str, required=True)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--feature_extract", action="store_true")
parser.add_argument("--data_dir", type=str, required=True, help="ex) ./img/animals")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--class_name", type=str, nargs="*", required=True)
parser.add_argument("--lr", type=float, default = 0.001)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument('--save_model', action='store_true', default=False)

args = parser.parse_args()

model_name = args.pretrained_model_name
new_model_name = args.new_model_name
new_model_path = "./model/" + new_model_name + ".pkl"
new_model_label_path = "./label_to_name/" + new_model_name + ".label_to_name"
new_model_hist_path = "./hist/" + new_model_name + ".hist"
num_classes = args.num_classes
feature_extract = args.feature_extract
data_dir = args.data_dir
batch_size = args.batch_size
num_epochs = args.num_epochs
label_to_name = args.class_name
lr = args.lr
momentum = args.momentum
save_model = args.save_model

model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

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
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
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






model, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))

if save_model:
    torch.save(model, new_model_path) 
    
    f = open(new_model_label_path, 'wb')
    pickle.dump(label_to_name, f)

else:
    pass


####改善点###
#label_to_nameを専用のフォルダを作ってそこに保存して、recognizeするときにlaodするという手法を取っているが、
# そうではなくて、モデルにlabel_to_nameを付与するようなcodeを描きたい