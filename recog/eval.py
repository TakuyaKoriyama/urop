import os
from torchvision import transforms, datasets
import torch
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from finetune_helper_function import load_model

parser = argparse.ArgumentParser(description="evaluation")
parser.add_argument("data_name", type=str, help="example: 'ants_bees'")
parser.add_argument("--batch_size", type=int, default=4)

args = parser.parse_args()

data_name = args.data_name
data_path = "./data/" + data_name

label_to_name_path = data_path + "/label_to_name"
img_path = data_path + "/img"
batch_size = args.batch_size

f = open(label_to_name_path, 'rb')
label_to_name = pickle.load(f)

num_classes = len(label_to_name)

data_transforms =transforms.Compose([
    transforms.Resize(224), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


image_datasets = datasets.ImageFolder(os.path.join(img_path, "val"), data_transforms)

dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4) 


model = load_model(root=data_path, num_classes=num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze() 
        if len(c) == 1:
            c = torch.reshape(c, (-1, len(c)))






        for i in range(len(images)): 

            label = labels[i]
            class_correct[label] += c[i].item()  
            class_total[label] += 1


print(label_to_name)
for i in range(num_classes):
    print('Accuracy of %5s : %2d %%' % (
        label_to_name[i], 100 * class_correct[i] / class_total[i]))

print(label_to_name)
class_acc_list = [100 * class_correct[i] / class_total[i] for i in range(num_classes)]



left = [i for i in range(num_classes)]
height = class_acc_list

 
plt.bar(left, height, width=0.5, color='#0096c8',
        edgecolor='b', linewidth=2, tick_label=label_to_name)
plt.hlines(100/num_classes, xmin=-1,  xmax=num_classes, linestyles='dashed')
plt.xlabel("class_name")
plt.ylabel("accuracy")
plt.show()