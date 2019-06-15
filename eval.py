import os
from torchvision import transforms, datasets
import torch
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description="evaluation")
parser.add_argument("data_name", type=str, help="example: 'ants_bees'")
parser.add_argument("--batch_size", type=int, default=4)

args = parser.parse_args()

data_name = args.data_name
data_path = "./data/" + data_name
model_path = data_path + "/model"
label_to_name_path = data_path + "/label_to_name"
img_path = data_path + "/img"
batch_size = args.batch_size

f = open(label_to_name_path, 'rb')
label_to_name = pickle.load(f)

num_classes = len(label_to_name)



data_transforms =transforms.Compose([
    transforms.Resize(224), ####input_sizeの一般化ができていない。
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


image_datasets = datasets.ImageFolder(os.path.join(img_path, "val"), data_transforms)

dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4) 


model = torch.load(model_path)
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        ###batch_sizeは４, class数は２
        #print("labels.size()", labels.size()) torch.Size([4])
        outputs = model(images)
        # print("outputs.size()", outputs.size()) torch.Size([4, 2])
        _, predicted = torch.max(outputs, 1)
        #print("predicted.size" ,predicted.size()) torch.Size([4])
        c = (predicted == labels).squeeze() 
        
        if c.size() != torch.Size([4]):
            break      
        #print("c.size()", c.size()) torch.Size([4])




        ###データ数がバッチサイズで割り切れないため、cが最後だけtorch.Size([])となっている。
        ###これが原因で、86行目でindexerrorが生じる。








        for i in range(batch_size):

            label = labels[i]
            #print('type of label') -- <class 'torch.Tensor'>
            #print("label.size()", label.size()) torch.Size([])
            #print("type of c[i]") -- <class 'torch.Tensor'>
            #print("c[i].size()", c[i].size()) torch.Size([])
            

            class_correct[label] += c[i].item()  #####ここで67行目で指摘したエラーが発生
            class_total[label] += 1


for i in range(num_classes):
    print('Accuracy of %5s : %2d %%' % (
        label_to_name[i], 100 * class_correct[i] / class_total[i]))


class_acc_list = [100 * class_correct[i] / class_total[i] for i in range(num_classes)]



left = [i for i in range(num_classes)]
height = class_acc_list

 
plt.bar(left, height, width=0.5, color='#0096c8',
        edgecolor='b', linewidth=2, tick_label=label_to_name)
plt.hlines(100/num_classes, xmin=-1,  xmax=num_classes, linestyles='dashed')
plt.xlabel("class_name")
plt.ylabel("accuracy")
plt.show()