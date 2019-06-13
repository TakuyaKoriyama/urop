import os
from torchvision import transforms, datasets
import torch
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

parser = argparse.ArgumentParser(description="evaluation")
parser.add_argument("--data_dir", type=str, required=True)
parser.add-argument("--model_path", type=str, required=True)
parser.add_argument("--label_to_name_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)

args = parser.parse_args()

data_dir = args.data_dir
model_path = args.model_path
label_to_name_path = args.label_to_name_path
batch_size = args.batch_size

f = open(label_to_name_path, 'rb')
label_to_name = pickle.load(f)

num_classes = len(label_to_name)



data_transforms =transforms.Compose([
    transforms.Resize(224), ####input_sizeの一般化ができていない。
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
image_datasets = datasets.ImageFolder(os.path.join(data_dir, "val")

dataloader=torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4) ###syntaxerror何故？？？？


model = torch.load(model_path)
model.eval()


class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
with torch.no_grad():
    for data in dataloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(num_classes):
    print('Accuracy of %5s : %2d %%' % (
        label_to_name[i], 100 * class_correct[i] / class_total[i]))


class_acc_list = [100 * class_correct[i] / class_total[i]: for i in range(num_classes)]


sns.set()
sns.set_style('whitegrid')
sns.set_palette('gray')
y = np.array(class_acc_list)
x = np.array(label_to_name)

x_position = np.arange(len(x))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(x_position, y, tick_label=x)
ax.set_xlabel('accuracy')
ax.axhline(1/num_classes)
ax.set_ylabel('class')
fig.show()














