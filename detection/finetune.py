import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import argparse
import pickle


class Dataset(object): 
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        #リストの最初は画像ではない物がくる。＜＜＜＜＜それにすら対応してるのかな。。。。？
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))[1:] 
        txt = open(os.path.join(root, 'box_label.txt'), 'r')
        boxes_al = []
        labels_al = []
        for i in range(len(self.imgs)):
            boxes_al.append([])
            labels_al.append([])
        
        for line in txt:
            x = line.split(',')
            for i in range(len(x)):
                x[i] = int(x[i])
            obj_id = x[0]
            
            label = x[2]
            box = x[3:]
            boxes_al[obj_id].append(box)
            labels_al[obj_id].append(label) 
        
        self.boxes_al = boxes_al
        self.labels_al = labels_al
        

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        boxes = self.boxes_al[idx]
        labels = self.labels_al[idx]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        num_objs = len(labels)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    parser = argparse.ArgumentParser(description='finetune')
    parser.add_argument('data_root', type = str, help = 'example: gender')
    args = parser.parse_args()

    root = os.path.join('data', args.data_root)
    f = open(os.path.join(root, 'label_to_name'), 'rb')
    label_to_name =pickle.load(f)
    num_classes = len(label_to_name)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # use our dataset and defined transformations
    dataset = Dataset(root, get_transform(train=True))
    dataset_test = Dataset(root, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
    
    torch.save(model, os.path.join(root, 'model'))
if __name__ == "__main__":
    main()
