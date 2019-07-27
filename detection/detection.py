import torchvision
from torchvision.transforms import functional as F
import cv2
from transforms import cv2pil, IOU_cheack
import time
import argparse
import pickle
import os
import torch

parser = argparse.ArgumentParser(description='segmentation using webcamera')
parser.add_argument('data_root', type=str, help='example: gender')
parser.add_argument('-st','--score_thr', type=float, default=0.5)
parser.add_argument('-it','--iou_thr', type=float, default=0.3)

args = parser.parse_args()
root = os.path.join('data', args.data_root)
score_thr = args.score_thr
iou_thr = args.iou_thr
f = open(os.path.join(root, 'label_to_name'), 'rb')
label_to_name = pickle.load(f)
if args.data_root == 'faster_rcnn' :
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
else:
    model = torch.load(os.path.join(root, 'model'))
    model.eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2pil(frame)
    img = F.to_tensor(img)


    pred = model([img])[0]
    boxes = pred['boxes']
    labels = pred['labels']
    scores = pred['scores']
    object_num = len(labels)

    high_score_id = []
    id_mapped = []

    for i in range(object_num):
        score = scores[i]
        if score < score_thr:
            pass
        else:
            high_score_id.append(i)


    for i in high_score_id:
        if i == 0:
            box = boxes[i]
            label = labels[i]
            score = scores[i]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
            cv2.putText(frame, label_to_name[label] + " {}%".format(int(score*100)), (box[0],box[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)
            id_mapped.append(i)
        else:
            for j in id_mapped:
                if not IOU_cheack(boxes[i], boxes[j], threshold=iou_thr):
                    break

                if j == id_mapped[-1]:
                    box = boxes[i]
                    label = labels[i]
                    score = scores[i]
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
                    cv2.putText(frame, label_to_name[label] + " {}%".format(int(score*100)), (box[0],box[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)
                    id_mapped.append(i)

    cv2.namedWindow('Detection frame', cv2.WINDOW_NORMAL)
    cv2.imshow('Detection frame', frame)    
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()