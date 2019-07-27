import torchvision
from torchvision.transforms import functional as F
from transforms import read_img, cv2pil, IOU_cheack
import cv2
import argparse
import pickle
import torch
import os

parser = argparse.ArgumentParser(description="detection_1frame")
parser.add_argument('img_path', type=str)
parser.add_argument('data_root', type=str, help='example: gender')
parser.add_argument('-st','--score_thr', type = float, default=0.5)
parser.add_argument('-it','--iou_thr', type = float, default=0.3)
args = parser.parse_args()

img_path = args.img_path
root = os.path.join('data', args.data_root)
score_thr = args.score_thr
iou_thr = args.iou_thr
f = open(os.path.join(root, 'label_to_name'), 'rb')
label_to_name = pickle.load(f)

if args.data_root == 'faster_rcnn':
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
else:
    model = torch.load(os.path.join(root, 'model'))
    model.eval()


img_cv = read_img(img_path)
img_pil = cv2pil(img_cv)
img_ts = F.to_tensor(img_pil)

pred = model([img_ts])[0]
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
        cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
        cv2.putText(img_cv, label_to_name[label] + " {}%".format(int(score*100)), (box[0],box[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)
        id_mapped.append(i)
    else:
        for j in id_mapped:
            if not IOU_cheack(boxes[i], boxes[j], threshold=iou_thr):
                break

            if j == id_mapped[-1]:
                box = boxes[i]
                label = labels[i]
                score = scores[i]
                cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
                cv2.putText(img_cv, label_to_name[label] + " {}%".format(int(score*100)), (box[0],box[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)
                id_mapped.append(i)

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.imshow('Detection', img_cv)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()