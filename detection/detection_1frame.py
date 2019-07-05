import torchvision
from torchvision.transforms import functional as F
from transforms import read_img, cv2pil, IOU_cheack
import cv2
import argparse

parser = argparse.ArgumentParser(description="segmentaion_1frame")
parser.add_argument('img_path')
parser.add_argument('-st','--score_thr', type = float, default=0.5)
parser.add_argument('-it','--iou_thr', type = float, default=0.3)
args = parser.parse_args()

img_path = args.img_path
score_thr = args.score_thr
iou_thr = args.iou_thr



model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
label_to_name = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

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


cv2.imshow('Detection', img_cv)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()