import torchvision
from torchvision.transforms import functional as F
import cv2
from transforms import cv2pil, IOU_cheack
import time
import argparse

parser = argparse.ArgumentParser(description='segmentation using webcamera')
parser.add_argument('-frame_num', default=20)
parser.add_argument('-score_thr', default=0.5)
parser.add_argument('-iou_thr', default=0.5)
args = parser.parse_args()

frame_num = args.frame_num
score_thr = args.score_thr
iou_thr = args.iou_thr

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

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
cap = cv2.VideoCapture(0)
time_sum = 0
for i in range(frame_num):
    start = time.time()

    ret, frame = cap.read()
    img = cv2pil(frame)
    img = F.to_tensor(img)
    
    
    pred = model([img])[0]
    boxes = pred['boxes']
    labels = pred['labels']
    scores = pred['scores']
    object_num = len(labels)

    high_score_id = []

    for i in range(object_num):
        score = scores[i]
        if score < score_thr:
            pass
        else:
            high_score_id.append(i)
    
    for i in high_score_id:
        for j in high_score_id:
            if not IOU_cheack(boxes[i], boxes[j]):
                break
            else:
                continue
        box = boxes[i]
        label = labels[i]
        score = scores[i]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
        cv2.putText(frame, label_to_name[label] + " {}%".format(int(score*100)), (box[0],box[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)
        

    cv2.imshow('Detection frame', frame)        
    elapsed_time = time.time() - start
    time_sum += elapsed_time

    k = cv2.waitKey(1)
    
    if k == 27:
        break

print(time_sum/frame_num)
cap.release()
cv2.destroyAllWindows()