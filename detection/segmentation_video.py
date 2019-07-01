import torchvision
from torchvision.transforms import functional as F
import cv2
from transforms import cv2pil
import time
import argparse

parser = argparse.ArgumentParser(description='segmentation using webcamera')
parser.add_argument('base_model', choices=['faster_rcnn', 'mask_rcnn', 'keypoint_rcnn'])
args = parser.parse_args()

base_model = args.base_model

label_to_name_1 = [
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

label_to_name_2 = ['__background__', 'person']
keypoint_label = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]
if base_model != 'keypoint_rcnn':
    if base_model == 'faster_rcnn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    model.eval()
    cap = cv2.VideoCapture(0)

    time_sum = 0
    
    for i in range(20):
        start = time.time()

        ret, frame = cap.read()
        img = cv2pil(frame)
        img = F.to_tensor(img)
        
        
        pred = model([img])
        pred = pred[0]
        object_num = len(pred['labels'])
        
        

        for i in range(object_num):

            box = pred['boxes'][i]
            label = pred['labels'][i]
            score = pred['scores'][i]
            
            if score < 0.5:
                pass
            else:    
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
                cv2.putText(frame, label_to_name_1[label] + " acc: {}%".format(int(score*100)), (box[0],box[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)
                
        
        cv2.imshow('Detection frame', frame)        
        elapsed_time = time.time() - start
        time_sum += elapsed_time

        k = cv2.waitKey(1)
        
        if k == 27:
            break

else:
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    cap = cv2.VideoCapture(0)

    time_sum = 0
    
    for i in range(20):
        start = time.time()

        ret, frame = cap.read()
        img = cv2pil(frame)
        img = F.to_tensor(img)
        
        
        pred = model([img])
        pred = pred[0]
        object_num = len(pred['labels'])
        
        
        

        for i in range(object_num):
            box = pred['boxes'][i]
            label = pred['labels'][i]
            score = pred['scores'][i]
            keypoint = pred['keypoints'][i]
            keypoint_num = len(keypoint)
            print(keypoint.size(), 'keypoint_size')
            print(keypoint, 'keypoint')

            if score < 0.5:
                pass
            else:    
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
                cv2.putText(frame, label_to_name_2[label] + " acc: {}%".format(int(score*100)), (box[0],box[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)
                for j in range(keypoint_num):
                    key_part = keypoint[j]
                    print(key_part.size(), 'key_part_size')
                    print(key_part, 'keypart') ##これがおそらくベクトル。
        cv2.imshow('Detection frame', frame)        
        elapsed_time = time.time() - start
        time_sum += elapsed_time

        k = cv2.waitKey(1)
        
        if k == 27:
            break


print(time_sum/20)

cap.release()
cv2.destroyAllWindows()