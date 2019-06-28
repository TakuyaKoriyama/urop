import torchvision
from torchvision.transforms import functional as F
import cv2
from transforms import cv2pil, mask_transform


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

cap = cv2.VideoCapture(0)

while True:
    
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
        

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
        cv2.putText(frame, label_to_name[label] + " acc: {}%".format(int(score*100)), (box[0],box[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)
        
    
    cv2.imshow('Detection frame', frame)

    # for i in range(object_num):
    #     mask = pred['masks'][i]
    #     mask_ed = mask_transform(mask)
    #     cv2.imshow('mask' + str(i), mask_ed)
    
    k = cv2.waitKey(1)
    
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()