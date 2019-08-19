import cv2
import numpy as np
import os
import argparse
import pickle
parser = argparse.ArgumentParser(description="annotation")
parser.add_argument('data_root', type=str, default='gender', help='example: gender')

args = parser.parse_args()

root = os.path.join('data', args.data_root)
label_to_name_path = os.path.join(root, 'label_to_name')
box_label_path = os.path.join(root,'box_label.txt')
imgs_list = list(sorted(os.listdir(os.path.join(root, "Images"))))

if imgs_list[0] == '.DS_Store':
    imgs_list.pop(0)

print('{}pictures annotated'.format(len(imgs_list)))

f = open(label_to_name_path, 'rb')
label_to_name = pickle.load(f)
txt = open(os.path.join(root, 'box_label.txt'), 'r')

boxes_al = []
labels_al = []
for i in range(len(imgs_list)):
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

for i in range(len(imgs_list)):
    img_path = os.path.join(root, "Images", imgs_list[i])
    boxes  = boxes_al[i]
    labels = labels_al[i]
    num_objs = len(labels)
    im = cv2.imread(img_path)
    window_name = 'Image:{}'.format(i)
    #print('{}boxes annotated'.format(num_objs))
    for j in range(num_objs):
        box = boxes[j]
        label = labels[j]
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=2)
        cv2.putText(im, label_to_name[label], (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow(window_name, im)
        k = cv2.waitKey(0)
        if k == 27:
            print('enter class_label')
            s = cv2.waitKey(0)
            s = s - 48
            if 0 <= s < len(label_to_name):
                print('class :{}'.format(label_to_name[s]))
                r = cv2.selectROI(window_name, im)
                class_label = s
                x_0 = str(r[1])
                x_1 = str(r[1]+r[3])
                y_0 = str(r[0])
                y_1 = str(r[0]+r[2])
                print(str(i) + ',' + str(j) + ',' +  str(class_label) + ',' + x_0 + ',' + y_0 + ',' + x_1 + ',' + y_1)
            else:
                print('enter number from {} to {}'.format(0, len(label_to_name)-1))

        if j == num_objs -1:
            cv2.destroyAllWindows()


