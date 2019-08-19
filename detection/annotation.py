import cv2
import numpy as np
import os
import argparse
import pickle
parser = argparse.ArgumentParser(description="annotation")
parser.add_argument('data_root', type=str, help='example: gender')
parser.add_argument('class_name', type=str, nargs="*", help="example: male female")

args = parser.parse_args()

root =  os.path.join('data', args.data_root)
label_to_name = args.class_name
f = open(os.path.join(root, 'label_to_name'), 'wb')
pickle.dump(label_to_name, f)

imgs_list = list(sorted(os.listdir(os.path.join(root, "Images"))))
if imgs_list[0] == '.DS_Store':
    imgs_list.pop(0)
print('{}pictures to annotate'.format(len(imgs_list)))
box_label_path = os.path.join(root, 'box_label.txt')
file = open(box_label_path, 'a')
for i in range(len(imgs_list)):
    img_path = os.path.join(root, "Images", imgs_list[i])
    window_name = 'Image:{}'.format(i)
    im = cv2.imread(img_path)
    #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) ここ
    cv2.imshow(window_name, im)
    box_id = 0
    while True:
        print('the next box_id:{} enter waitkey'.format(box_id))
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif 48 <= k < len(label_to_name) + 48:
            k = k - 48
            print('class :{}'.format(label_to_name[k]))
            r = cv2.selectROI(window_name, im)
            class_label = k
            y_0 = str(r[1])
            y_1 = str(r[1]+r[3])
            x_0 = str(r[0])
            x_1 = str(r[0]+r[2])
            file.write(str(i) + ',' + str(box_id) + ',' +  str(class_label) + ',' + x_0 + ',' + y_0 + ',' + x_1 + ',' + y_1 + '\n')
            box_id += 1
        else:
            print('enter number from {} to {} or go to next img by push esc botton'.format(0, len(label_to_name)-1))

file.close()