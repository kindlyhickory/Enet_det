import csv
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

datasets = [
    'route1',
    'route2',
    'route3',
    'route4',
    'route5',
    'route6',
    'route7',
    'route8',
]

det_annotations = []
images = []
classes = [
    'people',
]

for dataset in datasets:
    img_list = list(os.listdir(os.path.join('adas', dataset, 'img')))
    for img_name in tqdm(img_list):
        if img_name in images:
            continue
        else:
            images.append(img_name)
        annotation_name = img_name + '.json'
        ann_path = os.path.join('adas', dataset, 'ann', img_name + '.json')
        with open(ann_path, 'r') as f:
            ann = json.load(f)

        img_path = os.path.join('adas', dataset, 'img', img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_h = img.shape[0]
        orig_w = img.shape[1]
        scale_h = 256 / orig_h
        scale_w = 256 / orig_w

        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        out_path = os.path.join('images', dataset + img_name)
        cv2.imwrite(out_path, img)

        #if len(ann['objects']) == 0:
        #    continue

        check = False

        for object in ann['objects']:
            if object['classTitle'] == 'people':
                check = True
                x_min = int(object['points']['exterior'][0][0])
                y_min = int(object['points']['exterior'][0][1])
                x_max = int(object['points']['exterior'][1][0])
                y_max = int(object['points']['exterior'][1][1])

                x_min = x_min * scale_w
                y_min = y_min * scale_h
                x_max = x_max * scale_w
                y_max = y_max * scale_h

                new_row = (dataset+img_name, np.round(x_min), np.round(y_min), np.round(x_max), np.round(y_max), 1)
                det_annotations.append(new_row)
        if not check:
            print('no pedestrian')
            new_row = (dataset + img_name, '', '', '', '', '')
            det_annotations.append(new_row)


with open('annotations.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(det_annotations)
