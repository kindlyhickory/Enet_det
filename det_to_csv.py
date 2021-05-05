import csv
import json
import os

import cv2
from tqdm import tqdm

datasets = [
    'adas/route1',
    'adas/route2',
    'adas/route3',
    'adas/route4',
    'adas/route5',
    'adas/route6',
    'adas/route7',
    'adas/route8',
]

det_annotations = []
images = []
classes = [
    'people',
]

for dataset in datasets:
    img_list = list(os.listdir(os.path.join(dataset, 'img')))
    for img_name in tqdm(img_list):
        if img_name in images:
            continue
        else:
            images.append(img_name)
        annotation_name = img_name + '.json'
        ann_path = os.path.join(dataset, 'ann', img_name + '.json')
        with open(ann_path, 'r') as f:
            ann = json.load(f)

        img_path = os.path.join(dataset, 'img', img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_h = img.shape[0]
        orig_w = img.shape[1]
        scale_h = 256 / orig_h
        scale_w = 256 / orig_w

        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        out_path = os.path.join('images', img_name)
        cv2.imwrite(out_path, img)

        if len(ann['objects']) == 0:
            continue

        for object in ann['objects']:
            if object['classTitle'] == 'people':
                x_min = int(object['points']['exterior'][0][0])
                y_min = int(object['points']['exterior'][0][1])
                x_max = int(object['points']['exterior'][1][0])
                y_max = int(object['points']['exterior'][1][1])

                x_min = x_min * scale_w
                y_min = y_min * scale_h
                x_max = x_max * scale_w
                y_max = y_max * scale_h

                new_row = (img_path, x_min, y_min, x_max, y_max, 1)
                det_annotations.append(new_row)

with open('annotations.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(det_annotations)
