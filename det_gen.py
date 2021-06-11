import os
import csv
import numpy as np
import cv2
from albumentations import (ToGray, OneOf, Compose, RandomBrightnessContrast,
                            RandomGamma, GaussianBlur, MotionBlur, ToSepia, InvertImg, RandomSnow, RandomSunFlare,
                            RandomRain, RandomShadow, HueSaturationValue, HorizontalFlip)
from albumentations import BboxParams, ShiftScaleRotate, Transpose, Cutout, VerticalFlip, GaussNoise, JpegCompression
from tensorflow.keras.utils import Sequence
from matplotlib import pyplot as plt


class CSVGenerator(Sequence):
    def __init__(self, annotations_path,
                 img_height,
                 img_width,
                 batch_size,
                 img_dir,
                 augs):
        self.annotations_path = annotations_path
        self.annotations = {}
        self.out_height = img_height
        self.out_width = img_width
        self.augs = augs
        self.img_dir = img_dir

        self.classes = {}
        self.num_classes = 1

        with open(self.annotations_path, 'r') as f:
            csv_reader = csv.reader(f)

            for row in csv_reader:
                if row[0] not in self.annotations:
                    self.annotations[row[0]] = []

                x_min = -1
                y_min = -1
                x_max = -1
                y_max = -1
                class_id = -1

                if row[1] != '':
                    x_min = int(float(row[1]))

                if row[2] != '':
                    y_min = int(float(row[2]))

                if row[3] != '':
                    x_max = int(float(row[3]))

                if row[4] != '':
                    y_max = int(float(row[4]))

                if row[5] != '':
                    class_id = int(row[5])

                if x_min > 0 or y_min > 0 or x_max > 0 or y_max > 0 or class_id > 0:
                    annotation = [x_min, y_min, x_max, y_max, class_id]
                    self.annotations[row[0]].append(annotation)

        self.images_list = list(self.annotations.keys())
        print(len(self.images_list))

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        print(self.batch_size)

    def __len__(self):
        return int(len(self.images_list) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.images_list[idx * self.batch_size: (idx + 1) * self.batch_size]

        imgs = np.empty((self.batch_size, self.img_height, self.img_width, 3), dtype=np.float32)
        batch_centers = np.zeros((self.batch_size, self.out_height, self.out_width, self.num_classes + 2),
                                 dtype=np.float32)

        for i in range(len(batch)):
            img_path = os.path.join(self.img_dir, batch[i])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(batch[i])
            annotations = self.annotations[batch[i]].copy()
            new_annotations = []
            if self.augs is not None:
                boxes = []
                category_ids = []

                for j in range(len(annotations)):
                    if annotations[j][2] - annotations[j][0] !=  0 and annotations[j][3] - annotations[j][1] != 0:
                        boxes.append([annotations[j][0], annotations[j][1],
                                      annotations[j][2], annotations[j][3]])
                        category_ids.append(annotations[j][4])

                data = {'image': img, 'bboxes': boxes, 'category_id': category_ids}
                augmented = self.augs(**data)
                img = augmented['image']
                annotations = []

                for j in range(len(augmented['bboxes'])):
                    annotations.append([int(np.round(augmented['bboxes'][j][0])),
                                        int(np.round(augmented['bboxes'][j][1])),
                                        int(np.round(augmented['bboxes'][j][2])),
                                        int(np.round(augmented['bboxes'][j][3])),
                                        augmented['category_id'][j]])

            else:
                for j in range(len(annotations)):
                    if annotations[j][2] - annotations[j][0] !=  0 and annotations[j][3] - annotations[j][1] != 0:
                        new_annotations.append([annotations[j][0], annotations[j][1],
                                                annotations[j][2], annotations[j][3], annotations[j][4]])
                annotations = new_annotations

            img = img.astype(np.float32)
            img /= 255
            imgs[i] = img                

            centers = np.zeros((self.out_height, self.out_width, self.num_classes), dtype=np.float32)
            scales = np.zeros((self.out_height, self.out_width, 2), dtype=np.float32)

            for j, bbox in enumerate(annotations):
                if (bbox[0] < 0): 
                    bbox[0] = 0
                if (bbox[1] < 0):
                    bbox[1] = 0 
                if (bbox[2] >= self.out_width):
                    bbox[2] = self.out_width - 1
                if (bbox[3] >= self.out_height):
                    bbox[3] = self.out_height - 1
                if  (bbox[2] - bbox[0] == 0) or (bbox[3] - bbox[1] == 0) or (bbox[2] < bbox[0]) or (bbox[3] < bbox[1]):
                    continue

                h = bbox[3] - bbox[1]
                sc_h = h / (self.out_height)
                if sc_h > 1.0:
                    sc_h = 1.0
                if sc_h < 0:
                    sc_h = 0
                    
                w = bbox[2] - bbox[0]
                sc_w = w / (self.out_width)
                if sc_w > 1.0:
                    sc_w = 1.0
                if sc_w < 0:
                    sc_w = 0

                rhmin = -int(h / 2)
                rhmax = int(h / 2)
                rwmin = -int(w / 2)
                rwmax = int(w / 2)

                if h % 2 != 0:
                    rhmax = int(h / 2) + 1

                if w % 2 != 0:
                    rwmax = int(w / 2) + 1

                y, x = np.ogrid[rhmin:rhmax, rwmin:rwmax]

                e = np.exp(-((x * x / (2 * (w / 12.0) * (w / 12.0)) + (y * y / (2 * (h / 12.0) * (h / 12.0))))))

                xmin = bbox[0]
                ymin = bbox[1]

                xmax = xmin + e.shape[0]
                ymax = ymin + e.shape[1]

                center_x = int((xmax + xmin) / 2)
                center_y = int((ymax + ymin) / 2)

                if xmax >= centers.shape[1] or ymax >= centers.shape[0]:
                    continue

                tmp = np.zeros_like(centers)
                
                try:
                    tmp[ymin:ymin + e.shape[0], xmin:xmin + e.shape[1], 0] = e
                except:
                    print(xmin)

                centers = np.where(tmp > centers, tmp, centers)
                centers[int(center_y - 1), int(center_x - 1), 0] = 1.0
                scales[int(center_y - 1), int(center_x - 1), 0] = sc_h
                scales[int(center_y - 1), int(center_x - 1), 1] = sc_w

            batch_centers[i] = np.concatenate([centers, scales], axis=2)

        return imgs, batch_centers


if __name__ == '__main__':
    gen = CSVGenerator('annotations.csv',
                       # 'class.csv',
                       240,
                       320,
                       1,
                       'images',
                       None)
    print(len(gen))
    for i in range(2583):
        imgs, centers = gen.__getitem__(i)
        plt.imshow(centers[0, :, :, 0])
        plt.show()
        # print(np.sum(centers[0, :, :, 2]))
        # print(np.count_nonzero(centers[0, :, :, 2]))
