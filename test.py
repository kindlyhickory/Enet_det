import numpy as np
import os
import argparse
import cv2
import keras as keras
import tensorflow as tf
from keras.optimizers import Adam, Nadam, Adadelta
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from albumentations import (ToGray, OneOf, Compose, RandomBrightnessContrast,
                            RandomGamma, GaussianBlur, ToSepia, MotionBlur, InvertImg, HueSaturationValue, VerticalFlip,
                            HorizontalFlip, ShiftScaleRotate, RandomShadow, RandomRain, Rotate)
from albumentations.core import composition
from albumentations.core.composition import BboxParams
from tqdm import tqdm
from keras.models import load_model
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter
import time

from decode import decode_centers_and_scales
from train import my_loss

model = load_model('./models/enet_val_83.h5', compile=False)
model.compile(optimizer=Nadam(), loss=my_loss)
imgs = list()
avg_time = 0

# Работаем с одной картинкой

img_path = os.path.join(os.getcwd(), '6.jpg')
print(img_path)
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, dsize=(320, 240), interpolation=cv2.INTER_AREA)
orig_img = img.copy()

img = img.astype(np.float32)
img /= 255.
img = np.reshape(img, (1, img.shape[0], img.shape[1], 3))

output = model.predict(img)

batch_detections = decode_centers_and_scales(output, 0.4, 100)

img_signed = orig_img.copy()
for detections in batch_detections:
    for i in range(len(detections)):
        h = detections[i][-2] * orig_img.shape[0] * 4
        w = detections[i][-1] * orig_img.shape[1] * 4
        cx = int(detections[i][1]) * 4
        cy = int(detections[i][2]) * 4
        # 4 eto vo skolko raz vihod setki menshe vhoda

        xmin = int(cx - w / 2)
        ymin = int(cy - h / 2)
        xmax = int(cx + w / 2)
        ymax = int(cy + h / 2)

        color = (0, 0, 255)

        cv2.rectangle(img_signed, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_AA)
    cv2.imwrite('/home/user-103/Detection/Enet_det/img_out/6.jpg', img_signed)
    