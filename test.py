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

from enet_det.decode import decode_centers_and_scales
from enet_det.train import my_loss

model = load_model('./models/enet_val_18.h5', compile=False)
model.compile(optimizer=Nadam(), loss=my_loss)
imgs = list()
avg_time = 0

# Работаем с одной картинкой

img_path = os.path.join(os.getcwd(), '6.jpg')
img = cv2.imread(img_path,cv2.IMREAD_COLOR)
orig_img = img.copy()
img = cv2.resize(img,dsize=(320,240), interpolation=cv2.INTER_AREA)

img = img.astype(np.float32)
img /= 255.
img = np.reshape(img, (1, img.shape[0], img.shape[1], 3))

output = model.predict(img)

batch_detections = decode_centers_and_scales(output)

for detections in batch_detections:
    for detection in detections:
        pass