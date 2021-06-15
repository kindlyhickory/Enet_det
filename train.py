import numpy as np
import os
import argparse
import cv2
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Nadam, Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from albumentations import (ToGray, OneOf, Compose, RandomBrightnessContrast, JpegCompression, Cutout, GaussNoise,
                            IAAAffine, IAAPerspective,
                            RandomGamma, GaussianBlur, ToSepia, MotionBlur, InvertImg, Transpose, HueSaturationValue,
                            VerticalFlip, HorizontalFlip, ShiftScaleRotate, RandomShadow, RandomRain, Rotate,
                            RandomGridShuffle, RandomRotate90, MedianBlur, CLAHE, IAASharpen)
from albumentations.core import composition
from albumentations.core.composition import BboxParams
from det_gen import CSVGenerator
from net import make_net


def strong_aug(p=0.75):
    return Compose([
        OneOf([
            ShiftScaleRotate(shift_limit=0.125),
            # RandomRotate90(),
            VerticalFlip(),
            HorizontalFlip(),
            IAAAffine(shear=0.1)
        ]),
        OneOf([
            GaussNoise(),
            GaussianBlur(),
            MedianBlur(),
            MotionBlur()
        ]),
        OneOf([
            RandomBrightnessContrast(),
            CLAHE(),
            IAASharpen()
        ]),
        Cutout(10, 2, 2, 128)
    ], bbox_params=BboxParams("pascal_voc", label_fields=["category_id"], min_area=0.0, min_visibility=0.5), p=p)


def my_loss(y_true, y_pred):
    hm = y_pred[:, :, :, :1]
    hm_t = y_true[:, :, :, :1]

    sc = y_pred[:, :, :, 1:]

    sc_t = y_true[:, :, :, 1:]

    pos_mask = tf.where(tf.equal(hm_t, 1.0), tf.ones_like(hm_t), tf.zeros_like(hm_t))
    neg_mask = tf.where(tf.less(hm_t, 1.0), tf.ones_like(hm_t), tf.zeros_like(hm_t))

    N = tf.reduce_sum(pos_mask)
    N = tf.clip_by_value(N, 1., 70000.)

    Nn = tf.reduce_sum(neg_mask)
    Nn = tf.clip_by_value(Nn, 1., 70000.)

    lc = tf.square(hm_t - hm)
    pos_loss = lc * pos_mask
    pos_loss = tf.reduce_sum(pos_loss) / N

    neg_loss = lc * neg_mask
    neg_loss = tf.reduce_sum(neg_loss) / Nn

    ls = tf.square(sc_t - sc) * pos_mask
    ls = tf.reduce_sum(ls) / N

    # return pos_loss + 2 * neg_loss + ls
    return (pos_loss + 10 * neg_loss + ls) / 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    train_generator = CSVGenerator('annotations.csv',
                                   320, 480,
                                   2, 'images_3', strong_aug())
    val_generator = CSVGenerator('annotations.csv',
                                 320, 480,
                                 2, 'images_3', None)

    callbacks = [
        ModelCheckpoint(os.path.join('models', 'enet_val_{epoch}.h5'),
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=False,
                        mode='min'),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.5,
                          patience=5,
                          verbose=1,
                          mode='min')
    ]

    model = make_net((320, 480, 3))
    # model.summary()

    model.compile(Nadam(1e-4), loss=my_loss, metrics=['binary_accuracy', 'mae'])

    model.fit(train_generator, validation_data=val_generator, epochs=100,
              verbose=1, callbacks=callbacks)
