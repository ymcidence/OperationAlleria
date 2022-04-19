from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

VGG_MEAN = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)


def vgg_caffe_style_preprocessing(x: tf.Tensor, shape=(224, 224)):
    x = tf.cast(x[..., ::-1], dtype=tf.float32)
    x = x - VGG_MEAN
    x = tf.image.resize(x, shape)

    return x


def vgg_caffe_style_postprocessing(x):
    x = x + VGG_MEAN
    x = x[..., ::-1]
    return tf.clip_by_value(x, clip_value_min=0, clip_value_max=255)
