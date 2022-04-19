from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

import tensorflow_hub as th


@tf.custom_gradient
def quantization(x):
    rslt = tf.cast(tf.sign(x), tf.float32)

    def grad(d_q):
        return d_q

    return rslt, grad


class GreedyHash(keras.Model):
    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = conf
        self.backbone = th.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5",
                                      trainable=False)

        self.hash_layer = keras.layers.Dense(conf.code_length)
        self.cls_layer = keras.layers.Dense(conf.cls_num, use_bias=False)
        # self.bn = keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None, step=-1):
        x = inputs['image']
        l = inputs['label']

        # noinspection PyCallingNonCallable
        x = self.backbone(x, training=training)
        cont = self.hash_layer(x, training=training)
        # cont = self.bn(cont, training=training)
        binary = tf.cast(tf.sign(cont), tf.float32)
        binary_ste = quantization(cont)

        pred = self.cls_layer(binary_ste)

        if training:
            commitment_loss = tf.reduce_mean(tf.pow(tf.stop_gradient(binary) - cont, 2)) / 2.
            cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=l, logits=pred))

            loss = cls_loss + .1 * commitment_loss

            self.add_loss(loss)
            if step >= 0:
                tf.summary.scalar('loss/commitment', commitment_loss, step=step)
                tf.summary.scalar('loss/cls', cls_loss, step=step)
                tf.summary.scalar('loss/total', loss, step=step)

        return binary, cont
