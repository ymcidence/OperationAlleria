from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as td
import os
import typing

from meta import DATA_PATH
from util import image_processing as ip

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_data(conf):
    set_name = conf.set_name
    dataset = {
        'cifar': load_cifar
    }
    return dataset.get(set_name)(conf)


def load_cifar(conf) -> typing.Dict[str, tf.data.Dataset]:
    """
        load from tfds
        :return: strictly a dict with keys of 'train' and 'test', with each one at least having 'image' and 'label'
        """

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    data = td.load('cifar10', data_dir=DATA_PATH)

    def _mapper(x):
        image = ip.vgg_caffe_style_preprocessing(x['image'])
        label = x['label']
        return {
            'image': image,
            'label': label
        }

    return {
        'train': _batching(data['train'], conf.batch_size, 50000, mapper=_mapper, split='train'),
        'test': _batching(data['test'], conf.batch_size, 10000, mapper=_mapper, split='test')
    }


def _dummy_mapper(x):
    return x


def _batching(data: tf.data.Dataset, batch_size, shuffle_size=10000, mapper=_dummy_mapper, split='train', repeat=False):
    data = data.repeat() if repeat else data
    if split == 'train':
        return data.shuffle(shuffle_size).map(mapper, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(
            buffer_size=AUTOTUNE)
    else:
        return data.map(mapper, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
