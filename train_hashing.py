from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from tensorflow import keras
import typing
from time import gmtime, strftime

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

from meta import ROOT_PATH
from model.greedy_hash import GreedyHash as Model
from util import eval_tools, data_loader
from config import parser

os.environ["TFHUB_CACHE_DIR"] = os.path.join(ROOT_PATH, 'data', 'pretrained')


def train_step(batch: dict, model: keras.Model, opt: keras.optimizers.Optimizer, step):
    _step = -1 if step % 100 > 0 else step

    with tf.GradientTape() as tape:
        binary, cont = model(batch, training=True, step=_step)
        loss = model.losses[0]

        gradient = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradient, model.trainable_variables))

    if _step >= 0:
        code = (binary + 1) / 2.
        code = code.numpy()
        label = tf.one_hot(batch['label'], model.conf.cls_num, dtype=tf.float32).numpy()
        batch_map = eval_tools.eval_cls_map(code, code, label, label)

        tf.summary.scalar('train/batch_map', batch_map, step=_step)


def main():
    conf = parser.parse_args()
    conf.cls_num = 10
    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())

    result_path = os.path.join(ROOT_PATH, 'result', conf.set_name + '_greedy')
    task_name = conf.task_name
    save_path = os.path.join(result_path, 'model', task_name + '_' + time_string)
    summary_path = os.path.join(result_path, 'log', task_name + '_' + time_string)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = Model(conf)
    data = data_loader.load_data(conf)

    opt = tf.keras.optimizers.Adam(5e-4)

    if conf.restore != '':
        restore_checkpoint = tf.train.Checkpoint(actor_opt=opt, model=model)
        restore_checkpoint.restore(conf.restore)
        print('Restored from {}'.format(conf.restore))
        starter = 1
    else:
        starter = 0
    writer = tf.summary.create_file_writer(summary_path)
    # checkpoint = tf.train.Checkpoint(actor_opt=opt, model=model)

    with writer.as_default():
        step = 0
        for epoch in range(conf.max_epoch):

            for i, batch in enumerate(data['train']):
                if (step + i) % 50 == 0:
                    print('epoch: {}, step: {}'.format(epoch, step + i))
                train_step(batch, model, opt, step + i)

            step = step + i
            model.save(os.path.join(save_path, '{}_{}'.format(epoch, step)))


if __name__ == '__main__':
    main()
