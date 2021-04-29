import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from codebase import models
from codebase.datasets import Dataset
from codebase.results import ResultLogger
from codebase.tester import Tester
from codebase.utils import get_npz_basename

PROJECT_ROOT = Path(__file__).absolute().parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)


def evaluate(sess, model, data):
    a_true, a_pred = list(), list()

    for x, y, a in data.get_batch_iterator('test', 512):
        a_pred.append(
            np.round(
                sess.run(model.A_hat, feed_dict={model.X: x})
            ).flatten().astype(np.int)
        )
        a_true.append(a.flatten().astype(np.int))

    a_pred = np.concatenate(a_pred, axis=0)
    a_true = np.concatenate(a_true, axis=0)

    return accuracy_score(a_true, a_pred)


def main(args):
    expdname = args['dirs']['exp_dir']
    expname = args['exp_name']
    resdirname = os.path.join(expdname, expname)
    npzfile = os.path.join(
        args['dirs']['data_dir'], args['data']['name'],
        get_npz_basename(**args['data'])
    )

    if args['data']['use_attr']:
        args['model'].update(xdim=args['model']['xdim'] + 1)

    # get dataset
    data = Dataset(
        npzfile=npzfile, **args['data'], batch_size=args['train']['batch_size']
    )

    # get model
    if 'Weighted' in args['model']['class']:
        A_weights = [1. / x for x in data.get_A_proportions()]
        Y_weights = [1. / x for x in data.get_Y_proportions()]
        AY_weights = [[1. / x for x in L] for L in data.get_AY_proportions()]

        if 'Eqopp' in args['model']['class']:
            # we only care about ppl with Y = 0 --- those who didn't get sick
            AY_weights[0][1] = 0.  # AY_weights[0][1]
            AY_weights[1][1] = 0.  # AY_weights[1][1]

        args['model'].update(
            A_weights=A_weights, Y_weights=Y_weights, AY_weights=AY_weights
        )

    model_class = getattr(models, args['model'].pop('class'))
    model = model_class(
        **args['model'], batch_size=args['train']['batch_size']
    )

    y_pos, y_tot = defaultdict(int), defaultdict(int)
    a_pos, a_tot = defaultdict(int), defaultdict(int)

    with tf.Session():
        for split in ['train', 'valid', 'test']:
            for x, y, a in data.get_batch_iterator(split, 1024):
                y_pos[split] += y.sum()
                a_pos[split] += a.sum()
                y_tot[split] += y.shape[0]
                a_tot[split] += a.shape[0]

    print('y_pos:', y_pos)
    print('y_tot:', y_tot)
    print('a_pos:', a_pos)
    print('a_tot:', a_tot)

    with tf.variable_scope('attack'):
        optimizer = tf.train.AdamOptimizer()
        attack_op = optimizer.minimize(
            model.aud_loss, var_list=tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='model/aud'
            ), name='attack_op'
        )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # restore the trained model
        saver = tf.train.Saver(
            var_list=tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='model'
            )
        )
        saver.restore(sess, tf.train.latest_checkpoint(resdirname))

        # test the trained model
        attack_dir = os.path.join(resdirname, 'attack')
        reslogger = ResultLogger(attack_dir)
        tester = Tester(model, data, sess, reslogger)
        tester.evaluate(args['train']['batch_size'])

        accuracy_after_loading = evaluate(sess, model, data)

        sess.run(
            tf.initialize_variables(
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='model/aud'
                )
            )
        )

        # test the trained model
        reslogger = ResultLogger(attack_dir)
        tester = Tester(model, data, sess, reslogger)
        tester.evaluate(args['train']['batch_size'])

        accuracy_after_initialization = evaluate(sess, model, data)

        for epoch_id in range(500):

            a_pred, a_true = list(), list()
            total = total_ce_loss = 0

            for x, y, a in data.get_batch_iterator('train', 512):
                A_hat, ce_loss, _ = sess.run(
                    [model.A_hat, model.aud_loss, attack_op],
                    feed_dict={model.X: x, model.A: a}
                )
                a_pred.append(np.round(A_hat).flatten().astype(np.int))
                a_true.append(a.flatten().astype(np.int))
                total_ce_loss += ce_loss.sum()
                total += a.shape[0]

            a_pred = np.concatenate(a_pred, axis=0)
            a_true = np.concatenate(a_true, axis=0)

            print(
                f'[train] epoch {epoch_id:3d}: '
                f'accuracy={accuracy_score(a_true, a_pred):.3f}, '
                f'loss={total_ce_loss / total:.9f}'
            )

            if epoch_id % 10 == 0:
                print(
                    f'[ test] epoch {epoch_id:3d}: '
                    f'accuracy={evaluate(sess, model, data):.3f}'
                )

        accuracy_after_training = evaluate(sess, model, data)

        print(f'accuracy [loading]        {accuracy_after_loading:.3f}')
        print(f'accuracy [initialization] {accuracy_after_initialization:.3f}')
        print(f'accuracy [training]       {accuracy_after_training:.3f}')

        # test the trained model
        reslogger = ResultLogger(attack_dir)
        tester = Tester(model, data, sess, reslogger)
        tester.evaluate(args['train']['batch_size'])

    with open(os.path.join(attack_dir, 'attack_results.csv'), 'w') as file:
        file.write(f'accuracy_loading,{accuracy_after_loading}\n')
        file.write(
            f'accuracy_initialization,{accuracy_after_initialization}\n'
        )
        file.write(f'accuracy_training,{accuracy_after_training}\n')

    # flush
    tf.reset_default_graph()


if __name__ == '__main__':
    from codebase.config import process_config

    opt = process_config(verbose=False)
    main(opt)
