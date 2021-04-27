import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)

from codebase import models
from codebase.datasets import Dataset
from codebase.mlp import MLP
from codebase.results import ResultLogger
from codebase.tester import Tester
from codebase.utils import get_npz_basename

PROJECT_ROOT = Path(__file__).absolute().parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)


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

    # prediction attack
    alpha = 0.0
    beta = 1e-5
    activation = 'relu+dropout'
    sensitive = tf.placeholder(
        name='sensitive', shape=(None, 1), dtype=tf.float32
    )
    keep_prob = 0.875
    dropout_keep_prob = tf.placeholder_with_default(keep_prob, shape=())
    with tf.variable_scope('predict_sensitive'):
        mlp = MLP(
            name='latents_to_sensitive_logits',
            shapes=[model.zdim] + args['hidden_layers_attack'] + [model.adim],
            activ=activation, keep_prob=dropout_keep_prob
        )
    sensitive_logits = mlp.forward(model._get_latents(model.X, reuse=True))
    sensitive_pred = tf.cast(tf.greater_equal(sensitive_logits, 0), tf.int32)
    l1 = tf.add_n([
        tf.reduce_sum(tf.abs(weight))
        for weights in mlp.weights.values() for weight in weights.values()
    ])
    l2 = tf.add_n([
        tf.reduce_sum(tf.square(weight))
        for weights in mlp.weights.values() for weight in weights.values()
    ])
    cross_entropy = tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(
            sensitive, sensitive_logits,
            pos_weight=a_tot['train'] / (2 * a_pos['train'])
        )
    )
    loss = cross_entropy + alpha * l1 + beta * l2
    optimizer = tf.train.AdamOptimizer()
    attack_op = optimizer.minimize(
        loss, var_list=tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='predict_sensitive'
        )
    )
    metrics = {
        'l1': tf.Variable(0., name='l1'), 'l2': tf.Variable(0., name='l2'),
        'cross_entropy': tf.Variable(0., name='cross_entropy'),
        'loss': tf.Variable(0., name='loss'), 'f1': tf.Variable(0., name='f1'),
        'precision': tf.Variable(0., name='precision'),
        'recall': tf.Variable(0., name='recall'),
        'accuracy': tf.Variable(0., name='accuracy'),
        'balanced_accuracy': tf.Variable(0., name='balanced_accuracy'),
        'majority_class_accuracy': tf.Variable(
            1. - a_pos[split] / a_tot[split], name='majority_class_accuracy'
        )
    }
    summaries = {
        split: tf.summary.merge(
            [
                tf.summary.scalar(
                    f'{key}/{split}', values
                ) for key, values in metrics.items()
            ] + [
                tf.summary.histogram(
                    f'{key}/{split}', values
                ) for key, values in metrics.items()
            ]
        ) for split in ['train', 'valid', 'test']
    }

    model_name = model_class.__name__
    architecture = '_'.join(map(str, args['hidden_layers_attack']))
    reg_weight = f'l1_weight_{alpha}_l2_weight_{beta}'
    current_date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    dropout = f'dropout_{keep_prob}' if 'dropout' in activation else ''
    writer = tf.summary.FileWriter(logdir=str(
        LOG_DIR / model_name / architecture / activation / dropout /
        reg_weight / current_date
    ))

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
        attack_dir = os.path.join(resdirname, 'attack', architecture)
        reslogger = ResultLogger(attack_dir)
        tester = Tester(model, data, sess, reslogger)
        tester.evaluate(args['train']['batch_size'])

        # initialize output bias to account for label imbalance
        sess.run(
            mlp.weights[len(mlp.weights) - 1]['b'].assign(
                np.log([a_pos['train'] / (a_tot['train'] - a_pos['train'])])
            )
        )

        # run prediction attack
        num_epochs = 500
        batch_size = 512

        for epoch_id in range(num_epochs):
            iterators = {
                'train': data.get_batch_iterator('train', batch_size),
                'valid': data.get_batch_iterator('valid', batch_size),
                'test': data.get_batch_iterator('test', batch_size)
            }

            for split, iterator in iterators.items():

                y_true, y_pred = list(), list()
                tot_loss = tot_cross_entropy = tot_l1 = tot_l2 = total = 0

                for batch_id, (x, y, a) in enumerate(iterator):

                    # hack for WGAN-GP training
                    if len(x) < batch_size:
                        continue

                    if split == 'train':
                        sess.run(
                            attack_op, feed_dict={model.X: x, sensitive: a}
                        )

                    sensitive_pred_, cross_entropy_, l1_, l2_, loss_ = sess.run(
                        [sensitive_pred, cross_entropy, l1, l2, loss],
                        feed_dict={
                            model.X: x, sensitive: a, dropout_keep_prob: 1.0
                        }
                    )
                    y_pred.append(sensitive_pred_.flatten())
                    y_true.append(a.flatten().astype(np.int))
                    total += a.shape[0]
                    tot_l1 += l1_ * a.shape[0]
                    tot_l2 += l2_ * a.shape[0]
                    tot_loss += loss_ * a.shape[0]
                    tot_cross_entropy += cross_entropy_ * a.shape[0]

                y_true = np.concatenate(y_true, axis=0)
                y_pred = np.concatenate(y_pred, axis=0)

                sess.run([
                    metrics['l1'].assign(tot_l1 / total),
                    metrics['l2'].assign(tot_l2 / total),
                    metrics['cross_entropy'].assign(tot_cross_entropy / total),
                    metrics['loss'].assign(tot_loss / total),
                    metrics['accuracy'].assign(accuracy_score(y_true, y_pred)),
                    metrics['balanced_accuracy'].assign(
                        balanced_accuracy_score(y_true, y_pred)
                    ),
                    metrics['precision'].assign(
                        precision_score(y_true, y_pred)
                    ),
                    metrics['recall'].assign(recall_score(y_true, y_pred)),
                    metrics['f1'].assign(f1_score(y_true, y_pred))
                ])

                print(
                    f'[{split.rjust(5, " ")}] epoch {epoch_id:03}: ' + (
                        f', '.join([
                            f'{key}=' + (
                                f'{value.eval():.3e}' if key in [
                                    "l1", "l2", "cross_entropy", "loss"
                                ] else f'{value.eval():.3f}'
                            ) for key, value in metrics.items()
                        ])
                    )
                )
                writer.add_summary(sess.run(summaries[split]), epoch_id)

        with open(os.path.join(attack_dir, 'attack_results.csv'), 'w') as file:
            for key, value in metrics.items():
                file.write(f'{key},{value.eval()}\n')

    writer.flush()
    writer.close()

    # flush
    tf.reset_default_graph()


if __name__ == '__main__':
    from codebase.config import process_config

    opt = process_config(verbose=False)
    main(opt)
