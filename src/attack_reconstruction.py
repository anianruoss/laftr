import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from codebase import models
from codebase.datasets import Dataset
from codebase.results import ResultLogger
from codebase.tester import Tester
from codebase.utils import get_npz_basename


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

    # restore the model
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(resdirname))

    a_pos = defaultdict(int)
    a_tot = defaultdict(int)

    with tf.Session():
        for split in ['train', 'valid', 'test']:
            for x, y, a in data.get_batch_iterator(split, 1024):
                a_pos[split] += a.sum()
                a_tot[split] += a.shape[0]

    print(a_pos)
    print(a_tot)

    # reconstruction attack
    batch_size = 512
    non_sensitive = tf.get_variable(
        name='non_sensitive', shape=(batch_size, model.xdim - 1)
    )
    sensitive = tf.placeholder(
        name='sensitive', shape=(batch_size, 1), dtype=tf.float32
    )
    reconstructed = tf.concat(
        [non_sensitive, sensitive], axis=1, name='reconstructed'
    )
    zero_init = non_sensitive.assign(tf.zeros_like(non_sensitive))
    latent_from_reconstructed = model._get_latents(
        reconstructed, reuse=True
    )
    reconstruction_sse = tf.reduce_sum(
        tf.square(model.Z - latent_from_reconstructed), axis=1
    )
    reconstruction_mse = tf.reduce_mean(reconstruction_sse)
    reconstruction_loss = reconstruction_mse
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    attack_op = optimizer.minimize(
        reconstruction_loss, var_list=[non_sensitive]
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # test the trained model
        reslogger = ResultLogger(os.path.join(resdirname, 'attack'))
        tester = Tester(model, data, sess, reslogger)
        tester.evaluate(args['train']['batch_size'])

        # run reconstruction attack
        a_pred = list()
        a_true = list()
        num_steps = 1000
        total = correct = 0
        test_iter = data.get_batch_iterator('test', batch_size)

        for batch_id, (x, y, a) in enumerate(test_iter):

            # hack for WGAN-GP training; don't process weird-sized batches
            if len(x) < batch_size:
                continue

            # ensure that last feature is protected attribute
            assert tf.reduce_all(tf.equal(x[:, -1], a.flatten())).eval()

            # reconstruction attack for sensitive = 0
            sess.run(zero_init)
            feed_dict_zeros = {model.X: x, sensitive: np.zeros_like(a)}

            for _ in range(num_steps):
                sess.run(attack_op, feed_dict=feed_dict_zeros)

            sse_zeros = sess.run(reconstruction_sse, feed_dict=feed_dict_zeros)

            # reconstruction attack for sensitive = 1
            sess.run(zero_init)
            feed_dict_ones = {model.X: x, sensitive: np.ones_like(a)}

            for _ in range(num_steps):
                sess.run(attack_op, feed_dict=feed_dict_ones)

            sse_ones = sess.run(reconstruction_sse, feed_dict=feed_dict_ones)

            a_pred.append(np.argmin([sse_zeros, sse_ones], axis=0))
            a_true.append(a.flatten().astype(int))

            correct += np.sum(a_pred[-1] == a_true[-1])
            total += a.shape[0]

    print('reconstruction accuracy ', correct / total)

    a_pred = np.concatenate(a_pred, axis=0)
    a_true = np.concatenate(a_true, axis=0)

    fig, ax = plt.subplots(1, 3)
    ax[0].hist(a_pred[a_true == 0])
    ax[1].hist(a_pred[a_true == 1])
    ax[2].hist(a_true)
    ax[0].set_xlabel('a_pred_neg')
    ax[1].set_xlabel('a_pred_pos')
    ax[2].set_xlabel('a_true')
    fig.legend()
    fig.tight_layout()
    plt.savefig('plot.eps')

    # flush
    tf.reset_default_graph()


if __name__ == '__main__':
    from codebase.config import process_config

    opt = process_config(verbose=False)
    main(opt)
