import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=Path, default='experiments/full_sweep_adult')
args = parser.parse_args()

coeffs = defaultdict(list)
attack_acc_load = defaultdict(list)
attack_acc_init = defaultdict(list)
attack_acc_train = defaultdict(list)

directories = [x for x in os.listdir(args.dir) if os.path.isdir(args.dir / x)]

for directory in directories:
    model = directory.split('model_class-')[1].split('--')[0]
    coeff = float(directory.split('fair_coeff-')[1].replace('_', '.'))

    attack_metrics = pd.read_csv(
        args.dir / directory / 'attack' / 'attack_results.csv', header=None
    )
    attack_metrics = dict(zip(attack_metrics[0], attack_metrics[1]))

    coeffs[model].append(coeff)
    attack_acc_load[model].append(attack_metrics['accuracy_loading'])
    attack_acc_init[model].append(attack_metrics['accuracy_initialization'])
    attack_acc_train[model].append(attack_metrics['accuracy_training'])

fig, ax = plt.subplots(1, 4, figsize=(20, 10))

for idx, (model, coeff_vals) in enumerate(coeffs.items()):
    indices = np.argsort(np.asarray(coeff_vals))

    ax[idx].plot(
        np.asarray(coeff_vals)[indices],
        np.asarray(attack_acc_load[model])[indices], label='loading'
    )
    ax[idx].plot(
        np.asarray(coeff_vals)[indices],
        np.asarray(attack_acc_init[model])[indices], label='initialization'
    )
    ax[idx].plot(
        np.asarray(coeff_vals)[indices],
        np.asarray(attack_acc_train[model])[indices], label='training'
    )
    ax[idx].set_xlabel(r'$\gamma$')
    ax[idx].set_ylabel('attack accuracy')
    ax[idx].set_ylim(0.30, 0.92)
    ax[idx].title.set_text(model)
    ax[idx].axhline(
        1 - (4907 / 15040), label='majority class', c='k', linestyle=':'
    )
    ax[idx].legend()

fig.tight_layout()
plt.savefig('attack_original.eps')
plt.close(fig)
