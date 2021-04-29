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
attack_acc = defaultdict(list)
num_models = defaultdict(int)

directories = [x for x in os.listdir(args.dir) if os.path.isdir(args.dir / x)]

for directory in directories:
    model = directory.split('model_class-')[1].split('--')[0]
    coeff = float(directory.split('fair_coeff-')[1].replace('_', '.'))

    attack_directory = args.dir / directory / 'attack'

    for attack_dir in os.listdir(attack_directory):
        if not os.path.isdir(attack_directory / attack_dir):
            continue

        if attack_dir in ['checkpoints', 'npz']:
            continue

        attack_metrics = pd.read_csv(
            attack_directory / attack_dir / 'attack_results.csv', header=None
        )
        attack_metrics = dict(zip(attack_metrics[0], attack_metrics[1]))

        num_models[f'{model}-{coeff}'] += 1
        coeffs[f'{model}-{attack_dir}'].append(coeff)
        attack_acc[f'{model}-{attack_dir}'].append(attack_metrics['accuracy'])

for model_count in num_models.values():
    assert model_count == 12

fig, ax = plt.subplots(3, 4, figsize=(20, 15))
models = list()

for idx, (model, attack_acc_values) in enumerate(attack_acc.items()):
    indices = np.argsort(np.asarray(coeffs[model]))
    model_name, architecture = model.split('-')

    if model_name not in models:
        models.append(model_name)

    j = models.index(model_name)
    i = {'5': 0, '25': 1, '200': 2}[architecture.split('_')[0]]

    ax[i][j].plot(
        np.asarray(coeffs[model])[indices],
        np.asarray(attack_acc[model])[indices], label=architecture
    )
    ax[i][j].set_xlabel(r'$\gamma$')
    ax[i][j].set_ylabel('attack accuracy')
    ax[i][j].set_ylim(0.62, 0.92)
    ax[i][j].title.set_text(model_name)

for i in range(3):
    for j in range(4):
        handles, labels = ax[i][j].get_legend_handles_labels()
        labels, handels = zip(*sorted(zip(labels, handles)))
        ax[i][j].legend(handles, labels, loc='lower right')

fig.tight_layout()
plt.savefig('attack.eps')
plt.close(fig)
