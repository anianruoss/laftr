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
dp = defaultdict(list)
eo = defaultdict(list)
eopp = defaultdict(list)
acc = defaultdict(list)

directories = [x for x in os.listdir(args.dir) if os.path.isdir(args.dir / x)]

for directory in directories:
    model = directory.split('model_class-')[1].split('--')[0]
    coeff = float(directory.split('fair_coeff-')[1].replace('_', '.'))

    if coeff == 0:
        model = 'Unfair Baseline'

    test_metrics = pd.read_csv(
        args.dir / directory / 'test_metrics.csv', header=None
    )
    test_metrics = dict(zip(test_metrics[0], test_metrics[1]))

    dp[model].append(test_metrics['DP'])
    eo[model].append(test_metrics['DI'])
    eopp[model].append(test_metrics['DI_FP'])
    acc[model].append(1 - test_metrics['ErrY'])
    coeffs[model].append(coeff)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

for model, dp_values in dp.items():
    indices = np.argsort(np.asarray(dp_values))
    ax[0].plot(
        np.asarray(dp_values)[indices], np.asarray(acc[model])[indices],
        label=model, marker='s' if model == 'Unfair Baseline' else None
    )

for model, eo_values in eo.items():
    indices = np.argsort(np.asarray(eo_values))
    ax[1].plot(
        np.asarray(eo_values)[indices], np.asarray(acc[model])[indices],
        label=model, marker='s' if model == 'Unfair Baseline' else None
    )

for model, eopp_values in eopp.items():
    indices = np.argsort(np.asarray(eopp_values))
    ax[2].plot(
        np.asarray(eopp_values)[indices], np.asarray(acc[model])[indices],
        label=model, marker='s' if model == 'Unfair Baseline' else None
    )

ax[0].set_xlabel(r'$\Delta DP$')
ax[1].set_xlabel(r'$\Delta EO$')
ax[2].set_xlabel(r'$\Delta EO_{pp}$')

for i in range(3):
    ax[i].legend()
    ax[i].set_ylabel('Accuracy')

fig.tight_layout()
plt.savefig('laftr.eps')
plt.close(fig)
