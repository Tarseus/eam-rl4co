from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'

COLORS = {
    'one_point': '#8C8C8C',
    'euclidean_two_point': '#E69F00',
    'fisher_two_point': '#009E73',
}
LABELS = {
    'one_point': 'One-point',
    'euclidean_two_point': 'Euclidean two-point',
    'fisher_two_point': 'Fisher two-point',
}
MARKERS = {'one_point': 'o', 'euclidean_two_point': 's', 'fisher_two_point': '^'}


def style() -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.labelsize': 9,
        'legend.fontsize': 7.5,
        'legend.frameon': False,
        'figure.dpi': 160,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.18,
        'grid.linestyle': '-',
        'lines.linewidth': 1.8,
        'lines.markersize': 4.5,
    })


def panel_instance_delta(ax: plt.Axes) -> None:
    raw = pd.read_csv(RESULTS / 'instance_count_raw.csv')
    keys = ['dataset', 'target', 'R', 'subset']
    reference = raw[raw.method == 'one_point'][keys + ['rho']].rename(
        columns={'rho': 'rho_reference'}
    )
    for method in ('euclidean_two_point', 'fisher_two_point'):
        current = raw[raw.method == method].merge(reference, on=keys)
        current['delta'] = current.rho - current.rho_reference
        grouped = current.groupby('R').delta
        x = np.asarray(sorted(current.R.unique()))
        median = grouped.median().reindex(x).to_numpy()
        low = grouped.quantile(0.025).reindex(x).to_numpy()
        high = grouped.quantile(0.975).reindex(x).to_numpy()
        ax.plot(
            x,
            median,
            color=COLORS[method],
            marker=MARKERS[method],
            label=LABELS[method],
        )
        ax.fill_between(x, low, high, color=COLORS[method], alpha=0.13, linewidth=0)
    ax.axhline(0.0, color='#555555', linewidth=0.8, linestyle='--')
    ax.set_xticks([2, 4, 8, 12, 16])
    ax.set_xlabel('Number of task instances, $R$')
    ax.set_ylabel(r'$\Delta\rho$ over one-point')
    ax.set_title('(a)', loc='left')
    ax.legend(loc='upper left')


def panel_coordinate(ax: plt.Axes) -> None:
    raw = pd.read_csv(RESULTS / 'coordinate_stress_raw.csv')
    raw = raw.drop_duplicates(['dataset', 'method', 'kappa', 'permutation'])
    for method in ('one_point', 'euclidean_two_point', 'fisher_two_point'):
        current = raw[raw.method == method]
        grouped = current.groupby('kappa').top1
        x = np.asarray(sorted(current.kappa.unique()))
        median = grouped.median().reindex(x).to_numpy()
        low = grouped.min().reindex(x).to_numpy()
        high = grouped.max().reindex(x).to_numpy()
        ax.plot(
            x,
            median,
            color=COLORS[method],
            marker=MARKERS[method],
            label=LABELS[method],
        )
        ax.fill_between(x, low, high, color=COLORS[method], alpha=0.12, linewidth=0)
    ax.set_xscale('log')
    ax.set_xticks([1, 3, 10, 30], labels=['1', '3', '10', '30'])
    ax.set_ylim(0.78, 1.012)
    ax.set_xlabel(r'Coordinate condition number, $\kappa(A)$')
    ax.set_ylabel('Top-1 neighbor agreement')
    ax.set_title('(b)', loc='left')
    ax.legend(loc='lower left')


def panel_nonuniform(ax: plt.Axes) -> None:
    raw = pd.read_csv(RESULTS / 'nonuniform.csv')
    raw = raw[raw.method == 'one_point'].drop_duplicates(['dataset', 'beta'])
    styles = {
        'matched': ('#0072B2', 'o', 'Matched-40'),
        'external': ('#D55E00', 's', 'External-65'),
    }
    for dataset, (color, marker, label) in styles.items():
        current = raw[raw.dataset == dataset].sort_values('beta')
        ax.plot(
            current.beta,
            current.ef_top1,
            color=color,
            marker=marker,
            label=label,
        )
    ax.set_xticks([0, 0.25, 0.5, 1.0])
    ax.set_ylim(0.78, 1.01)
    ax.set_xlabel(r'Non-uniformity, $\beta$')
    ax.set_ylabel('Euclidean/Fisher top-1 agreement')
    ax.set_title('(c)', loc='left')
    ax.legend(loc='lower right')


def main() -> int:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    panel_instance_delta(axes[0])
    panel_coordinate(axes[1])
    panel_nonuniform(axes[2])
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(RESULTS / 'fig_validation.pdf')
    fig.savefig(RESULTS / 'fig_validation.png', dpi=300)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
