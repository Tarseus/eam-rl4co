from pathlib import Path
import json
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(r'E:/CAS/perfer/pre-finder')
OUT_DIR = ROOT / 'figures' / 'objective_search'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    'Family-aware': ROOT / 'runs' / 'pref_loss_tsp100_discovery' / '20260317-131507',
    'Family-disabled': ROOT / 'runs' / 'pref_loss_tsp100_discovery_family_off' / '20260421-213240',
}
METHOD_COLORS = {
    'Family-aware': '#1d4ed8',
    'Family-disabled': '#d97706',
}
METHOD_MARKERS = {
    'Family-aware': 'o',
    'Family-disabled': 'D',
}
METHOD_JITTER = {
    'Family-aware': -0.055,
    'Family-disabled': 0.055,
}
FAMILY_ORDER = [
    'Cost / Pairwise margin',
    'Rank / Pairwise margin',
    'Rank / Rank-weighted',
    'Rank / Rank-weighted norm',
    'Rank / Rank-weighted scale',
    'Rank / Adv-weighted norm',
    'Rank / Margin clipped',
    'Regret / Pairwise margin',
    'Advantage / Pairwise margin',
    'Advantage / Logistic',
    'Prob / Rank-prob',
    'Other / Pairwise margin',
]
FAMILY_COLORS = {
    'Cost / Pairwise margin': '#7c8798',
    'Rank / Pairwise margin': '#0072b2',
    'Rank / Rank-weighted': '#009e73',
    'Rank / Rank-weighted norm': '#8b5cf6',
    'Rank / Rank-weighted scale': '#56b4e9',
    'Rank / Adv-weighted norm': '#2ca02c',
    'Rank / Margin clipped': '#b79f00',
    'Regret / Pairwise margin': '#cc79a7',
    'Advantage / Pairwise margin': '#e69f00',
    'Advantage / Logistic': '#8c564b',
    'Prob / Rank-prob': '#d55e00',
    'Other / Pairwise margin': '#4b5563',
}
AWARE_BUCKET_COLORS = [
    '#00a6a6',
    '#6a3d9a',
    '#f781bf',
    '#a6761d',
    '#66a61e',
    '#17becf',
    '#e7298a',
    '#1b9e77',
    '#7570b3',
    '#bcbd22',
    '#fb8072',
    '#80b1d3',
    '#b15928',
    '#cab2d6',
    '#33a02c',
    '#ff7f00',
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10.4,
    'axes.titlesize': 12.4,
    'axes.labelsize': 10.8,
    'legend.fontsize': 8.8,
    'xtick.labelsize': 9.6,
    'ytick.labelsize': 9.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 160,
    'savefig.dpi': 300,
})


def finite_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def normalize_value(value):
    if isinstance(value, list):
        value = '+'.join(str(v) for v in value)
    text = str(value or '').strip().lower()
    text = text.replace('-', '_').replace(' ', '_').replace(':', '_')
    return text if text and text not in {'none', '<missing>'} else 'unknown'


def parse_signature(signature):
    out = {}
    for part in str(signature or '').split('|'):
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        out[key.strip()] = normalize_value(value)
    return out


def semantic_family(paradigm='', signal='', link='', constraint='', name=''):
    text = ' '.join(
        normalize_value(v)
        for v in [paradigm, signal, link, constraint, name]
    )
    if 'rankprob' in text or 'rank_prob' in text or 'probabilistic' in text or 'probability' in text or 'bayesian' in text:
        return 'Probabilistic ranking'
    if 'rank' in text:
        return 'Rank-based margin'
    if 'regret' in text:
        return 'Regret margin'
    if 'cost' in text:
        return 'Cost-gap margin'
    if 'advantage' in text:
        return 'Advantage margin'
    if 'zscore' in text or 'normal' in text or 'scale' in text:
        return 'Normalized margin'
    return 'Generic pairwise'


def primary_signal(paradigm='', signal='', link='', constraint='', name=''):
    text = ' '.join(normalize_value(v) for v in [signal, paradigm, name, link, constraint])
    if 'rankprob' in text or 'rank_prob' in text or 'probabilistic' in text or 'probability' in text or 'bayesian' in text:
        return 'Prob'
    if 'rank' in text:
        return 'Rank'
    if 'regret' in text:
        return 'Regret'
    if 'cost' in text:
        return 'Cost'
    if 'advantage' in text:
        return 'Advantage'
    return 'Other'


def paradigm_variant(paradigm=''):
    raw = normalize_value(paradigm)
    if 'rankprob' in raw or 'rank_prob' in raw:
        return 'Rank-prob'
    if 'logsigmoid' in raw or 'logistic' in raw:
        return 'Logistic'
    if 'advantage_weighted' in raw:
        return 'Adv-weighted norm' if 'normal' in raw else 'Adv-weighted'
    if 'rank_weight' in raw or 'ranked' in raw or 'rank_gap' in raw:
        if 'scale' in raw:
            return 'Rank-weighted scale'
        if 'normal' in raw:
            return 'Rank-weighted norm'
        return 'Rank-weighted'
    if 'with_clipping' in raw or 'clipping' in raw:
        return 'Margin clipped'
    if 'pairwise_margin' in raw or 'pairwise_comparison' in raw:
        return 'Pairwise margin'
    return raw.replace('pairwise_', '').replace('_', ' ').title()


def signal_paradigm_family(paradigm='', signal='', link='', constraint='', name=''):
    return f'{primary_signal(paradigm, signal, link, constraint, name)} / {paradigm_variant(paradigm)}'


def family_from_ir(ir):
    hp = (ir or {}).get('hyperparams') or {}
    return signal_paradigm_family(
        hp.get('paradigm_family'),
        hp.get('signal_family'),
        hp.get('link_family'),
        hp.get('constraint_family'),
        (ir or {}).get('name'),
    )


def load_loss_meta_by_id(run_dir):
    out = {}
    with (run_dir / 'losses.jsonl').open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            fid = row.get('id')
            hp = (row.get('ir') or {}).get('hyperparams') or {}
            sig = parse_signature(row.get('family_signature') or row.get('family') or '')
            signature = row.get('family_signature') or row.get('family') or fid
            out[fid] = {
                'family': signal_paradigm_family(
                    hp.get('paradigm_family') or sig.get('paradigm'),
                    hp.get('signal_family') or sig.get('signal'),
                    hp.get('link_family') or sig.get('link'),
                    hp.get('constraint_family') or sig.get('constraint'),
                    (row.get('ir') or {}).get('name') or row.get('name'),
                ),
                'bucket_key': signature,
            }
    return out


def load_loss_rows(run_dir):
    rows = []
    with (run_dir / 'losses.jsonl').open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            hp = (row.get('ir') or {}).get('hyperparams') or {}
            sig = parse_signature(row.get('family_signature') or row.get('family') or '')
            rows.append({
                'generation': int(row.get('generation', -1)),
                'family': signal_paradigm_family(
                    hp.get('paradigm_family') or sig.get('paradigm'),
                    hp.get('signal_family') or sig.get('signal'),
                    hp.get('link_family') or sig.get('link'),
                    hp.get('constraint_family') or sig.get('constraint'),
                    (row.get('ir') or {}).get('name') or row.get('name'),
                ),
            })
    return [r for r in rows if r['generation'] >= 0]


def row_family(row, meta_by_id):
    fid = row.get('f_id') or row.get('f_id_after_repair') or row.get('f_id_before_repair')
    if fid in meta_by_id:
        return meta_by_id[fid]['family']
    return family_from_ir(row.get('f_ir'))


def row_bucket_key(row, meta_by_id):
    fid = row.get('f_id') or row.get('f_id_after_repair') or row.get('f_id_before_repair')
    if fid in meta_by_id:
        return meta_by_id[fid]['bucket_key']
    return fid or family_from_ir(row.get('f_ir'))


def load_pairs(run_dir, meta_by_id):
    rows = []
    with (run_dir / 'pairs.jsonl').open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            score = finite_float(row.get('score'))
            gen = row.get('generation')
            if score is None or gen is None:
                continue
            stage = row.get('stage_final') or row.get('stage') or ''
            if stage != 'high_fidelity':
                continue
            fid = row.get('f_id') or row.get('f_id_after_repair') or row.get('f_id_before_repair')
            rows.append({
                'generation': int(gen),
                'score': score,
                'improvement': -score,
                'id': fid,
                'family': row_family(row, meta_by_id),
                'bucket_key': row_bucket_key(row, meta_by_id),
            })
    return rows


def load_best(run_dir, meta_by_id):
    with (run_dir / 'best_loss.json').open('r', encoding='utf-8') as f:
        data = json.load(f)
    score = finite_float(data.get('score'))
    fid = data.get('id', '')
    generation = data.get('generation')
    if generation is None:
        generation = data.get('best_pair_generation', -1)
    return {
        'score': score,
        'improvement': -score if score is not None else None,
        'generation': int(generation),
        'id': fid,
        'name': data.get('name', ''),
        'family': meta_by_id.get(fid, {}).get('family') or family_from_ir(data.get('ir')),
        'bucket_key': meta_by_id.get(fid, {}).get('bucket_key') or fid,
    }


def best_so_far(rows, max_gen):
    by_gen = {g: [] for g in range(max_gen + 1)}
    for r in rows:
        if 0 <= r['generation'] <= max_gen:
            by_gen[r['generation']].append(r['improvement'])

    xs = [0]
    ys = [0.0]
    current = 0.0
    for g in range(1, max_gen + 1):
        vals = by_gen[g]
        if vals:
            current = max(current, max(vals))
        xs.append(g)
        ys.append(current)
    return np.array(xs), np.array(ys)


def control_view_family(label, row):
    return row['family']


def main():
    records = {}
    max_gen = 0
    for label, run_dir in RUNS.items():
        meta_by_id = load_loss_meta_by_id(run_dir)
        all_losses = load_loss_rows(run_dir)
        pairs = load_pairs(run_dir, meta_by_id)
        best = load_best(run_dir, meta_by_id)
        best_already_in_pairs = any(
            r['generation'] == best['generation']
            and r.get('id') == best['id']
            and abs(r['score'] - best['score']) < 1e-12
            for r in pairs
            if best['score'] is not None
        )
        if best['score'] is not None and best['generation'] >= 0 and not best_already_in_pairs:
            pairs.append({
                'generation': best['generation'],
                'score': best['score'],
                'improvement': best['improvement'],
                'id': best['id'],
                'family': best['family'],
                'bucket_key': best['bucket_key'],
            })
        max_gen = max(
            max_gen,
            best['generation'],
            max([r['generation'] for r in pairs], default=0),
            max([r['generation'] for r in all_losses], default=0),
        )
        records[label] = {'pairs': pairs, 'all_losses': all_losses, 'best': best}

    present_families = {
        control_view_family(label, r)
        for label, rec in records.items()
        for r in rec['pairs']
        if r['generation'] > 0 and r['improvement'] > 0
    }
    plot_family_order = [fam for fam in FAMILY_ORDER if fam in present_families]
    overflow_families = sorted(fam for fam in present_families if fam not in FAMILY_COLORS)
    plot_family_order.extend(overflow_families)
    plot_family_colors = dict(FAMILY_COLORS)
    for idx, fam in enumerate(overflow_families):
        plot_family_colors[fam] = AWARE_BUCKET_COLORS[idx % len(AWARE_BUCKET_COLORS)]

    fig, ax = plt.subplots(figsize=(8.9, 4.95))
    fig.patch.set_facecolor('#fbfaf7')
    ax.set_facecolor('#fbfaf7')
    ax.grid(axis='y', color='#d8d2c4', lw=0.72, alpha=0.58)
    ax.grid(axis='x', color='#e8e1d3', lw=0.52, alpha=0.32)

    rng = np.random.default_rng(11)
    method_handles = []
    for label, rec in records.items():
        method_color = METHOD_COLORS[label]
        marker = METHOD_MARKERS[label]

        pairs = rec['pairs']
        display_pairs = [r for r in pairs if r['generation'] > 0 and r['improvement'] > 0]
        if display_pairs:
            x = np.array([r['generation'] for r in display_pairs], dtype=float)
            y = np.array([r['improvement'] for r in display_pairs], dtype=float)
            jitter = METHOD_JITTER[label] + rng.normal(0, 0.032, size=len(x))
            for family in plot_family_order:
                idx = np.array([control_view_family(label, r) == family for r in display_pairs], dtype=bool)
                if not np.any(idx):
                    continue
                ax.scatter(
                    x[idx] + jitter[idx],
                    y[idx],
                    s=52 if label == 'Family-aware' else 38,
                    marker=marker,
                    facecolor=plot_family_colors[family],
                    edgecolor=method_color,
                    linewidth=0.68,
                    alpha=0.78 if label == 'Family-aware' else 0.42,
                    zorder=6,
                )

        xs, ys = best_so_far(pairs, max_gen)
        ax.step(xs, ys, where='post', color='white', lw=6.2, alpha=0.94, zorder=2, solid_capstyle='round')
        line = ax.step(
            xs,
            ys,
            where='post',
            color=method_color,
            lw=2.75,
            marker=marker,
            markersize=4.9,
            markerfacecolor='white',
            markeredgewidth=1.35,
            label=label,
            zorder=3,
            solid_capstyle='round',
        )[0]
        method_handles.append(line)

        best = rec['best']
        if best['improvement'] is not None and best['generation'] >= 0:
            best_family = control_view_family(label, best)
            ax.scatter(
                [best['generation']],
                [best['improvement']],
                s=92,
                marker=marker,
                facecolor=plot_family_colors.get(best_family, '#d6d3d1'),
                edgecolor=method_color,
                linewidth=1.5,
                zorder=7,
            )

    ax.axhline(0, color='#2d2a24', lw=0.85, alpha=0.58)
    ax.set_title('TSP100 family ablation', loc='left', fontweight='bold', pad=7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Discovery score')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.45, max_gen + 0.45)
    best_values = [rec['best']['improvement'] for rec in records.values() if rec['best']['improvement'] is not None]
    upper = max(best_values + [0.001]) * 1.15
    ax.set_ylim(-0.00085, upper)

    method_legend = ax.legend(
        handles=method_handles,
        loc='upper left',
        frameon=True,
        facecolor='#fbfaf7',
        edgecolor='#d8d2c4',
        title='Best-so-far',
    )
    ax.add_artist(method_legend)

    family_handles = [
        Patch(facecolor=plot_family_colors[fam], edgecolor='none', label=fam)
        for fam in plot_family_order
    ]
    ax.legend(
        handles=family_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.17),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.0,
        fontsize=7.5,
    )

    for label, rec in records.items():
        positive_families = len({
            control_view_family(label, r)
            for r in rec['pairs']
            if r['generation'] > 0 and r['improvement'] > 0
        })
        positive_count = sum(1 for r in rec['pairs'] if r['generation'] > 0 and r['improvement'] > 0)
        all_families = len({r['family'] for r in rec['all_losses']})
        best = rec['best']
        print(
            f'{label}: best={best["improvement"]:.6f} gen={best["generation"]} '
            f'positive_candidates={positive_count} positive_families={positive_families} generated_families={all_families}'
        )

    fig.tight_layout(rect=(0.02, 0.11, 0.995, 0.98))

    out = OUT_DIR / '04_tsp_family_ablation.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
