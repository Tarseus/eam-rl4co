from pathlib import Path
import json
import math
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

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
FAMILY_ORDER = [
    'Pairwise margin',
    'Rank-weighted',
    'Regret/advantage',
    'Probabilistic ranking',
    'Preference/ranking',
    'Regression alignment',
    'Logsigmoid logistic',
    'Weighted margin',
    'Structural regularized',
    'Incomplete signature',
]
FAMILY_COLORS = {
    'Pairwise margin': '#8fa0b5',
    'Rank-weighted': '#3b82f6',
    'Regret/advantage': '#14b8a6',
    'Probabilistic ranking': '#f97316',
    'Preference/ranking': '#facc15',
    'Regression alignment': '#a855f7',
    'Logsigmoid logistic': '#64748b',
    'Weighted margin': '#f59e0b',
    'Structural regularized': '#ec4899',
    'Incomplete signature': '#d6d3d1',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10.2,
    'axes.titlesize': 12.0,
    'axes.labelsize': 10.8,
    'legend.fontsize': 8.8,
    'xtick.labelsize': 9.6,
    'ytick.labelsize': 9.3,
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

def canonical_paradigm(raw_value):
    raw = normalize_value(raw_value)
    if raw in {'unknown', ''}:
        return 'Incomplete signature'
    if 'margin_rank' in raw or 'rank_weight' in raw or 'ranked_margin' in raw or 'rank_based' in raw or 'rank_gap' in raw or 'rank_blend' in raw or 'rank_bandpass' in raw or 'discounted_topk' in raw or 'ranking_correlation' in raw or 'ranksigmoid' in raw:
        return 'Rank-weighted'
    if 'rankprob' in raw or 'rank_prob' in raw or 'probabilistic' in raw or 'probability' in raw or 'bayesian' in raw:
        return 'Probabilistic ranking'
    if 'regret' in raw or 'advantage_weighted' in raw or 'margin_advantage' in raw:
        return 'Regret/advantage'
    if 'regression' in raw or 'zscore' in raw or 'relative_difference' in raw:
        return 'Regression alignment'
    if raw in {'ranking', 'pairwise_ranking', 'preference'} or 'preference' in raw:
        return 'Preference/ranking'
    if 'logsigmoid' in raw or 'logistic' in raw or 'sigmoid_reg' in raw:
        return 'Logsigmoid logistic'
    if 'weighted_margin' in raw or 'margin_weighted' in raw or 'cost_weighted' in raw or 'bandpass_weighted_margin' in raw:
        return 'Weighted margin'
    if raw == 'pairwise_margin' or raw.endswith('_margin') or 'pairwise_margin' in raw:
        return 'Pairwise margin'
    return 'Structural regularized'

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
            paradigm = hp.get('paradigm_family') or sig.get('paradigm')
            out[fid] = {
                'family': canonical_paradigm(paradigm),
                'generation': int(row.get('generation', -1)),
                'parents': list(row.get('parents') or []),
                'origin': row.get('origin_base') or row.get('origin') or '',
            }
    return out

def load_candidate_rows(run_dir):
    meta_by_id = load_loss_meta_by_id(run_dir)
    rows = []
    with (run_dir / 'pairs.jsonl').open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            gen = row.get('generation')
            fid = row.get('f_id') or row.get('f_id_after_repair') or row.get('f_id_before_repair')
            if gen is None or fid not in meta_by_id:
                continue
            score = finite_float(row.get('score'))
            stage = row.get('stage_final') or row.get('stage') or ''
            rows.append({
                'generation': int(gen),
                'id': fid,
                'family': meta_by_id[fid]['family'],
                'parents': meta_by_id[fid]['parents'],
                'stage': stage,
                'score': score,
                'improvement': -score if score is not None else None,
            })
    return rows, meta_by_id

def summarize(rows):
    max_gen = max(r['generation'] for r in rows)
    by_gen = defaultdict(list)
    for row in rows:
        by_gen[row['generation']].append(row)
    counts_by_gen = []
    shares_by_gen = []
    dominance = []
    active = []
    best = {}
    for g in range(max_gen + 1):
        counter = Counter(r['family'] for r in by_gen[g])
        total = sum(counter.values())
        counts_by_gen.append(counter)
        if total > 0:
            shares = {fam: counter.get(fam, 0) / total for fam in FAMILY_ORDER}
            dominance.append(max(shares.values()))
            active.append(sum(1 for v in counter.values() if v > 0))
        else:
            shares = {fam: 0.0 for fam in FAMILY_ORDER}
            dominance.append(np.nan)
            active.append(0)
        shares_by_gen.append(shares)
        scored = [r for r in by_gen[g] if r['improvement'] is not None]
        if scored:
            best[g] = max(scored, key=lambda r: r['improvement'])
    return {
        'max_gen': max_gen,
        'counts_by_gen': counts_by_gen,
        'shares_by_gen': shares_by_gen,
        'dominance': dominance,
        'active': active,
        'best': best,
    }

records = {}
for name, path in RUNS.items():
    rows, meta_by_id = load_candidate_rows(path)
    records[name] = {'rows': rows, 'meta_by_id': meta_by_id}
    records[name].update(summarize(rows))

max_gen = max(rec['max_gen'] for rec in records.values())
display_gens = np.arange(4, max_gen + 1)
xs = display_gens

fig = plt.figure(figsize=(10.6, 7.8), constrained_layout=False)
gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.15, 0.64], hspace=0.44)
axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
fig.patch.set_facecolor('#fbfaf7')

for ax, (name, rec) in zip(axes[:2], records.items()):
    method_color = METHOD_COLORS[name]
    ax.set_facecolor('#fbfaf7')
    ax.grid(axis='y', color='#e8e1d6', lw=0.65, alpha=0.55)
    ax.grid(axis='x', color='#eee8df', lw=0.45, alpha=0.35)
    bottom = np.zeros(len(xs))
    for fam in FAMILY_ORDER:
        heights = np.array([rec['shares_by_gen'][int(g)].get(fam, 0.0) for g in xs], dtype=float)
        ax.bar(
            xs,
            heights,
            bottom=bottom,
            width=0.66,
            color=FAMILY_COLORS[fam],
            edgecolor='#fbfaf7',
            linewidth=0.45,
            alpha=0.80,
            zorder=2,
        )
        bottom += heights

    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel('Candidate\nfamily share')
    ax.set_title(name, loc='left', color=method_color, fontweight='bold', pad=7)
    ax.set_xlim(float(display_gens[0]) - 0.55, max_gen + 0.55)
    ax.set_xticks(xs)
    if ax is axes[0]:
        ax.set_xticklabels([])

ax_div = axes[2]
ax_div.set_facecolor('#fbfaf7')
ax_div.grid(axis='y', color='#e8e1d6', lw=0.65, alpha=0.65)
ax_div.grid(axis='x', color='#eee8df', lw=0.45, alpha=0.35)
for name, rec in records.items():
    ax_div.plot(xs, [rec['dominance'][int(g)] for g in xs], color=METHOD_COLORS[name], lw=2.4, marker='o', markersize=5.2, label=name)
ax_div.set_xlim(float(display_gens[0]) - 0.55, max_gen + 0.55)
ax_div.set_ylim(0.0, 1.05)
ax_div.set_xticks(xs)
ax_div.set_ylabel('Dominant family\nshare')
ax_div.yaxis.set_major_formatter(PercentFormatter(1.0))
ax_div.set_title('Dominant family share over generations', loc='left', fontweight='bold', fontsize=11.5, pad=7)
ax_div.legend(loc='upper left', frameon=True, facecolor='#fbfaf7', edgecolor='#d8d2c4')

family_handles = [Patch(facecolor=FAMILY_COLORS[fam], edgecolor='none', label=fam) for fam in FAMILY_ORDER]
fig.legend(
    handles=family_handles,
    loc='lower center',
    ncol=4,
    frameon=True,
    facecolor='#fbfaf7',
    edgecolor='#d8d2c4',
    bbox_to_anchor=(0.5, 0.035),
    fontsize=8.4,
    handlelength=1.25,
    handletextpad=0.45,
    columnspacing=0.95,
)

fig.suptitle('TSP100 family ablation: family control prevents late-stage family collapse', fontsize=13.8, fontweight='bold', y=0.99)
fig.subplots_adjust(left=0.12, right=0.985, top=0.89, bottom=0.245, hspace=0.48)

out = OUT_DIR / '05_tsp_family_diversity_timeline.png'
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(out)
for name, rec in records.items():
    print(name)
    print('active_families_display', [rec['active'][int(g)] for g in display_gens])
    print('dominance_display', [round(rec['dominance'][int(g)], 3) for g in display_gens])
    for g, counter in enumerate(rec['counts_by_gen']):
        print(g, dict(counter))
