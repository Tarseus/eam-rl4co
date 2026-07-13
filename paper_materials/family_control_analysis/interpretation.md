# Family-Control Interpretation Material

## Core Claim

Family control is not another semantic gate, and it is not merely a way to maximize the number of distinct strings or signatures. A gate rejects an individual objective because its local behavior is invalid; family control changes the evolutionary pressure over the population. It prevents a single high-variance lineage from occupying most parent and elite slots, so that different semantic hypotheses keep competing: cost/rank calibration, advantage normalization, bounded links, aggregation choices, and regularization choices.

## Evidence Summary

### TSP100

With family control, the best TSP100 loss reaches score `-0.02145` at generation `4`. Without family control, the best score is weaker, `-0.01394`, even though the family-disabled run evaluates more high-fidelity candidates (`86` vs `40`).

The late family-disabled population is not simply broader in the useful sense. It may have many distinct signatures, but those signatures increasingly decorate the same nearby motifs: late-generation extra-regularizer usage rises to `91.7%` versus `27.1%` with family control; advantage-heavy formulas rise to `97.9%` versus `60.4%`; probabilistic/rank-prob variants rise to `37.5%` versus `6.2%`. This supports the interpretation that no-family search spends budget on surface novelty and regularized variants rather than on complementary preference hypotheses.

### FFSP100

The FFSP100 contrast is sharper. With family control, the best score is `-0.193` and `22` high-fidelity candidates improve over the baseline. Without family control, the best score is `1.04` and no high-fidelity candidate has negative/improving score. The late family-disabled population concentrates on advantage-blend/tanh-ReLU motifs (`45.8%`), while the family-aware run keeps the simpler logsigmoid advantage-margin backbone dominant (`85.4%` logsigmoid backbone in late generations).

## Suggested Paper Wording

Family control addresses a different failure mode from gate filtering. Gates enforce individual-level validity: a candidate must point gradients in the right preference direction, remain sensitive to the optimization objective, and avoid affine-scale artifacts. Family control operates at the population level. When it is removed, search does not merely explore a larger hypothesis space; it increasingly reuses successful-looking motifs and decorates them with additional regularizers, probabilistic rank terms, or advantage-heavy blends. These variants can be syntactically diverse while remaining semantically correlated, so raw signature count overstates the amount of useful exploration. The result is weaker competition among genuinely different design hypotheses and weaker high-fidelity discoveries, especially on FFSP100 where the family-disabled run finds no improving high-fidelity candidate. Family control is therefore best understood as maintaining hypothesis-level competition, not as adding another correctness check.

## One-Sentence Version

Gate control asks whether an individual objective is valid; family control asks whether the search population is still testing different objective-design hypotheses rather than many decorated descendants of the same motif.
