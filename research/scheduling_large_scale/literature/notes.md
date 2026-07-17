# Literature Notes

## MatNet (Kwon et al., 2021)

- FFSP training sizes: 20, 50, and 100 jobs with three stages and four machines per stage.
- The supplement evaluates the trained models zero-shot on FFSP1000.
- Relevance: establishes FFSP1000 as a defensible main scale and supports checkpoint transfer rather than from-scratch FFSP1000 training.
- Source: https://openreview.net/forum?id=C__ChZs8WjU

## BOPO (Liao et al., 2025)

- JSSP training pool contains shapes through 20x20; reported configuration uses B=256, K=16 and 20 epochs.
- Evaluation includes 50x20 and 100x20.
- Relevance: establishes the JSSP scale ladder and an upper training-horizon reference, while the repository's matched pipeline currently uses B=128/K=16.
- Source: https://openreview.net/forum?id=FLy6yXdrlW
