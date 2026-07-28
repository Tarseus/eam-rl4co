# Fisher-invariant validation results

Protocol: `../protocol.md`. All descriptors use ell = 0.03.

## Numerical checks

- Helmert orthonormal error: 2.220e-16
- Helmert centering error: 1.665e-15
- Maximum Fisher arc error: 4.545e-14
- matched_identity_distance_error: 2.331e-15
- external_identity_distance_error: 2.220e-15

## A. Instance-count sweep (median across subsets)

| Data | Target | Method | R | rho | NN median | False skip |
|---|---|---|---:|---:|---:|---:|
| matched | scratch | one_point | 2 | 0.802 | 0.00659 | 0.148 |
| matched | scratch | one_point | 4 | 0.800 | 0.00659 | 0.146 |
| matched | scratch | one_point | 8 | 0.803 | 0.00659 | 0.143 |
| matched | scratch | one_point | 12 | 0.802 | 0.00659 | 0.145 |
| matched | scratch | one_point | 16 | 0.803 | 0.00659 | 0.141 |
| matched | scratch | euclidean_two_point | 2 | 0.805 | 0.00659 | 0.037 |
| matched | scratch | euclidean_two_point | 4 | 0.802 | 0.00659 | 0.037 |
| matched | scratch | euclidean_two_point | 8 | 0.808 | 0.00659 | 0.037 |
| matched | scratch | euclidean_two_point | 12 | 0.806 | 0.00659 | 0.033 |
| matched | scratch | euclidean_two_point | 16 | 0.807 | 0.00659 | 0.034 |
| matched | scratch | fisher_two_point | 2 | 0.807 | 0.00659 | 0.038 |
| matched | scratch | fisher_two_point | 4 | 0.807 | 0.00659 | 0.042 |
| matched | scratch | fisher_two_point | 8 | 0.811 | 0.00659 | 0.039 |
| matched | scratch | fisher_two_point | 12 | 0.810 | 0.00659 | 0.033 |
| matched | scratch | fisher_two_point | 16 | 0.811 | 0.00659 | 0.034 |
| matched | warm | one_point | 2 | 0.846 | 0.00135 | 0.125 |
| matched | warm | one_point | 4 | 0.848 | 0.00133 | 0.125 |
| matched | warm | one_point | 8 | 0.852 | 0.00139 | 0.125 |
| matched | warm | one_point | 12 | 0.851 | 0.00133 | 0.125 |
| matched | warm | one_point | 16 | 0.852 | 0.00133 | 0.125 |
| matched | warm | euclidean_two_point | 2 | 0.849 | 0.00133 | 0.000 |
| matched | warm | euclidean_two_point | 4 | 0.850 | 0.00133 | 0.000 |
| matched | warm | euclidean_two_point | 8 | 0.855 | 0.00122 | 0.000 |
| matched | warm | euclidean_two_point | 12 | 0.855 | 0.00133 | 0.000 |
| matched | warm | euclidean_two_point | 16 | 0.855 | 0.00133 | 0.000 |
| matched | warm | fisher_two_point | 2 | 0.850 | 0.00113 | 0.000 |
| matched | warm | fisher_two_point | 4 | 0.855 | 0.00123 | 0.000 |
| matched | warm | fisher_two_point | 8 | 0.859 | 0.00113 | 0.000 |
| matched | warm | fisher_two_point | 12 | 0.859 | 0.00113 | 0.000 |
| matched | warm | fisher_two_point | 16 | 0.861 | 0.00133 | 0.000 |
| external | scratch | one_point | 2 | 0.701 | 0.00662 | 0.106 |
| external | scratch | one_point | 4 | 0.709 | 0.00662 | 0.106 |
| external | scratch | one_point | 8 | 0.716 | 0.00662 | 0.106 |
| external | scratch | one_point | 12 | 0.717 | 0.00638 | 0.106 |
| external | scratch | one_point | 16 | 0.718 | 0.00662 | 0.100 |
| external | scratch | euclidean_two_point | 2 | 0.702 | 0.00689 | 0.107 |
| external | scratch | euclidean_two_point | 4 | 0.710 | 0.00689 | 0.107 |
| external | scratch | euclidean_two_point | 8 | 0.716 | 0.00689 | 0.107 |
| external | scratch | euclidean_two_point | 12 | 0.717 | 0.00689 | 0.108 |
| external | scratch | euclidean_two_point | 16 | 0.718 | 0.00731 | 0.105 |
| external | scratch | fisher_two_point | 2 | 0.702 | 0.00689 | 0.107 |
| external | scratch | fisher_two_point | 4 | 0.709 | 0.00689 | 0.107 |
| external | scratch | fisher_two_point | 8 | 0.716 | 0.00689 | 0.107 |
| external | scratch | fisher_two_point | 12 | 0.718 | 0.00689 | 0.108 |
| external | scratch | fisher_two_point | 16 | 0.719 | 0.00662 | 0.105 |

## B. Coordinate stress (median across permutations)

| Data | Target | Method | kappa | rank | top-1 | top-5 | rho |
|---|---|---|---:|---:|---:|---:|---:|
| matched | scratch | one_point | 1 | 1.000 | 1.000 | 1.000 | 0.803 |
| matched | scratch | fisher_two_point | 1 | 1.000 | 1.000 | 1.000 | 0.811 |
| matched | scratch | euclidean_two_point | 1 | 1.000 | 1.000 | 1.000 | 0.807 |
| matched | warm | one_point | 1 | 1.000 | 1.000 | 1.000 | 0.852 |
| matched | warm | fisher_two_point | 1 | 1.000 | 1.000 | 1.000 | 0.861 |
| matched | warm | euclidean_two_point | 1 | 1.000 | 1.000 | 1.000 | 0.855 |
| matched | scratch | one_point | 3 | 1.000 | 1.000 | 1.000 | 0.803 |
| matched | scratch | fisher_two_point | 3 | 1.000 | 1.000 | 1.000 | 0.811 |
| matched | scratch | euclidean_two_point | 3 | 1.000 | 0.975 | 0.980 | 0.807 |
| matched | warm | one_point | 3 | 1.000 | 1.000 | 1.000 | 0.852 |
| matched | warm | fisher_two_point | 3 | 1.000 | 1.000 | 1.000 | 0.861 |
| matched | warm | euclidean_two_point | 3 | 1.000 | 0.975 | 0.980 | 0.855 |
| matched | scratch | one_point | 10 | 1.000 | 1.000 | 1.000 | 0.803 |
| matched | scratch | fisher_two_point | 10 | 1.000 | 1.000 | 1.000 | 0.811 |
| matched | scratch | euclidean_two_point | 10 | 0.998 | 0.925 | 0.940 | 0.807 |
| matched | warm | one_point | 10 | 1.000 | 1.000 | 1.000 | 0.852 |
| matched | warm | fisher_two_point | 10 | 1.000 | 1.000 | 1.000 | 0.861 |
| matched | warm | euclidean_two_point | 10 | 0.998 | 0.925 | 0.940 | 0.858 |
| matched | scratch | one_point | 30 | 1.000 | 1.000 | 1.000 | 0.803 |
| matched | scratch | fisher_two_point | 30 | 1.000 | 1.000 | 1.000 | 0.811 |
| matched | scratch | euclidean_two_point | 30 | 0.997 | 0.900 | 0.930 | 0.808 |
| matched | warm | one_point | 30 | 1.000 | 1.000 | 1.000 | 0.852 |
| matched | warm | fisher_two_point | 30 | 1.000 | 1.000 | 1.000 | 0.861 |
| matched | warm | euclidean_two_point | 30 | 0.997 | 0.900 | 0.930 | 0.860 |
| external | scratch | one_point | 1 | 1.000 | 1.000 | 1.000 | 0.718 |
| external | scratch | fisher_two_point | 1 | 1.000 | 1.000 | 1.000 | 0.719 |
| external | scratch | euclidean_two_point | 1 | 1.000 | 1.000 | 1.000 | 0.719 |
| external | scratch | one_point | 3 | 1.000 | 1.000 | 1.000 | 0.718 |
| external | scratch | fisher_two_point | 3 | 1.000 | 1.000 | 1.000 | 0.719 |
| external | scratch | euclidean_two_point | 3 | 0.999 | 0.877 | 0.883 | 0.718 |
| external | scratch | one_point | 10 | 1.000 | 1.000 | 1.000 | 0.718 |
| external | scratch | fisher_two_point | 10 | 1.000 | 1.000 | 1.000 | 0.719 |
| external | scratch | euclidean_two_point | 10 | 0.998 | 0.846 | 0.865 | 0.720 |
| external | scratch | one_point | 30 | 1.000 | 1.000 | 1.000 | 0.718 |
| external | scratch | fisher_two_point | 30 | 1.000 | 1.000 | 1.000 | 0.719 |
| external | scratch | euclidean_two_point | 30 | 0.992 | 0.831 | 0.852 | 0.712 |

## C. Non-uniform starts

| Data | Target | Method | beta | ESS | E/F rank | E/F top-1 | rho | NN |
|---|---|---|---:|---:|---:|---:|---:|---:|
| matched | scratch | one_point | 0.00 | 1.000 | 0.998 | 0.975 | 0.803 | 0.00659 |
| matched | scratch | euclidean_two_point | 0.00 | 1.000 | 0.998 | 0.975 | 0.807 | 0.00659 |
| matched | scratch | fisher_two_point | 0.00 | 1.000 | 0.998 | 0.975 | 0.811 | 0.00659 |
| matched | warm | one_point | 0.00 | 1.000 | 0.998 | 0.975 | 0.852 | 0.00133 |
| matched | warm | euclidean_two_point | 0.00 | 1.000 | 0.998 | 0.975 | 0.855 | 0.00133 |
| matched | warm | fisher_two_point | 0.00 | 1.000 | 0.998 | 0.975 | 0.861 | 0.00133 |
| matched | scratch | one_point | 0.25 | 0.935 | 0.999 | 0.850 | 0.792 | 0.00834 |
| matched | scratch | euclidean_two_point | 0.25 | 0.935 | 0.999 | 0.850 | 0.794 | 0.00753 |
| matched | scratch | fisher_two_point | 0.25 | 0.935 | 0.999 | 0.850 | 0.796 | 0.00847 |
| matched | warm | one_point | 0.25 | 0.935 | 0.999 | 0.850 | 0.826 | 0.00151 |
| matched | warm | euclidean_two_point | 0.25 | 0.935 | 0.999 | 0.850 | 0.827 | 0.00164 |
| matched | warm | fisher_two_point | 0.25 | 0.935 | 0.999 | 0.850 | 0.826 | 0.00139 |
| matched | scratch | one_point | 0.50 | 0.802 | 0.997 | 0.875 | 0.739 | 0.00834 |
| matched | scratch | euclidean_two_point | 0.50 | 0.802 | 0.997 | 0.875 | 0.751 | 0.00759 |
| matched | scratch | fisher_two_point | 0.50 | 0.802 | 0.997 | 0.875 | 0.731 | 0.00834 |
| matched | warm | one_point | 0.50 | 0.802 | 0.997 | 0.875 | 0.736 | 0.00164 |
| matched | warm | euclidean_two_point | 0.50 | 0.802 | 0.997 | 0.875 | 0.752 | 0.00164 |
| matched | warm | fisher_two_point | 0.50 | 0.802 | 0.997 | 0.875 | 0.725 | 0.00164 |
| matched | scratch | one_point | 1.00 | 0.521 | 0.998 | 0.900 | 0.677 | 0.00861 |
| matched | scratch | euclidean_two_point | 1.00 | 0.521 | 0.998 | 0.900 | 0.691 | 0.00861 |
| matched | scratch | fisher_two_point | 1.00 | 0.521 | 0.998 | 0.900 | 0.671 | 0.00866 |
| matched | warm | one_point | 1.00 | 0.521 | 0.998 | 0.900 | 0.664 | 0.00201 |
| matched | warm | euclidean_two_point | 1.00 | 0.521 | 0.998 | 0.900 | 0.680 | 0.00201 |
| matched | warm | fisher_two_point | 1.00 | 0.521 | 0.998 | 0.900 | 0.656 | 0.00178 |
| external | scratch | one_point | 0.00 | 1.000 | 1.000 | 0.862 | 0.718 | 0.00662 |
| external | scratch | euclidean_two_point | 0.00 | 1.000 | 1.000 | 0.862 | 0.718 | 0.00731 |
| external | scratch | fisher_two_point | 0.00 | 1.000 | 1.000 | 0.862 | 0.719 | 0.00662 |
| external | scratch | one_point | 0.25 | 0.935 | 0.999 | 0.923 | 0.671 | 0.00662 |
| external | scratch | euclidean_two_point | 0.25 | 0.935 | 0.999 | 0.923 | 0.666 | 0.00689 |
| external | scratch | fisher_two_point | 0.25 | 0.935 | 0.999 | 0.923 | 0.669 | 0.00689 |
| external | scratch | one_point | 0.50 | 0.802 | 0.998 | 0.815 | 0.579 | 0.00689 |
| external | scratch | euclidean_two_point | 0.50 | 0.802 | 0.998 | 0.815 | 0.585 | 0.00689 |
| external | scratch | fisher_two_point | 0.50 | 0.802 | 0.998 | 0.815 | 0.580 | 0.00689 |
| external | scratch | one_point | 1.00 | 0.521 | 0.997 | 0.862 | 0.495 | 0.00689 |
| external | scratch | euclidean_two_point | 1.00 | 0.521 | 0.997 | 0.862 | 0.491 | 0.00731 |
| external | scratch | fisher_two_point | 1.00 | 0.521 | 0.997 | 0.862 | 0.492 | 0.00731 |
