# Gaussian q0 on the checkpoint-135 probe bank

The Gaussian coordinate is normalized within-probe tour-cost rank;
rank 0 is the best tour and rank 1 is the worst.

| Target | Condition | Method | rho | false skip | NN median | E/F top-1 |
|---|---|---|---:|---:|---:|---:|
| scratch | uniform | euclidean_two_point | 0.8070 | 0.0312 | 0.00659 | 0.950 |
| scratch | uniform | fisher_two_point | 0.8113 | 0.0312 | 0.00659 | 0.950 |
| warm | uniform | euclidean_two_point | 0.8555 | 0.0000 | 0.00133 | 0.950 |
| warm | uniform | fisher_two_point | 0.8606 | 0.0000 | 0.00113 | 0.950 |
| scratch | mid_s010 | euclidean_two_point | 0.6422 | 0.1119 | 0.00856 | 0.900 |
| scratch | mid_s010 | fisher_two_point | 0.6319 | 0.1444 | 0.00866 | 0.900 |
| warm | mid_s010 | euclidean_two_point | 0.6732 | 0.0000 | 0.00129 | 0.900 |
| warm | mid_s010 | fisher_two_point | 0.6617 | 0.0000 | 0.00129 | 0.900 |
| scratch | mid_s020 | euclidean_two_point | 0.6703 | 0.1562 | 0.00856 | 0.875 |
| scratch | mid_s020 | fisher_two_point | 0.6712 | 0.1562 | 0.00866 | 0.875 |
| warm | mid_s020 | euclidean_two_point | 0.6929 | 0.0000 | 0.00129 | 0.875 |
| warm | mid_s020 | fisher_two_point | 0.6936 | 0.0000 | 0.00129 | 0.875 |
| scratch | mid_s035 | euclidean_two_point | 0.7756 | 0.1119 | 0.00847 | 0.900 |
| scratch | mid_s035 | fisher_two_point | 0.7782 | 0.0381 | 0.00847 | 0.900 |
| warm | mid_s035 | euclidean_two_point | 0.8305 | 0.0000 | 0.00164 | 0.900 |
| warm | mid_s035 | fisher_two_point | 0.8303 | 0.0000 | 0.00170 | 0.900 |
| scratch | best_s015 | euclidean_two_point | 0.6266 | 0.1119 | 0.00952 | 1.000 |
| scratch | best_s015 | fisher_two_point | 0.6170 | 0.1119 | 0.00952 | 1.000 |
| warm | best_s015 | euclidean_two_point | 0.6399 | 0.0000 | 0.00234 | 1.000 |
| warm | best_s015 | fisher_two_point | 0.6300 | 0.0000 | 0.00234 | 1.000 |
| scratch | best_s025 | euclidean_two_point | 0.6498 | 0.1562 | 0.01050 | 0.900 |
| scratch | best_s025 | fisher_two_point | 0.6348 | 0.1562 | 0.00909 | 0.900 |
| warm | best_s025 | euclidean_two_point | 0.6686 | 0.0000 | 0.00164 | 0.900 |
| warm | best_s025 | fisher_two_point | 0.6512 | 0.0000 | 0.00177 | 0.900 |
| scratch | best_s040 | euclidean_two_point | 0.6683 | 0.1562 | 0.00866 | 0.925 |
| scratch | best_s040 | fisher_two_point | 0.6647 | 0.1562 | 0.00909 | 0.925 |
| warm | best_s040 | euclidean_two_point | 0.6829 | 0.0000 | 0.00178 | 0.925 |
| warm | best_s040 | fisher_two_point | 0.6754 | 0.0000 | 0.00178 | 0.925 |
| scratch | tails_s010 | euclidean_two_point | 0.6846 | 0.2006 | 0.01061 | 0.825 |
| scratch | tails_s010 | fisher_two_point | 0.6790 | 0.1562 | 0.00866 | 0.825 |
| warm | tails_s010 | euclidean_two_point | 0.6762 | 0.0000 | 0.00171 | 0.825 |
| warm | tails_s010 | fisher_two_point | 0.6691 | 0.0000 | 0.00177 | 0.825 |
| scratch | tails_s020 | euclidean_two_point | 0.7793 | 0.1444 | 0.00856 | 0.975 |
| scratch | tails_s020 | fisher_two_point | 0.7759 | 0.1119 | 0.00856 | 0.975 |
| warm | tails_s020 | euclidean_two_point | 0.7933 | 0.0000 | 0.00145 | 0.975 |
| warm | tails_s020 | fisher_two_point | 0.7880 | 0.0000 | 0.00164 | 0.975 |
| scratch | tails_s035 | euclidean_two_point | 0.8077 | 0.0381 | 0.00753 | 0.975 |
| scratch | tails_s035 | fisher_two_point | 0.8093 | 0.1119 | 0.00856 | 0.975 |
| warm | tails_s035 | euclidean_two_point | 0.8551 | 0.0000 | 0.00129 | 0.975 |
| warm | tails_s035 | fisher_two_point | 0.8564 | 0.0000 | 0.00164 | 0.975 |
