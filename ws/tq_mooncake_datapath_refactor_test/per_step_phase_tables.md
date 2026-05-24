### `gen` per-step transfer time (s)

| backend | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (no TQ) | 33.7 | 28.4 | 41.8 | 36.1 | 42.8 | 37.7 | 48.5 | 34.4 | 36.8 | 34.9 |
| TQ + SimpleStorage | 5.8 | 0.43 | 10.2 | 6.1 | 19.3 | 0.54 | 41.2 | 0.19 | 0.59 | 1.6 |
| TQ + Mooncake refactor | 0.63 | 0.55 | 5.5 | 17.1 | 24.5 | 5.1 | 32.4 | 13.0 | 0.55 | 0.24 |

### `old_log_prob` per-step transfer time (s)

| backend | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (no TQ) | 34.3 | 39.3 | 63.5 | 62.6 | 77.9 | 37.9 | 56.8 | 42.9 | 30.0 | 43.9 |
| TQ + SimpleStorage | 0.25 | 0.24 | 0.56 | 0.56 | 0.24 | 0.24 | 0.24 | 0.27 | 0.23 | 0.24 |
| TQ + Mooncake refactor | 0.87 | 0.42 | 0.46 | 0.92 | 0.42 | 0.46 | 0.54 | 0.42 | 0.43 | 0.90 |

### `adv` per-step transfer time (s)

| backend | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (no TQ) | — | — | — | — | — | — | — | — | — | — |
| TQ + SimpleStorage | 0.30 | 0.41 | 0.41 | 0.32 | 0.64 | 0.62 | 0.31 | 0.38 | 0.31 | 0.31 |
| TQ + Mooncake refactor | 0.50 | 0.57 | 0.53 | 0.49 | 0.79 | 0.51 | 0.52 | 0.48 | 0.56 | 0.51 |

### `update_actor` per-step transfer time (s)

| backend | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (no TQ) | 42.1 | 37.3 | 58.9 | 58.0 | 76.4 | 35.4 | 55.1 | 42.6 | 29.3 | 48.1 |
| TQ + SimpleStorage | 11.0 | 8.7 | 12.7 | 14.4 | 16.6 | 9.6 | 15.7 | 10.4 | 7.4 | 11.4 |
| TQ + Mooncake refactor | 7.1 | 5.8 | 8.0 | 7.9 | 9.3 | 6.3 | 7.9 | 6.1 | 5.0 | 6.9 |
