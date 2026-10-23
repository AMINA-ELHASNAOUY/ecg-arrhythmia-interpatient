# Final evaluation

Main pipeline uses lead A only. Ensemble = 0.75 x logistic regression + 0.25 x CNN probabilities (two-lead ensemble: 0.5), weights chosen on DS1 validation patients.
AF detector: gradient boosting, threshold 0.75, chosen by leave-patient-out CV on DS1 (CV F1 0.8212). On DS2: AF sensitivity 82.6%, +P 92.8%, specificity 98.6%.
Fusion gate: F predictions are dropped in recordings where fewer than 1% of beats are predicted V (threshold fixed in advance, see src/reliability.py).

## DS2 — 22 unseen MIT-BIH patients

| Model | Acc | Macro-F1 | N Se | N +P | S Se | S +P | V Se | V +P | F Se | F +P |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | 85.5% | 0.593 | 85.8% | 99.2% | 90.4% | 31.3% | 82.8% | 92.8% | 52.6% | 6.3% |
| CNN (seed ensemble) | 94.3% | 0.626 | 97.6% | 96.5% | 26.6% | 36.0% | 96.7% | 91.8% | 20.4% | 47.0% |
| Ensemble (LR + CNN) | 88.5% | 0.617 | 89.1% | 98.9% | 86.0% | 35.8% | 87.0% | 93.1% | 45.6% | 7.3% |
| Ensemble + AF-aware rule | 90.7% | 0.645 | 91.9% | 98.6% | 79.8% | 48.4% | 87.0% | 93.1% | 45.6% | 7.3% |
| **Ensemble + AF rule + fusion gate** | 92.3% | 0.658 | 93.6% | 98.6% | 80.1% | 48.0% | 87.0% | 90.8% | 45.6% | 11.5% |
| Logistic regression + AF rule + fusion gate | 90.0% | 0.630 | 91.0% | 98.7% | 84.7% | 40.1% | 82.8% | 89.6% | 52.6% | 10.1% |
| Joint rhythm-and-beat model + fusion gate | 93.0% | 0.593 | 96.7% | 96.0% | 23.9% | 38.2% | 87.5% | 90.3% | 35.3% | 16.6% |
| Two leads: logistic regression | 86.4% | 0.556 | 87.9% | 98.2% | 70.8% | 28.5% | 80.5% | 72.1% | 38.7% | 7.8% |
| Two leads: CNN | 94.2% | 0.574 | 97.3% | 96.5% | 30.6% | 41.0% | 98.8% | 92.7% | 1.6% | 2.1% |
| Two leads: ensemble | 92.3% | 0.618 | 94.8% | 97.5% | 55.8% | 38.4% | 86.9% | 87.7% | 27.8% | 13.4% |

Support: N 44,218 · S 1,836 · V 3,219 · F 388

## SVDB — 78 patients, second database (external)

| Model | Acc | Macro-F1 | N Se | N +P | S Se | S +P | V Se | V +P | F Se | F +P |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | 70.1% | 0.498 | 70.9% | 98.1% | 66.5% | 43.9% | 60.2% | 68.2% | 47.8% | 0.0% |
| CNN (seed ensemble) | 91.0% | 0.516 | 96.6% | 94.5% | 21.4% | 59.4% | 85.6% | 72.6% | 39.1% | 0.4% |
| Ensemble (LR + CNN) | 76.3% | 0.523 | 77.8% | 97.9% | 64.7% | 48.0% | 65.5% | 69.7% | 47.8% | 0.0% |
| Ensemble + AF-aware rule | 76.6% | 0.525 | 78.5% | 97.5% | 61.3% | 50.8% | 65.5% | 69.7% | 47.8% | 0.0% |
| **Ensemble + AF rule + fusion gate** | 79.0% | 0.528 | 81.1% | 97.5% | 61.4% | 50.8% | 65.6% | 68.7% | 4.3% | 0.0% |
| Logistic regression + AF rule + fusion gate | 74.2% | 0.501 | 75.9% | 97.6% | 63.5% | 45.6% | 60.3% | 63.6% | 4.3% | 0.0% |
| Joint rhythm-and-beat model + fusion gate | 84.6% | 0.500 | 88.3% | 96.5% | 35.8% | 51.4% | 83.9% | 53.8% | 0.0% | 0.0% |
| Two leads: logistic regression | 77.8% | 0.493 | 79.5% | 97.8% | 61.9% | 43.7% | 70.2% | 49.7% | 47.8% | 0.1% |
| Two leads: CNN | 92.7% | 0.555 | 97.1% | 95.8% | 39.5% | 60.1% | 85.5% | 71.6% | 0.0% | 0.0% |
| Two leads: ensemble | 90.0% | 0.563 | 93.2% | 97.3% | 55.6% | 57.9% | 79.9% | 67.7% | 8.7% | 0.0% |

Support: N 162,193 · S 12,194 · V 9,937 · F 23

## INCART — 75 recordings, third database (external)

| Model | Acc | Macro-F1 | N Se | N +P | S Se | S +P | V Se | V +P | F Se | F +P |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | 69.9% | 0.456 | 69.8% | 99.5% | 95.8% | 12.0% | 68.2% | 91.8% | 46.1% | 0.3% |
| CNN (seed ensemble) | 93.4% | 0.621 | 93.9% | 99.3% | 85.1% | 44.0% | 90.8% | 90.9% | 44.3% | 1.5% |
| Ensemble (LR + CNN) | 76.9% | 0.487 | 77.4% | 99.5% | 95.5% | 15.3% | 71.5% | 92.5% | 48.4% | 0.4% |
| Ensemble + AF-aware rule | 77.0% | 0.488 | 77.5% | 99.4% | 95.5% | 15.5% | 71.5% | 92.5% | 48.4% | 0.4% |
| **Ensemble + AF rule + fusion gate** | 78.6% | 0.490 | 79.4% | 99.5% | 95.5% | 15.4% | 71.5% | 91.8% | 47.9% | 0.4% |
| Logistic regression + AF rule + fusion gate | 72.4% | 0.458 | 72.7% | 99.5% | 95.8% | 11.7% | 68.2% | 89.8% | 45.7% | 0.3% |
| Joint rhythm-and-beat model + fusion gate | 90.5% | 0.573 | 90.9% | 99.0% | 69.2% | 33.8% | 90.2% | 85.6% | 27.9% | 0.6% |
| Two leads: logistic regression | 79.1% | 0.454 | 79.2% | 99.4% | 85.7% | 12.3% | 78.4% | 64.9% | 31.1% | 0.4% |
| Two leads: CNN | 96.9% | 0.674 | 97.7% | 99.0% | 77.7% | 72.6% | 93.7% | 92.8% | 13.7% | 1.5% |
| Two leads: ensemble | 94.3% | 0.607 | 95.0% | 99.3% | 84.3% | 39.3% | 90.5% | 88.3% | 26.5% | 1.4% |

Support: N 153,541 · S 1,958 · V 20,000 · F 219

## Per-patient errors on DS2 (ensemble + AF rule + fusion gate)

`false F (no gate)` = false fusion alarms before the fusion gate.

| Record | Beats | S beats | S found | false S | V beats | V found | false V | F beats | false F (no gate) | false F | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 222 | 2481 | 209 | 110 | 389 | 0 | 0 | 19 | 0 | 432 | 0 | 79.6% |
| 233 | 3077 | 7 | 7 | 47 | 830 | 756 | 4 | 11 | 472 | 472 | 82.9% |
| 105 | 2565 | 0 | 0 | 78 | 41 | 4 | 134 | 0 | 169 | 169 | 85.2% |
| 113 | 1793 | 6 | 6 | 219 | 0 | 0 | 0 | 0 | 1 | 0 | 87.8% |
| 232 | 1778 | 1381 | 1174 | 1 | 0 | 0 | 4 | 0 | 93 | 0 | 88.3% |
| 202 | 2134 | 55 | 44 | 208 | 19 | 18 | 4 | 1 | 76 | 0 | 89.6% |
| 228 | 2051 | 3 | 3 | 72 | 362 | 361 | 12 | 0 | 125 | 125 | 89.8% |
| 213 | 3249 | 28 | 27 | 6 | 220 | 119 | 4 | 362 | 126 | 126 | 90.0% |
| 221 | 2425 | 0 | 0 | 7 | 396 | 368 | 0 | 0 | 214 | 214 | 90.0% |
| 214 | 2258 | 0 | 0 | 82 | 256 | 176 | 0 | 1 | 127 | 127 | 90.7% |
| 231 | 1569 | 1 | 0 | 131 | 2 | 1 | 0 | 0 | 2 | 0 | 91.6% |
| 219 | 2152 | 7 | 3 | 148 | 64 | 41 | 0 | 1 | 10 | 10 | 91.9% |
| 210 | 2648 | 22 | 7 | 39 | 195 | 175 | 14 | 10 | 77 | 77 | 94.3% |
| 200 | 2599 | 30 | 27 | 77 | 825 | 772 | 1 | 2 | 40 | 40 | 95.2% |
| 111 | 2122 | 0 | 0 | 3 | 1 | 1 | 55 | 0 | 130 | 0 | 97.3% |
| 121 | 1861 | 1 | 1 | 18 | 1 | 1 | 15 | 0 | 28 | 0 | 98.2% |
| 117 | 1533 | 1 | 1 | 4 | 0 | 0 | 17 | 0 | 62 | 0 | 98.6% |
| 103 | 2082 | 2 | 2 | 27 | 0 | 0 | 0 | 0 | 2 | 0 | 98.7% |
| 123 | 1516 | 0 | 0 | 19 | 3 | 3 | 0 | 0 | 1 | 0 | 98.8% |
| 234 | 2751 | 50 | 26 | 1 | 3 | 3 | 1 | 0 | 11 | 0 | 99.1% |
| 212 | 2746 | 0 | 0 | 19 | 0 | 0 | 0 | 0 | 21 | 0 | 99.3% |
| 100 | 2271 | 33 | 33 | 1 | 1 | 1 | 0 | 0 | 20 | 0 | 100.0% |
