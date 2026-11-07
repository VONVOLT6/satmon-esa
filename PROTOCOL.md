# Evaluation protocol (phase P2, frozen 2026-11-08)

This file fixes how every new method is tuned and judged **before** it touches the test set. Change it only through a dated addendum at the bottom, committed before the affected test run.

## Data

| Period | Dates | Use |
|---|---|---|
| Training | 2000-01-01 to 2001-10-01 (21 months) | Fit models |
| Development | 2001-10-01 to 2007-01-01, 33 labelled events | Tune and select. Unseen by the 21-month models, and still before the test period |
| Test | 2007-01-01 to 2014-01-01, 65 events (29 anomalies, 36 rare events) | Final check only |

Data: ESA-ADB Mission 1, channels 41-46. The development period exists only for the 21-month models, because the 84-month models train on it. Confirmatory comparisons therefore use the 21-month models.

## Baseline

**B0 = Telemanom-ESA-Pruned**, 21 months, `min_error_value` 0.007, last-epoch weights, seeds 42, 1, 2, 3, 4. Test F0.5 = 0.801 for every seed.

## Metrics

- **Primary:** corrected event-wise F0.5 (official `ESAScores`, anomalies and rare events), averaged over the 5 seeds.
- **Secondary:** anomalies-only F0.5, event-wise precision and recall, alarming precision and affiliation F0.5. For compute-related methods, also share of windows sent to the LSTM, wall-clock time and GPU time.

## Statistics

- **Paired monthly-block bootstrap** (`src/stats.py`): the test period is split into calendar months, months are resampled with replacement (5,000 replicates), and both detectors are scored on the same resample. Its decomposition reproduces the official scores exactly (`python src/stats.py check`).
- **Improvement claim:** the 95% interval for (method − B0) lies entirely above 0, after Holm correction across the confirmatory hypotheses below.
- **"No worse" claim** (for methods that save compute): the lower 95% bound is above −0.02 F0.5.
- **Reporting:** point estimates are always reported with their interval. Negative and null results are reported too.

## Confirmatory hypotheses

**H1 · Calibrated error floor.** The Pruned threshold of 0.007 was hand-picked by the paper's authors.

- **Candidates:** 0, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03. This grid was fixed on 2026-11-03, before any development results.
- **Selection rule:** the floor with the highest mean development F0.5 across the 5 seeds.
- **Test:** the selected floor is applied once to the test set.
- **Claim:** test F0.5 improves on B0.

**H2 · Adaptive computation (C1).** A cheap screening detector plus the LSTM on demand loses no more than 0.02 F0.5 against B0 while running the LSTM on fewer windows. The routing design, its own development-period selection and the exact compute measure will be fixed in an addendum before its test run.

## Rules

1. Nothing is tuned on the test set. Every choice is made on training or development data.
2. Each test-set run is logged in `results/test_log.md` with date, commit and hypothesis.
3. A hypothesis is tested once. A rerun after changes counts as a new, exploratory result and is labelled as such.
4. The ablation design (configurations A0-A8 and L1-L4 in the assessment document) is evaluated under these same rules in phase P6.

## Addenda

None yet.
