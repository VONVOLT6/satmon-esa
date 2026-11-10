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

### Addendum 1 (2026-11-08): C1 burst design, before any C1 test run

The first C1 design opened a burst of LSTM forecasts only from the trigger onwards (plus 2,400 warm-up samples). On the development period this lost alarms whenever the cheap tier fired partway into an event. Seed 42 lost F0.5 even at 92% LSTM share (0.721 vs 0.792). Those partial results are kept in `results/c1_dev_v1_no_lookback.*`.

A look-back/post grid on seed 42's development data (`src/diag_c1_grid.py`) showed that 2,100 samples of look-back and 2,100 after the trigger recover the full-LSTM development score (0.792 at τ = 0.999, 12.3% LSTM share). Longer bursts gave no further gain. The final C1 design therefore uses LOOKBACK = 2100 and POST = 2100.

H2 selection, unchanged in spirit:
- **Choosing τ:** the full router is run on all 5 seeds over τ ∈ {0.995, 0.998, 0.999, 0.9995, 0.9999}. The chosen τ is the one with the smallest mean LSTM share whose mean development F0.5 is ≥ 0.776 − 0.01.
- **Test:** the chosen τ is applied once to the test set.
- **Claim:** non-inferiority (lower 95% bound of F0.5 − B0 above −0.02), together with the measured LSTM share.

### Addendum 2 (2026-11-09): C2 sensor reliability (hypotheses H3a, H3b), before any C2 code was run

ESA-AD has no sensor-fault labels, so C2 is evaluated on **injected** faults. All claims are limited to those.

- **Fault injection.**
  - *Count and types:* 50 single-channel faults per period, 10 of each type. The types are stuck (value frozen), drift (linear ramp to k·σ), bias (step of k·σ), noise (added Gaussian noise of k·σ) and dropout (value reads 0).
  - *Size:* σ is the channel's training standard deviation. k is drawn from U(4, 8) for bias and drift, and from U(2, 4) for noise.
  - *Placement:* each fault lasts 2 to 24 hours, never overlaps a labelled event, and stays at least 3,000 samples from another fault.
  - *Random seeds:* fixed at 101 for the development period and 202 for the test period.
- **Pipeline.** The baseline is the always-on LSTM (B0) run on the injected data, with its forecasts recomputed wherever a fault changes the input. C2 scores every alarm with a logistic model on peer-consistency features (does one channel disagree with the five others in its group?), plus stuck, noise and out-of-range features. Alarms scored above a threshold are relabelled "sensor fault" and removed from the system-alarm stream.
- **Training and threshold, on the development period only.**
  - *Training labels:* alarms overlapping an injected fault count as sensor faults, and alarms overlapping a labelled event count as system events. All other alarms are left out of training.
  - *Threshold:* the one that maximises sensor-fault F1, provided at most 5% of real-event alarms are suppressed (pooled over the 5 seeds).
- **H3a (improvement):** on the injected test set, system-alarm F0.5 (scored against the real labels only) improves on B0. The 95% paired bootstrap interval must lie above 0.
- **H3b (no harm):** on the clean test set, C2 applied to B0's alarms has a lower 95% bound above −0.02 F0.5 relative to B0.
- **Descriptive results:** the share of injected faults that raise any alarm, and per fault type, the share of those alarms correctly attributed to the sensor.
- **Holm correction:** applies across H3a and H3b.

### Addendum 3 (2026-11-09): C3 incident correlation (hypothesis H4), before any C3 data was prepared

- **Data:** all 76 Mission 1 channels, on a 30-second zero-order-hold grid. The periods match those above: training 2000-01 to 2001-09 for the graph and the detector calibration, development 2001-10 to 2006-12 for tuning, and test 2007-2013.
- **Alerts:** a cheap per-channel detector (robust z-score of a univariate autoregressive residual, smoothed) produces channel alerts. Its threshold is set on training data only, at a fixed false-alert rate. The full-set LSTM is not used, because the paper reports 8.5 hours of training for it.
- **Dependency graph:** learned from training data only. Channels are linked when the correlation of their smoothed residual activity reaches ρ. The subsystem and group lists from `channels.csv` are reported as a comparison prior.
- **Grouping methods compared:**
  - *G0:* every channel alert is its own incident.
  - *G1:* alerts within ±W samples of each other are merged regardless of channel, by time only.
  - *G2 (proposed):* alerts within ±W are merged only if their channels are connected in the learned graph.
  - *Tuning:* W and ρ are chosen on the development period to maximise pairwise grouping F1.
- **Ground truth:** a pair of alerts that both overlap labelled events belongs together if both overlap the same event ID.
- **Metrics:** pairwise grouping F1, incidents per detected event (operator load), the share of incidents mixing two or more event IDs, and onset-order agreement within multi-channel events.
- **H4:** on the test set, G2's pairwise F1 is higher than G1's. The 95% interval comes from a paired bootstrap over labelled events.

Limits fixed in advance: channel meanings are anonymised, so the graph shows learned statistical dependence, not physical causation.

### Addendum 4 (2026-11-09): C4 cost-aware response replay (hypothesis H5), before any C4 code was run

C4 replays monitoring decisions on the recorded timeline. It is **not** an estimate of what the spacecraft would have done, and ESA-AD has no response or cost records. So every result is conditional on a declared cost grid, and the grid is reported in full.

- **Alarms:** the always-on LSTM's alarm events on the fault-injected periods from C2 (development seed 101, test seed 202).
- **True class of each alarm:**
  - *anomaly:* overlaps a labelled Anomaly.
  - *rare:* overlaps a labelled Rare Event and no Anomaly.
  - *sensor:* overlaps an injected fault.
  - *false:* none of the above.
- **Actions and costs:**

  | Action | anomaly | rare | sensor | false |
  |---|---|---|---|---|
  | page | 0 | F | F | F |
  | ticket | D | T | 0 | T |
  | ignore | M | 0 | S | 0 |

  - *Cost grid:* M ∈ {20, 50, 100}, D ∈ {2, 5, 10}, F ∈ {1, 3}, S ∈ {1, 3}, T = 0.5 (36 grid points).
- **Policies:**
  - *P0:* page every alarm, as today.
  - *P1:* C2 rule. Ticket if C2 says sensor fault, otherwise page.
  - *P2 (proposed):* a multinomial logistic model estimates the class probabilities from C2's features plus alarm duration, trained on development alarms only. It then picks the action with the lowest expected cost at each grid point.
  - *Oracle:* knows the true class.
- **H5:** P2's regret (cost minus the oracle's) is below both P0's and P1's at ≥ 80% of the 36 grid points, averaged over the 5 seeds. A paired monthly-block bootstrap of the cost difference P2 − P1 is reported at the central grid point (M = 50, D = 5, F = 1, S = 1).
- **Stability (G5):** how often each policy ranks first across the grid is reported. An unstable ranking is reported as conditions, not a winner.

### Addendum 5 (2026-11-10): C5 ablation (configurations A0-A8, L1-L4), before any C5 code was run

C5 measures what each contribution adds when they are combined in one pipeline. It is **descriptive**: no new confirmatory hypothesis, no Holm correction. Point estimates come with paired monthly-block bootstrap intervals. All components keep the settings already fixed on the development period. Nothing new is tuned, except one small model described under C4 below.

- **Setting:** channels 41-46, 21-month models, 5 seeds. The main condition is the test period with C2's injected faults (seed 202), the only setting where C1, C2, C3 and C4 all act. The clean test period is reported for F0.5 only.
- **Components:**
  - *C1:* the router at τ = 0.999 (addendum 1). On the injected test the cheap tier is re-scored on the injected values, and the LSTM uses forecasts recomputed where faults change its input, as in C2. Variants: confidence only, confidence + impact, full.
  - *C2:* the classifier and threshold from addendum 2. Alarms labelled "sensor fault" leave the system-alarm stream and become tickets.
  - *C3:* alarm events within W = 2,880 samples are merged into one incident, using G2's W chosen on development (addendum 3). The learned graph links all six channels 41-46 (every pairwise correlation ≥ 0.68, far above ρ = 0.1), so on this group G2 reduces to time grouping. With C2 on, system alarms and sensor-fault alarms are grouped separately.
  - *C4:* P2 from addendum 4 when C2 is present. Without C2, its features are unavailable, so a no-C2 variant of the same multinomial model is fitted on the development alarms using only log duration and number of channels alarmed. When C4 is off, the policy is P1 (C2 rule) if C2 is present, otherwise P0 (page every alarm).
- **Configurations:**

  | Config | Detector | C2 | C3 | C4 |
  |---|---|---|---|---|
  | A0 | always-on LSTM (B0) | | | |
  | A1 | cheap tier alone, its triggers are the alarms | | | |
  | A2 | C1 confidence only | | | |
  | A3 | C1 confidence + impact | | | |
  | A4 | C1 full | | | |
  | A5 | C1 full | yes | | |
  | A6 | C1 full | | yes | |
  | A7 | C1 full | | | yes (no-C2 variant) |
  | A8 | C1 full | yes | yes | yes |
  | L1 | always-on LSTM (A8 minus C1) | yes | yes | yes |
  | L2 | C1 full (A8 minus C2) | | yes | yes (no-C2 variant) |
  | L3 | C1 full (A8 minus C3) | yes | | yes |
  | L4 | C1 full (A8 minus C4) | yes | yes | P1 |

- **Metrics, mean of 5 seeds:**
  - *Detection:* system-alarm event-wise F0.5 against the real labels, on the injected test and on the clean test.
  - *Compute:* LSTM share.
  - *Operator load:* items raised (incidents with C3, alarm events without), and items that mix a real event with an injected fault.
  - *Response cost:* cost at the central grid point and mean regret over the 36-point grid. Cost stays per alarm event, as in addendum 4; C3 does not change it.
- **Intervals:** paired monthly-block bootstrap of the injected-test F0.5 difference for A8 − A0 and for A8 − L1 ... L4. Monthly bootstrap of the central-point cost difference for A8 − L4.
