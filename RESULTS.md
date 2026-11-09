# Satellite Telemetry Anomaly Detection: Reproducing Telemanom-ESA on ESA-ADB

## 1. Purpose

This experiment trains the Telemanom-ESA anomaly detector on the ESA Anomaly Detection Benchmark (ESA-ADB) and compares it with the published results.

The aim is a verified baseline. Our later contributions (adaptive computation, sensor reliability, incident correlation and response planning) will be measured against it.

> **Comparing with other experiments.** These scores are only comparable with results that use the same protocol: ESA-ADB **Mission 1, channels 41-46**, the **official chronological split** (training before 2007, test 2007-2013) and the **official event-wise metrics** (`ESAScores`, corrected event-wise F0.5).
>
> Results on another mission or channel, a random or percentage split, or sample-level precision/recall/F1/PR-AUC measure a different task. They cannot be ranked against the numbers here. Sample-level F1 counts every 30-second point separately. Event-wise F0.5 counts whole anomaly events and weights false alarms more heavily.

## 2. Dataset

- **Dataset:** ESA Anomaly Dataset (ESA-AD), Zenodo record 15237121, CC BY 3.0 IGO licence
- **Mission:** Mission 1
- **Channels:** 41-46, the benchmark's "lightweight" subset. All six belong to subsystem 5.
- **Sampling:** a 30-second grid with zero-order hold, as in the official preprocessing
- **Labels:** event start and end times annotated by ESA spacecraft operations engineers

| Split | Period | Rows |
|---|---|---:|
| Training (21 months) | 2000-01-01 to 2001-10-01 | 1,840,321 |
| Training (84 months) | 2000-01-01 to 2007-01-01 | 7,364,161 |
| Test | 2007-01-01 to 2014-01-01 | 7,364,161 |

The test period contains **65 labelled events: 29 anomalies and 36 rare nominal events**.

The training-row counts match the official `datasets.csv` exactly. This confirms our data preparation reproduces the benchmark's time grid.

The validation period is the last 3 months of each training period, as in the official experiment script. The test period is never used for training or tuning.

## 3. Model

**Telemanom-ESA** (Hundman et al. 2018, adapted by Kotowski et al. 2024) is a forecasting detector:

1. A two-layer LSTM (80 units each, dropout 0.3) reads the last 250 samples (about 2 hours) of all six channels. It predicts the next 10 samples.
2. The error between forecast and actual value is smoothed for each channel.
3. A non-parametric dynamic threshold flags unusually large errors. The original `errors.py` from the ESA-ADB repository does this step, unchanged.

The two variants share one trained model and differ in a single setting:

| Variant | `min_error_value` | Effect |
|---|---|---|
| Telemanom-ESA | 0 | Every error spike above the dynamic threshold raises an alarm |
| Telemanom-ESA-Pruned | 0.007 | Error spikes smaller than 0.007 are ignored |

## 4. Experimental Setup

| Setting | Value | Source |
|---|---|---|
| Window / forecast horizon | 250 / 10 samples | Official defaults |
| Batch size, optimiser | 70, Adam (learning rate 0.001) | Official defaults |
| Steps per epoch | 1,000 | Official code |
| Early stopping | patience 20, min delta 0.0003 on validation loss | Official experiment script |
| Weights used | Last epoch | The original overwrites its best checkpoint with the final model |
| Seeds | 42, 1, 2, 3, 4 | Ours, to measure run-to-run variation |
| Hardware | NVIDIA RTX 5060 Laptop GPU, Intel Core Ultra 9 275HX | |
| Framework | PyTorch 2.11 (CUDA 12.8) | The official TensorFlow 2.5.1 image cannot use this GPU |

All scores are computed by the official ESA-ADB metric code (`ESAScores`), loaded unmodified from the benchmark repository at commit 67194d3. Ten hand-calculated test cases confirmed the metric code behaves as documented before we used it.

**Main metric:** the corrected event-wise F0.5-score. It counts whole events rather than single samples, weights precision above recall (so false alarms cost more than misses), and penalises the share of nominal time covered by false alarms.

## 5. Results

### 5.1 Comparison with the published paper

![Ours vs paper](results/figures/1_ours_vs_paper.png)

Corrected event-wise F0.5, anomalies and rare events, mean of five seeds:

| Model | Training data | Paper | Ours (mean) | Ours (min-max) | Difference |
|---|---|---:|---:|---:|---:|
| **Telemanom-ESA-Pruned** | 21 months | 0.776 | **0.801** | 0.801-0.801 | **+0.025** |
| **Telemanom-ESA-Pruned** | 84 months | 0.786 | **0.784** | 0.773-0.791 | **-0.002** |
| Telemanom-ESA | 21 months | 0.122 | 0.105 | 0.084-0.117 | -0.017 |
| Telemanom-ESA | 84 months | 0.178 | 0.106 | 0.087-0.141 | -0.072 |

**Telemanom-ESA-Pruned, the paper's best Mission 1 model, reproduces the published result**: it exceeds it at 21 months (+0.025) and matches it at 84 months (-0.002). Both are within our ±0.03 acceptance band. The unpruned variant scores lower than the paper, mainly because it raises more false alarms (Section 5.4).

### 5.2 All official metrics

![All metrics, 21 months](results/figures/2_all_metrics_21_months.png)

Telemanom-ESA-Pruned, trained on 21 months, mean of five seeds:

| Metric | All events | Anomalies only |
|---|---:|---:|
| Event-wise precision (corrected) | 0.999 | 0.999 |
| Event-wise recall | 0.446 | 0.862 |
| **Event-wise F0.5** | **0.801** | **0.968** |
| Alarming precision | 0.829 | 0.806 |
| Affiliation-based F0.5 | 0.634 | 0.913 |

The 84-month chart is in `results/figures/2_all_metrics_84_months.png`.

### 5.3 Event-level results

![Events detected](results/figures/6_events_detected.png)

Telemanom-ESA-Pruned, trained on 21 months, seed 42:

| | Detected | Missed | Total |
|---|---:|---:|---:|
| Anomalies | **25** | 4 | 29 |
| Rare nominal events | 4 | 32 | 36 |
| Alarms that hit no labelled event | 0 | | |

The model finds 25 of the 29 real anomalies, and none of its alarms land outside a labelled event. Most missed events are rare nominal events: unusual but harmless spacecraft behaviour, which ESA engineers annotated and the benchmark still counts.

### 5.4 Example detection

![Example detection](results/figures/5_example_detection.png)

The LSTM forecast (blue) follows the normal periodic signal closely. When the channel drops and then jumps during a labelled anomaly, the forecast cannot follow, the error grows and the alarm switches on.

### 5.5 Run-to-run stability and training

![Seed stability](results/figures/3_seed_stability.png)

- **Pruned variant:** at 21 months, all five seeds give the same event-wise F0.5 (0.801). The five models make different predictions and flag different samples (34,033 to 34,658), but they catch the same events. The fixed 0.007 error gate decides which events survive, more than the network does. At 84 months, three seeds score 0.791 and two score 0.773: those two each raised one false alarm, and all five caught the same events.
- **Unpruned variant:** this varies by seed: 0.084 to 0.117 at 21 months. At 84 months its recall matches the paper (0.898 mean here, 0.894 in the paper), but its precision is lower (0.071-0.117, paper 0.148). It therefore raises more false alarms than the paper's single run.

![Training curves](results/figures/4_training_curves_21_months.png)

At 21 months, training stopped after 37 to 58 epochs and took 5.5 to 10.5 minutes per seed. At 84 months, it stopped after 26 to 43 epochs and took 3.6 to 6.7 minutes. Scoring the 7-year test period takes about 10 minutes per model.

## 6. Related IEEE Papers

Three IEEE papers on satellite telemetry anomaly detection that our project builds on and combines:

| # | IEEE paper | What it does | What we take from it |
|---|---|---|---|
| 1 | Z. Zeng, G. Jin, C. Xu, S. Chen, Z. Zeng, L. Zhang, "Satellite Telemetry Data Anomaly Detection Using Causal Network and Feature-Attention-Based LSTM," *IEEE Transactions on Instrumentation and Measurement*, vol. 71, 2022. doi:10.1109/TIM.2022.3151930 | Builds a causal network between telemetry parameters (transfer entropy), predicts telemetry with an attention LSTM, and flags anomalies with a nonparametric dynamic k-sigma threshold. Reports F1 0.9462 and FPR 0.0021 on a real satellite. | Same detection principle as our baseline: LSTM forecasting plus a dynamic threshold. Its causal network motivates our dependency-graph contribution (C3), which links alarms across channels. Its focus on cutting false positives matches our F0.5 metric, which weights precision above recall. |
| 2 | Z. Xu, Z. Cheng, B. Guo, "A Multivariate Anomaly Detector for Satellite Telemetry Data Using Temporal Attention-Based LSTM Autoencoder," *IEEE Transactions on Instrumentation and Measurement*, vol. 72, 2023. doi:10.1109/TIM.2023.3296125 | A temporal-attention LSTM autoencoder for multivariate telemetry. It scores anomalies by the Mahalanobis distance of the reconstruction error, uses an adaptive threshold balancing reliability and accuracy, and identifies which parameters caused each anomaly. | Its adaptive threshold and per-parameter attribution feed our trust-aware decision layer. Our results show the threshold, not the network, decides which events are caught (Section 5.5). Its attribution step relates to our sensor-reliability contribution (C2). |
| 3 | Y. Zhang, P. Hu, "A Fourier Graph Attention Network for Satellite Anomaly Detection," *2025 IEEE Canadian Conference on Electrical and Computer Engineering (CCECE)*, 2025. doi:10.1109/CCECE64018.2025.11364456 | Fourier-GAT combines graph attention with frequency-domain features to capture relationships across channels and time in satellite telemetry. It targets anomaly detection with root-cause analysis and reports better results than mainstream methods on three real datasets. | A graph-based model across channels, the direction of our dependency-graph work (C3). Our telemetry is strongly periodic (Section 5.4), so frequency features suit it. |

Also relevant: A. Mathur, S. Paheding, J. Salafia, S. Salafia, "ADEPTRS: Adversarially Resilient Anomaly Detection for Satellite Telemetry under Evasion and Poisoning Attacks," *IGARSS 2026*, doi:10.1109/IGARSS60577.2026.11730340. It tests how robust telemetry anomaly detectors are, which is relevant to the "trust-aware" part of our title.

**Sources of the dataset and baseline model** (not IEEE, cited because we reproduce them):

- K. Kotowski et al., "European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry," arXiv:2406.17826, 2024. The ESA-ADB dataset, metrics and the published results in Section 5.1.
- K. Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding," KDD 2018, arXiv:1802.04431. The original Telemanom method.

## 7. How to Run

```bash
cd src
python fetch_subset.py            # download channels 41-46 and labels (506 MB)
python prepare.py 21_months       # build the 30-second grid; also 84_months
python test_metrics.py            # check the official metric code
python train.py 21_months --seed 42
python evaluate.py 21_months --seed 42
python summarize_seeds.py 21_months
python make_figures.py            # charts in results/figures/
```

Each step resumes after an interruption. `run_seeds.py 21_months 42 1 2 3 4` runs several seeds in a row.

## 8. Output Files

| File | Contents |
|---|---|
| `results/telemanom_<months>_seed<n>_last.json` | All official metrics for one run |
| `results/seed_summary_<months>_last.md` | Mean, spread and per-seed values |
| `results/figures/` | Charts used in this report, and `tables.json` |
| `models/telemanom_<months>_seed<n>/last.pt` | Trained model weights |

## 9. Limitations

1. Only the six lightweight channels of Mission 1 have been evaluated so far, not the full 76-channel set or Mission 2.
2. The model is a PyTorch port, not the original TensorFlow 2.5.1 code. The settings and thresholding code match, but small numerical differences remain. They are the most likely reason the unpruned variant has lower precision than the paper.
3. The current Zenodo labels give 65 test events. The paper's recall values (0.894 = 59/66 and 0.424 = 28/66) imply 66. This suggests a later label revision, which we have not confirmed.
4. The Pruned variant's 0.007 threshold was hand-picked by the paper's authors. Its results depend on that one number.
5. Recall is low on rare nominal events (4 of 36 detected).

## 10. Next Steps

1. Freeze the evaluation protocol: validation folds inside the training period and the comparisons we will report (phase P2).
2. Build adaptive computation (C1): a cheap detector screens most data, and the LSTM runs only where confidence is low, impact is high or the pattern is unfamiliar. It will be evaluated against this baseline on detection quality and compute cost.
3. Improve rare-event recall without adding false alarms. This is the baseline's clearest weakness.

## 11. Conclusion

We trained Telemanom-ESA on ESA-ADB Mission 1 and evaluated it with the official benchmark code. **The Pruned variant reaches a corrected event-wise F0.5 of 0.801 at 21 months (paper: 0.776) and 0.784 at 84 months (paper: 0.786), averaged over five training seeds.**

This gives a verified, reproducible baseline for the project's contributions. It also shows where the room for improvement lies: the decision layer that turns forecast errors into alarms, and recall on rare events.

## 12. Update (2026-11-08): calibrated error floor (H1)

The Pruned variant's error floor of 0.007 was hand-picked by the paper's authors. Under the frozen protocol in `PROTOCOL.md`, we chose a floor from 8 candidates on a labelled development period (2001-10 to 2006-12, 33 events, unseen by the 21-month models). We then applied it **once** to the test set.

| Floor (development, mean of 5 seeds) | 0 | 0.003 | **0.005** | 0.007 | 0.01 | 0.015 | 0.02 | 0.03 |
|---|---|---|---|---|---|---|---|---|
| Event-wise F0.5 | 0.096 | 0.485 | **0.816** | 0.776 | 0.752 | 0.752 | 0.538 | 0.492 |

0.005 was best on all 5 seeds in development.

**Test result (21 months, 5 seeds):**

| | Paper's floor 0.007 | Calibrated 0.005 | Difference (95% CI, paired monthly bootstrap) |
|---|---|---|---|
| F0.5, all events | 0.801 | **0.818** | +0.017 [−0.009, +0.055] |
| F0.5, anomalies only | 0.968 | 0.957 | −0.012 [−0.032, −0.000] |
| Events found (of 65) | 29 | 31-32 | +2 to +3, all rare events |
| False alarms | 0 | 0 (3 seeds), 1 (2 seeds) | |

**Verdict: no clear difference.** The calibrated floor scores higher on every seed and finds 2-3 more rare events. However, the 95% interval includes zero, so by the protocol this is not a confirmed improvement. Anomaly detection is unchanged at 25 of 29. The small anomalies-only drop comes from one extra false alarm in two of the five seeds.

This is the main lesson for the next contributions. With only 65 test events, a gain of about 0.02 F0.5 is within run-to-run noise, so a confirmed improvement needs either a larger effect or more events (the full channel set). Full numbers: `results/h1_result.json`. Test-set log: `results/test_log.md`.

## 13. Update (2026-11-09): adaptive computation (C1, hypothesis H2)

**Idea.** The original detector runs the LSTM on every 30-second reading. C1 adds a cheap screening tier that watches everything and calls the LSTM only when something looks wrong. The tier has three signals:
- **Confidence:** a small linear forecaster's error.
- **Novelty:** how unusual the recent pattern is compared with the training data.
- **Impact:** how many channels deviate at once.

When the screener fires, the LSTM runs on a burst around that point, because its alarm logic needs about 2,100 samples of error history. The burst covers 2,400 samples of warm-up and 2,100 of look-back before the trigger, and 2,100 after it. Every sample in a burst counts as LSTM work, padding included.

**Development period (21-month models, 5 seeds; setting chosen by the frozen rule in `PROTOCOL.md`, addendum 1):**

| Screening threshold τ | F0.5, full router | LSTM share | F0.5, confidence signal only |
|---|---|---|---|
| 0.995 | 0.776 | 99.1% | 0.776 |
| 0.998 | 0.776 | 54.7% | 0.776 |
| 0.999 | 0.776 | 12.3% | 0.776 |
| 0.9995 | 0.752 | 2.3% | 0.752 |
| 0.9999 | 0.752 | 2.1% | 0.730 |

Full-LSTM development F0.5: 0.776. The rule picks the smallest LSTM share within 0.01 of that score: **τ = 0.999**. Novelty and impact add little at τ = 0.999. They matter at the most aggressive setting, τ = 0.9999, where they hold 0.752 against 0.730 for confidence alone.

**Test result (run once, 5 seeds):**

| | Always-on LSTM (B0) | C1 adaptive |
|---|---|---|
| F0.5, all events | 0.8007 | **0.8007** (difference +0.0000, 95% CI [+0.0000, +0.0000]) |
| F0.5, anomalies only | 0.9684 | 0.9684 |
| Share of test data the LSTM runs on | 100% | **6.4%** |
| Measured LSTM inference time (seed 42, GPU) | 844 s | 29 s LSTM + 12 s screening (52 bursts) |
| End-to-end inference speed-up | 1× | **20.3×** |

**Verdict: non-inferior, with a 20-fold cut in inference time.** C1 catches exactly the same events with the same false alarms on every seed, while running the LSTM on 6.4% of the test data. This is the project's first confirmed contribution. The speed-up was measured in a single run on one laptop: both timings come from the same session, but the absolute seconds depend on machine load.

Limits:
- **Unchanged recall:** C1 saves compute but cannot find events the baseline misses.
- **Lookback found on development data:** the 2,100-sample lookback was found after a first design failed on the development period. Both the failure and the fix are recorded in `PROTOCOL.md`.
- **Interrupted run:** the test run was interrupted by a PC shutdown during the timing step and re-executed. The scores are deterministic and identical in both runs (`results/h2_run.log`).

Full numbers: `results/h2_result.json`, `results/c1_dev.json`.

## 14. Update (2026-11-09): sensor reliability (C2, hypotheses H3a and H3b)

**Idea.** A broken sensor and a real spacecraft problem can both make the LSTM raise an alarm, but they need different responses: a maintenance ticket against an incident response. Channels 41-46 form one sensor group, and in the test period 62 of the 64 events with a channel count move all six channels together. C2 asks four questions of every alarm:
- Does one channel disagree with what its five peers predict?
- How many channels are involved?
- Does the signal look stuck, noisy or out of range?

A logistic model turns these features into P(sensor fault). Alarms above the threshold become "sensor fault" and leave the system-alarm stream.

**Data.** ESA-AD has no sensor-fault labels, so 50 single-channel faults were injected into each period, 10 each of stuck, drift, bias, noise and dropout, as fixed in `PROTOCOL.md` addendum 2 before any C2 run. The model and threshold were fitted on the development period only. The resulting threshold is 0.498, with development sensor-fault F1 0.949, and 3 of 114 real-event alarms suppressed.

**Test result (run once, 5 seeds, scored against the real labels only):**

| Condition | Always-on LSTM (B0) | With C2 | Difference (95% CI) | Verdict |
|---|---|---|---|---|
| Test with injected sensor faults (H3a) | 0.341 | **0.628** | +0.288 [+0.223, +0.356] | improvement |
| Clean test, no faults (H3b) | 0.8007 | 0.7987 | -0.0020 [-0.0076, +0.0000] | no harm (non-inferior) |

Without C2, every injected fault the LSTM notices becomes a false system alarm, roughly halving F0.5. With C2 most of those alarms are correctly re-labelled, recovering about 0.29 F0.5. On clean data C2 almost never fires: it re-labelled 1 real-event alarm in one of the five seeds.

**Attribution by fault type (test, 5 seeds × 10 faults each):**

| Fault type | Injected | Raised an LSTM alarm | Correctly attributed to the sensor |
|---|---|---|---|
| stuck | 50 | 27 (54%) | 20 of 27 (74%) |
| drift | 50 | 48 (96%) | 48 of 48 (100%) |
| bias | 50 | 50 (100%) | 50 of 50 (100%) |
| noise | 50 | 50 (100%) | 50 of 50 (100%) |
| dropout | 50 | 50 (100%) | 48 of 50 (96%) |

Bias, noise, drift and dropout faults are almost always caught and attributed. Stuck sensors are the weak spot: frozen values often look normal to the LSTM, so only about half raise an alarm at all.

**Limits.**
- *Injected faults only:* the injected faults come from the same simple generator used for development, so the result is likely optimistic for real sensor failures.
- *Single-channel faults only:* faults hitting several sensors at once were not tested.
- *Favourable group:* C2 relies on correlated peer channels, which this 6-channel group provides. Groups without close peers would be harder.

Full numbers: `results/h3_result.json`, `results/c2_dev.json`.

## 15. Update (2026-11-09): cost-aware response replay (C4, hypothesis H5)

**Idea.** Detecting an alarm is only half the job: someone has to decide what to do with it. Each alarm on the fault-injected test period gets one of three actions. A *page* calls an operator now, a *ticket* opens a maintenance task and a slower response, and *ignore* drops it. Each policy's cost is replayed on the recorded timeline.

This is a counterfactual replay of the **alert policy only**, not of the spacecraft's behaviour. ESA-AD has no cost or response records, so results hold for the declared cost grid in `PROTOCOL.md` addendum 4. The grid has 36 points; the most important parameters are the cost of ignoring an anomaly M, of delaying it D, and of a false page F.

**Policies:**
- **P0:** page every alarm.
- **P1:** use C2's sensor-fault label: ticket sensor faults, page everything else.
- **P2 (proposed):** a learned model estimates whether each alarm is an anomaly, rare event, sensor fault or false alarm. It then picks the cheapest action in expectation.
- **Oracle:** knows the truth.

Test alarms, 5 seeds pooled: 170 anomaly, 20 rare event, 287 sensor fault and 17 false.

| Policy | Cost at the central point (M = 50, D = 5, F = 1, S = 1) | Ranked first across the 36 grid points |
|---|---|---|
| P0 page everything | 64.8 | 0 |
| **P1 C2 sensor rule** | **17.2** | **24** |
| P2 learned cost-aware | 34.9 | 12 |
| Oracle | 0.0 | n/a |

**Verdict: H5 not supported.** P2 beats both P0 and P1 at only 12 of 36 grid points; the protocol required 80%. At the central point P2 costs +17.7 more than P1, with 95% CI [-9.7, +56.6].

P2 wins mainly when delaying a real anomaly is cheap (D = 2: 8 of 12 points) and false pages are expensive (F = 3). It never wins when delay is expensive (D = 10: 0 of 12). It misjudges some real anomalies as ticket-worthy, which is costly whenever delay matters.

**What did work:** the simple C2-driven rule P1 cuts response cost against paging everything at **every** grid point, from 64.8 to 17.2 at the central point. For the response-planning question, a reliable sensor-fault label matters more than a learned cost model. This negative result is useful in its own right: a more complex policy is not automatically better.

Full numbers: `results/h5_result.json`.
