# Satellite Telemetry Anomaly Detection: Reproducing Telemanom-ESA on ESA-ADB

## 1. Purpose

This experiment trains the Telemanom-ESA anomaly detector on the ESA Anomaly Detection Benchmark (ESA-ADB) and compares it with the published results.

The aim is a verified baseline. Our later contributions (adaptive computation, sensor reliability, incident correlation and response planning) will be measured against it.

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
