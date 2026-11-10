# satmon-esa

Trust-aware, resource-efficient satellite health monitoring on the ESA Anomaly Detection Benchmark (ESA-ADB).

The project first reproduces the paper's Telemanom-ESA model on Mission 1 (channels 41-46) with the official metric code: Pruned scores a corrected event-wise F0.5 of 0.801 against the paper's 0.776. It then tests five contributions on top of it, each under a protocol frozen before its test run (`PROTOCOL.md`).

| Contribution | Result on the test period | Status |
|---|---|---|
| C1 adaptive computation | Same F0.5 (0.801), LSTM on 6.4% of the data, 20.3× faster inference | **Confirmed** |
| C2 sensor reliability | F0.5 0.341 → 0.628 with injected sensor faults; no harm on clean data | **Confirmed** (injected faults) |
| C3 incident correlation | Learned graph worse than time-only grouping; time grouping cuts 30.2 alerts per event to 1.48 | Not supported |
| C4 cost-aware response | Learned policy loses to the simple C2 sensor rule, which cuts response cost 64.8 → 17.2 | Not supported |
| C5 ablation | Best combination: C1 + C2 + time grouping + C2 rule | Descriptive |

Full write-up with charts: [RESULTS.md](RESULTS.md). Every test-set run: [results/test_log.md](results/test_log.md).

## What is here

| File | What it does |
|---|---|
| `src/fetch_subset.py` | Downloads only the 6 channels and 4 label files we need (506 MB) from the 3.78 GB Zenodo zip, using range requests. Resumable. |
| `src/prepare.py` | Rebuilds the official 30 s grid, labels and split dates for the 6 channels. Checks the training row count against the repo's `datasets.csv`. |
| `src/train.py` | PyTorch port of the Telemanom-ESA LSTM, with the original settings. Resumes after a shutdown. |
| `src/evaluate.py` | Predicts the 2007-2013 test period, runs the original `errors.py` thresholding unchanged, and scores with the official `ESAScores`. |
| `src/metrics.py` | Loads the official metric file from `baseline/` without the Docker framework. |
| `src/test_metrics.py` | 10 hand-calculated checks on the official metric. |
| `src/smoke_test.py` | Checks training, prediction shapes and thresholding on made-up data. |
| `src/stats.py` | Paired monthly-block bootstrap used for every comparison. |
| `src/c1_router.py`, `c1_dev.py`, `test_h2.py` | C1 cheap screening tier and LSTM bursts; development selection; test. |
| `src/c2_reliability.py`, `c2_run.py` | C2 fault injection, peer-consistency classifier; `dev` and `test` stages. |
| `src/prepare_full.py`, `c3_correlation.py`, `c3_run.py` | C3 on all 76 channels: data preparation, alert grouping, `dev` and `test` stages. |
| `src/c4_policy.py` | C4 page/ticket/ignore replay over the cost grid; `dev` and `test` stages. |
| `src/c5_ablation.py` | C5 ablation of all configurations; `dev` and `test` stages. |
| `src/make_figures.py`, `make_contribution_figures.py` | Charts in `results/figures/` from saved results only. |
| `baseline/` | kplabs-pl/ESA-ADB at commit 67194d3, unmodified (not in git). |

## Run it

Windows, Python 3.13, NVIDIA GPU:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
git clone https://github.com/kplabs-pl/ESA-ADB.git baseline && git -C baseline checkout 67194d3
cd src
python fetch_subset.py
python prepare.py 21_months
python test_metrics.py
python train.py 21_months --seed 42
python evaluate.py 21_months --seed 42
```

Contributions, after the five 21-month seeds are trained and evaluated (`run_seeds.py`). Each `dev` stage only uses the development period; each `test` stage runs once and appends to `results/test_log.md`:

```bash
python dev_tune.py 42 1 2 3 4 --dev-only && python test_h1.py
python c1_dev.py 42 1 2 3 4 && python test_h2.py
python c2_run.py dev && python c2_run.py test
python c4_policy.py dev && python c4_policy.py test
python fetch_subset.py --all && python prepare_full.py && python c3_run.py dev && python c3_run.py test
python c5_ablation.py dev && python c5_ablation.py test
python make_figures.py && python make_contribution_figures.py
```

## Differences from the official setup

- **Framework.** The official model runs on TensorFlow 2.5.1 in Docker, which does not support the RTX 50 series. This port is PyTorch 2.11. It matches the Keras weight initialisation, Adam epsilon, dropout placement, 1000 steps per epoch and early stopping rule. Thresholding uses the original `errors.py` file unchanged.
- **Channels.** Only the 6 lightweight channels are prepared. The time grid is fixed to the official one, and the training row count matches `datasets.csv` (1,840,321 rows).
- **Weights.** The original saves the last epoch's weights: its checkpoint callback saves the best epoch, then `model.save()` overwrites it. `evaluate.py --weights last` is therefore the faithful setting. `--weights best` is also available.
- **Pruned setting.** The repo's `mission1_experiments.py` sets `min_error_value: 0`. The paper's Pruned variant uses 0.007. `evaluate.py` scores both settings from the same predictions.

## P0 result (seed 42, 21 months, last-epoch weights)

Training stopped early at epoch 58 after 634 s on an RTX 5060 Laptop GPU. Test predictions took 404 s, and thresholding plus scoring took about 7 minutes for both settings. The test period has 65 events, 29 of them anomalies, which matches the paper.

| Model | Corrected event-wise F0.5 | Precision | Recall | Alarming precision | Paper (Supp. Table 13, 21 months) |
|---|---|---|---|---|---|
| Telemanom-ESA | 0.094 | 0.076 | 0.908 | 0.855 | 0.122 |
| Telemanom-ESA-Pruned | **0.801** | 0.999 | 0.446 | 0.829 | **0.776** |

Both results fall within the ±0.03 acceptance band, so the reproduction passes G1 for this seed. Anomalies only, Pruned scores F0.5 0.968 (precision 0.999, recall 0.862). One seed is not enough evidence: run seeds 1-4 before using this as the baseline in comparisons.

Full numbers: `results/telemanom_21_months_seed42_last.json`. Trained weights: `models/telemanom_21_months_seed42/last.pt`.

## Baseline across five seeds (21 months, last-epoch weights)

Seeds 42, 1, 2, 3 and 4. Full tables: `results/seed_summary_21_months_last.md`.

| Model | Corrected event-wise F0.5, mean ± SD (range) | Paper |
|---|---|---|
| Telemanom-ESA | 0.105 ± 0.016 (0.084-0.117) | 0.122 |
| Telemanom-ESA-Pruned | **0.801 ± 0.000** (all five 0.801) | 0.776 |

Training stopped between epochs 37 and 58 and took 5.5 to 10.5 minutes per seed. The best validation loss ranged from 0.0849 to 0.0855.

**Pruned gives the same score for every seed.** The five models make different predictions and flag different samples, from 34,033 to 34,658 each. Yet all five catch the same 29 of 65 events, with no false alarms that count against them. The 0.007 `min_error_value` gate decides which events survive, not the trained network. This matters for the novelty work: improvements to the forecaster alone are unlikely to move this score. The thresholding and decision layer is where gains are available. Recall is the clear weakness: 0.446 overall and 0.862 on anomalies, so most missed events are rare events.

The unpruned variant does vary by seed, so comparisons against it need the mean and spread, not one run.

## P1 result: 84 months (five seeds)

| Model | Ours, mean (range) | Paper |
|---|---|---|
| Telemanom-ESA | 0.106 (0.087-0.141) | 0.178 |
| Telemanom-ESA-Pruned | **0.784** (0.773-0.791) | **0.786** |

Pruned passes G1 at both training lengths. Two of the five 84-month seeds raised one false alarm each, which accounts for their 0.773. The full write-up, with charts and related IEEE papers, is in [RESULTS.md](RESULTS.md).

## Dashboard

An interactive presentation of these results (Mission Control plus Research Results, sections A-H) lives in [dashboard/](dashboard/README.md):

```bash
cd dashboard
npm install
npm run dev        # http://localhost:5181
```
