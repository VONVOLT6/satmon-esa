# Test-set log

Every run on the 2007-2013 test set after the protocol freeze (PROTOCOL.md, 2026-11-08) is listed here.

| Date | Commit | Hypothesis | What was run | Result |
|---|---|---|---|---|
| 2026-11-08 | 56ab0a5 | H1 | floor 0.005 vs 0.007, 21 months, 5 seeds | F0.5 0.8182 vs 0.8007, diff +0.0175, 95% CI [-0.0088, +0.0549]: no clear difference |
| 2026-11-09 | e0a12c5 | H2 | C1 router tau 0.999, 21 months, 5 seeds | F0.5 0.8007 vs 0.8007, diff +0.0000, 95% CI [+0.0000, +0.0000], LSTM share 6.4%: non-inferior (first execution was interrupted by a PC shutdown during GPU timing after scoring; the identical deterministic script was re-run to finish) |
| 2026-11-09 | 83bd34c | H3a | C2 reliability, injected test, 21 months, 5 seeds | F0.5 0.6283 vs 0.3405, diff +0.2877, 95% CI [+0.2225, +0.3557]: improvement |
| 2026-11-09 | 83bd34c | H3b | C2 reliability, clean test, 21 months, 5 seeds | F0.5 0.7987 vs 0.8007, diff -0.0020, 95% CI [-0.0076, +0.0000]: no harm (non-inferior) |
| 2026-11-09 | 449659d | H5 | C4 cost-aware policy replay on injected test, 36-point cost grid | P2 lower regret than P0 and P1 at 12/36 points; central P2-P1 cost +17.7 [-9.7, +56.6]: not supported |
| 2026-11-10 | d2723a1 | H4 | C3 graph grouping vs time-only grouping, all 76 channels | pairwise F1 0.406 vs 0.978, diff -0.572, 95% CI [-0.576, -0.179]: worse |
| 2026-11-10 | 11dfb6e | C5 (descriptive) | ablation A0-A8, L1-L4 on injected and clean test, 5 seeds | A8 injected F0.5 0.6283 vs A0 0.3405, diff +0.2877 [+0.2225, +0.3557]; A8 LSTM share 11.9%, central cost 34.9 vs 64.8 (first execution stopped by a PC sleep/shutdown during seed 42, before any summary; the identical deterministic script resumed from its cache and reproduced seed 42's scores exactly) |
