# Test-set log

Every run on the 2007-2013 test set after the protocol freeze (PROTOCOL.md, 2026-11-08) is listed here.

| Date | Commit | Hypothesis | What was run | Result |
|---|---|---|---|---|
| 2026-11-08 | 56ab0a5 | H1 | floor 0.005 vs 0.007, 21 months, 5 seeds | F0.5 0.8182 vs 0.8007, diff +0.0175, 95% CI [-0.0088, +0.0549]: no clear difference |
| 2026-11-09 | e0a12c5 | H2 | C1 router tau 0.999, 21 months, 5 seeds | F0.5 0.8007 vs 0.8007, diff +0.0000, 95% CI [+0.0000, +0.0000], LSTM share 6.4%: non-inferior (first execution was interrupted by a PC shutdown during GPU timing after scoring; the identical deterministic script was re-run to finish) |
