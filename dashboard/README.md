# SatMon research dashboard

An interactive presentation of the ESA-ADB reproduction results, with two views:

- **Mission Control** (`#/mission`): project overview with a 3D Earth, real dataset facts and project status.
- **Research Results** (`#/research`): sections A-G cover the ESA-ADB Mission 1 reproduction (overview, paper vs ours, baseline vs Pruned, all metrics, event-level analysis, seed stability, compute). Sections H-N cover our contributions, each tested once on the test set: H is the overview with every verdict and the calibrated floor (H1), then C1 adaptive computation (I), C2 sensor reliability (J), C3 incident correlation (K), C4 cost-aware response (L), C5 ablation (M) and rare events (N). The team's separate Random Forest feature experiments on Mission 2 channel 9 have their own dashboard, outside this repository.

Built with React 18, TypeScript, Vite, Tailwind CSS, Recharts, three.js and Lucide icons.

## Run it

```bash
cd dashboard
npm install
npm run dev        # http://localhost:5181
```

For a static build: `npm run build`, then `npm run preview`.

## Where the numbers come from

All values are read from `src/data/results.json`. Nothing is typed into the components. That file is generated from the saved experiment outputs (`results/*.json`, `models/*/train_summary.json` and `history.json`, the evaluation logs, `results/figures/tables.json` and `results/diagnose_unpruned.csv`).

```bash
npm run data       # runs ../src/export_dashboard_data.py (sections A-G) and ../src/export_contributions_data.py (sections H-N)
```

Sections H-N read `src/data/contributions.json`, built from the test and development result files (`results/h1_result.json` ... `h6_result.json`, `c5_result.json` and the development JSONs).

Run it again after any new experiment. Published values come from Kotowski et al. (2024), Supplementary Table 13 and Table 2. The exact source is noted in `src/export_dashboard_data.py`.

## Notes

- The Earth textures in `public/textures/` are NASA Blue Marble, topology and night-lights imagery (public domain), as distributed with the three-globe examples.
- The orbiting satellite is decorative and labelled as illustrative. No live telemetry or spacecraft positions are shown.
- Every contribution result shows its verdict under the frozen protocol (`../PROTOCOL.md`): confirmed, not confirmed, not supported or descriptive. Negative results are shown alongside the positive ones.
