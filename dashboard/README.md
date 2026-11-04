# SatMon research dashboard

An interactive presentation of the ESA-ADB reproduction results, with two views:

- **Mission Control** (`#/mission`): project overview with a 3D Earth, real dataset facts and project status.
- **Research Results** (`#/research`): sections A-H cover the ESA-ADB Mission 1 reproduction (overview, paper vs ours, baseline vs Pruned, all metrics, event-level analysis, seed stability, compute and the proposed contributions). Sections I-J cover the team's Random Forest feature experiments on Mission 2 channel 9: I is Experimental analysis, with six analyses, and J is Improvements achieved. Those use sample-level metrics and are kept apart from A-H.

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
npm run data       # runs ../src/export_dashboard_data.py (sections A-H)
```

Sections I-J read `src/data/channel9.json`, built by `../src/export_channel9_data.py` from `results/channel9_rf/`. That folder holds the teammate's result files, copied unchanged (see `PROVENANCE.md`), plus `team_reported.json` for values supplied from their screenshots: incident counts, model sizes and latency. Those values are marked "team-reported" in the dashboard. The baseline-vs-calibrated threshold comparison and monthly recall are shown as "data not yet provided" until their result files are added.

Run it again after any new experiment. Published values come from Kotowski et al. (2024), Supplementary Table 13 and Table 2. The exact source is noted in `src/export_dashboard_data.py`.

## Notes

- The Earth textures in `public/textures/` are NASA Blue Marble, topology and night-lights imagery (public domain), as distributed with the three-globe examples.
- The orbiting satellite is decorative and labelled as illustrative. No live telemetry or spacecraft positions are shown.
- Contributions C1-C3 are shown as proposed work. No results are claimed for them.
