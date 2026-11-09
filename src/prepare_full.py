"""All 76 Mission 1 channels on one 30-second grid, 2000-01-01 to 2014-01-01, for C3.

Same per-channel treatment as prepare.py: monotonic channels 4-11 differenced, labels filled per
row of labels.csv, zero-order hold. Written as memory-mapped arrays so later steps can slice periods
without loading 14 years of 76 channels into memory.
Resumable: channels already written are skipped (tracked in data/processed/full/done.txt).

Usage: python src/prepare_full.py
Output: data/processed/full/{values.f32, labels.u8, ids.u16, index.npy, channels.json, id_map.json}
"""
import json
import time

import numpy as np
import pandas as pd

from prepare import ANOMALY, GAP, RARE_EVENT, RAW, RULE

ROOT = RAW.parents[2]
OUT = ROOT / "data" / "processed" / "full"
START, END = "2000-01-01", "2014-01-01"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = pd.date_range(START, END, freq=RULE)
    channels = pd.read_csv(RAW / "channels.csv")["Channel"].tolist()
    labels = pd.read_csv(RAW / "labels.csv")
    for col in ["StartTime", "EndTime"]:
        labels[col] = pd.to_datetime(labels[col], utc=True).dt.tz_localize(None)
    kinds = dict(pd.read_csv(RAW / "anomaly_types.csv")[["ID", "Category"]].values)
    id_map = {eid: i + 1 for i, eid in enumerate(sorted(labels["ID"].unique(), key=lambda s: int(s.split("_")[1])))}
    np.save(OUT / "index.npy", index.values)
    (OUT / "channels.json").write_text(json.dumps(channels))
    (OUT / "id_map.json").write_text(json.dumps(id_map))

    shape = (len(index), len(channels))
    mode = "r+" if (OUT / "values.f32").exists() else "w+"
    values = np.memmap(OUT / "values.f32", np.float32, mode, shape=shape)
    flags = np.memmap(OUT / "labels.u8", np.uint8, mode, shape=shape)
    ids = np.memmap(OUT / "ids.u16", np.uint16, mode, shape=shape)
    done_path = OUT / "done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()

    for c, name in enumerate(channels):
        if name in done:
            continue
        t0 = time.time()
        df = pd.read_pickle(RAW / "channels" / f"{name}.zip").rename(columns={name: "value"})
        if 4 <= int(name.split("_")[1]) <= 11:  # monotonic counters are differenced, as in the official script
            df["value"] = np.diff(df["value"], append=df["value"].iloc[-1])
        df["label"] = np.uint8(0)
        df["id"] = np.uint16(0)
        for row in labels[labels["Channel"] == name].itertuples():
            kind = kinds[row.ID]
            df.loc[row.StartTime:row.EndTime, "label"] = ANOMALY if kind == "Anomaly" else RARE_EVENT if kind == "Rare Event" else GAP
            df.loc[row.StartTime:row.EndTime, "id"] = id_map[row.ID]
        grid = df.reindex(index, method="ffill")
        values[:, c] = grid["value"].ffill().bfill().to_numpy(np.float32)
        # labels and ids on the grid: any labelled raw sample inside a 30 s bin marks that bin
        binned = df[df["label"] > 0].groupby(df[df["label"] > 0].index.ceil(RULE))[["label", "id"]].max()
        lab = np.zeros(len(index), np.uint8)
        idv = np.zeros(len(index), np.uint16)
        pos = index.get_indexer(binned.index)
        ok = pos >= 0
        lab[pos[ok]] = binned["label"].to_numpy()[ok]
        idv[pos[ok]] = binned["id"].to_numpy()[ok]
        # fill labels across the labelled spans themselves (zero-order hold of the label column)
        held = grid["label"].fillna(0).to_numpy(np.uint8)
        held_id = grid["id"].fillna(0).to_numpy(np.uint16)
        flags[:, c] = np.maximum(lab, held)
        ids[:, c] = np.where(idv > 0, idv, held_id)
        values.flush(), flags.flush(), ids.flush()
        done.add(name)
        done_path.write_text("\n".join(sorted(done)))
        print(f"{name}: {len(df):,} raw samples, {int((flags[:, c] > 0).sum()):,} labelled bins [{time.time() - t0:.0f}s]", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
