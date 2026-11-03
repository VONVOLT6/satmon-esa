"""Build train/test arrays for Mission 1 channels 41-46.

Follows notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py from the
ESA-ADB repo: 30 s zero-order hold, annotated samples restored after resampling,
same split dates. The only difference is that it keeps the six channels instead
of all 87, so the time grid is forced to the official one rather than derived
from channels we did not download.

Usage: python src/prepare.py 21_months
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_date

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ESA-Mission1"
OUT = ROOT / "data" / "processed"
CHANNELS = [f"channel_{i}" for i in range(41, 47)]
RULE = pd.Timedelta(seconds=30)
MISSION_START = "2000-01-01"
TEST_SPLIT = "2007-01-01"
SPLITS = {"3_months": "2000-04-01", "10_months": "2000-11-01", "21_months": "2001-10-01",
          "42_months": "2003-07-01", "84_months": "2007-01-01"}
# rows in the repo's own datasets.csv, used as a check on the grid
OFFICIAL_TRAIN_LENGTH = {"3_months": 262081, "10_months": 878401, "21_months": 1840321,
                         "42_months": 3677761, "84_months": 7364161}
NOMINAL, ANOMALY, RARE_EVENT, GAP = 0, 1, 2, 3


def load_labels():
    labels = pd.read_csv(RAW / "labels.csv")
    for col in ["StartTime", "EndTime"]:
        labels[col] = pd.to_datetime(labels[col], utc=True).dt.tz_localize(None)
    types = pd.read_csv(RAW / "anomaly_types.csv")
    return labels, types


def load_channel(param, labels, types, split_kind, split_at):
    df = pd.read_pickle(RAW / "channels" / f"{param}.zip")
    df = df.rename(columns={param: "value"})
    df["label"] = np.uint8(NOMINAL)

    category = dict(zip(types["ID"], types["Category"]))
    annotated = False
    for _, row in labels[labels["Channel"] == param].iterrows():
        kind = category[row["ID"]]
        value = ANOMALY if kind == "Anomaly" else RARE_EVENT if kind == "Rare Event" else GAP
        df.loc[row["StartTime"]:row["EndTime"], "label"] = value
        annotated = True

    if split_kind == "train":
        df = df[df.index <= parse_date(split_at)].copy()
    else:
        df = df[df.index > parse_date(TEST_SPLIT)].copy()

    first = pd.Timestamp(df.index[0]).floor(freq=RULE)
    last = pd.Timestamp(df.index[-1]).ceil(freq=RULE)
    grid = pd.date_range(first, last, freq=RULE)
    out = df.reindex(grid, method="ffill")
    out.iloc[0] = df.iloc[0]

    if annotated:
        for timestamp, group in df.groupby(pd.Grouper(freq=RULE)).indices.items():
            if len(group) <= 1:
                continue
            org = df.iloc[group]
            if org["label"].values[-1] != NOMINAL:
                continue
            mask = (org["label"] == ANOMALY) | (org["label"] == RARE_EVENT)
            if mask.any():
                out.loc[timestamp + RULE] = org[mask].iloc[-1]
    return out


def build(split_kind, split_at, start, end):
    labels, types = load_labels()
    index = pd.date_range(start, end, freq=RULE)
    values = np.zeros((len(index), len(CHANNELS)), dtype=np.float32)
    flags = np.zeros((len(index), len(CHANNELS)), dtype=np.uint8)
    for c, param in enumerate(CHANNELS):
        df = load_channel(param, labels, types, split_kind, split_at).reindex(index)
        values[:, c] = df["value"].ffill().bfill().to_numpy(dtype=np.float32)
        flags[:, c] = df["label"].ffill().bfill().to_numpy(dtype=np.uint8)
        print(f"  {split_kind} {param}: {np.count_nonzero(flags[:, c])} labelled samples", flush=True)
    return index, values, flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(SPLITS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    split_at = SPLITS[args.dataset]
    index, values, flags = build("train", split_at, MISSION_START, split_at)
    expected = OFFICIAL_TRAIN_LENGTH[args.dataset]
    print(f"train rows {len(index)}, official {expected}: {'match' if len(index) == expected else 'DIFFERENT'}")
    np.savez(OUT / f"{args.dataset}.train.npz", index=index.values, values=values, flags=flags, channels=CHANNELS)

    test_path = OUT / "test.npz"
    if not test_path.exists():
        ends = []
        for param in CHANNELS:
            ends.append(pd.read_pickle(RAW / "channels" / f"{param}.zip").index[-1])
        end = pd.Timestamp(max(ends)).ceil(freq=RULE)
        start = pd.Timestamp(TEST_SPLIT)
        index, values, flags = build("test", split_at, start, end)
        print(f"test rows {len(index)}, {index[0]} to {index[-1]}")
        np.savez(test_path, index=index.values, values=values, flags=flags, channels=CHANNELS)


if __name__ == "__main__":
    main()
