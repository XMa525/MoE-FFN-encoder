import argparse
import ast
import csv
import os
import re
from typing import Dict, List, Optional


def safe_literal_dict(s: str) -> Optional[dict]:
    s = s.strip()
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


TRAIN_DETAIL_RE = re.compile(r"^\[Train\]\s*Epoch\s+(\d+)\s+Loss detail:\s*(\{.*\})\s*$")
VAL_DETAIL_RE = re.compile(r"^\[Val\]\s*Epoch\s+(\d+)\s+Loss detail:\s*(\{.*\})\s*$")
SUMMARY_RE = re.compile(
    r"^Epoch\s*\[(\d+)/(\d+)\]\s*\|\s*Train Loss:\s*([-+eE0-9\.]+)\s*\|\s*Val Loss:\s*([-+eE0-9\.]+)\s*$"
)
TRAIN_WSI_RE = re.compile(
    r"^\[Train\]\[WSI\]\s*bag_total=([-+eE0-9\.]+)\s*\|\s*bag_bce=([-+eE0-9\.]+)\s*\|\s*bag_margin=([-+eE0-9\.]+)\s*\|\s*topk_mean=([-+eE0-9\.]+)\s*\|\s*prob=([-+eE0-9\.]+)\s*$"
)
VAL_WSI_RE = re.compile(
    r"^\[Val\]\[WSI\]\s*bag_total=([-+eE0-9\.]+)\s*\|\s*bag_bce=([-+eE0-9\.]+)\s*\|\s*bag_margin=([-+eE0-9\.]+)\s*\|\s*topk_mean=([-+eE0-9\.]+)\s*\|\s*prob=([-+eE0-9\.]+)\s*$"
)


def ensure_epoch(store: Dict[int, dict], epoch: int) -> dict:
    if epoch not in store:
        store[epoch] = {"epoch": epoch}
    return store[epoch]


FLOAT_KEYS_WSI = ["bag_total", "bag_bce", "bag_margin", "topk_mean", "prob"]


def maybe_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return v
    return v



def flatten_metrics(prefix: str, metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        out[f"{prefix}_{k}"] = maybe_float(v)
    return out



def parse_log(log_path: str) -> List[dict]:
    epochs: Dict[int, dict] = {}
    current_train_epoch: Optional[int] = None
    current_val_epoch: Optional[int] = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m = TRAIN_DETAIL_RE.match(line)
            if m:
                ep = int(m.group(1))
                metrics = safe_literal_dict(m.group(2))
                row = ensure_epoch(epochs, ep)
                if metrics is not None:
                    row.update(flatten_metrics("train", metrics))
                current_train_epoch = ep
                continue

            m = VAL_DETAIL_RE.match(line)
            if m:
                ep = int(m.group(1))
                metrics = safe_literal_dict(m.group(2))
                row = ensure_epoch(epochs, ep)
                if metrics is not None:
                    row.update(flatten_metrics("val", metrics))
                current_val_epoch = ep
                continue

            m = SUMMARY_RE.match(line)
            if m:
                ep = int(m.group(1))
                total_epochs = int(m.group(2))
                row = ensure_epoch(epochs, ep)
                row["epoch_total"] = total_epochs
                row["summary_train_loss"] = float(m.group(3))
                row["summary_val_loss"] = float(m.group(4))
                continue

            m = TRAIN_WSI_RE.match(line)
            if m and current_train_epoch is not None:
                row = ensure_epoch(epochs, current_train_epoch)
                for key, val in zip(FLOAT_KEYS_WSI, m.groups()):
                    row[f"train_wsi_line_{key}"] = float(val)
                continue

            m = VAL_WSI_RE.match(line)
            if m and current_val_epoch is not None:
                row = ensure_epoch(epochs, current_val_epoch)
                for key, val in zip(FLOAT_KEYS_WSI, m.groups()):
                    row[f"val_wsi_line_{key}"] = float(val)
                continue

    rows = [epochs[k] for k in sorted(epochs.keys())]
    return rows



def write_csv(rows: List[dict], out_csv: str):
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    preferred_front = [
        "epoch",
        "epoch_total",
        "summary_train_loss",
        "summary_val_loss",
        "train_total_loss",
        "val_total_loss",
    ]
    rest = sorted(k for k in all_keys if k not in preferred_front)
    fieldnames = [k for k in preferred_front if k in all_keys] + rest

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)



def write_summary(rows: List[dict], out_txt: str):
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"num_epochs_parsed: {len(rows)}\n")
        if not rows:
            return
        f.write(f"epochs: {[r['epoch'] for r in rows]}\n\n")
        for r in rows:
            ep = r.get("epoch")
            train_loss = r.get("train_total_loss", r.get("summary_train_loss", None))
            val_loss = r.get("val_total_loss", r.get("summary_val_loss", None))
            f.write(f"epoch {ep}: train_total_loss={train_loss}, val_total_loss={val_loss}\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to stage2 training .log file")
    parser.add_argument("--out-csv", required=True, help="Output epoch metrics csv")
    parser.add_argument("--out-summary", default=None, help="Optional summary txt")
    args = parser.parse_args()

    rows = parse_log(args.log)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    write_csv(rows, args.out_csv)

    out_summary = args.out_summary
    if out_summary is None:
        root, _ = os.path.splitext(args.out_csv)
        out_summary = root + "_summary.txt"
    write_summary(rows, out_summary)

    print(f"[Saved] {args.out_csv}")
    print(f"[Saved] {out_summary}")
    print(f"[Info] Parsed epochs: {[r['epoch'] for r in rows]}")


if __name__ == "__main__":
    main()
