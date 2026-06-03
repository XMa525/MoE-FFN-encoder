#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, List


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def find_metric_from_dict(d: Dict, keys: List[str]) -> Optional[float]:
    """
    从 dict 中按照多个候选 key 查找 metric。
    兼容 test_auc / auc / balanced_auc / val_auc 等命名。
    """
    lower_map = {str(k).lower(): v for k, v in d.items()}

    for key in keys:
        key = key.lower()
        if key in lower_map:
            val = parse_float(lower_map[key])
            if val is not None:
                return val

    # 模糊匹配
    for k, v in lower_map.items():
        for key in keys:
            if key.lower() in k:
                val = parse_float(v)
                if val is not None:
                    return val

    return None


def parse_json_file(path: Path) -> Optional[Dict[str, float]]:
    try:
        with open(path, "r") as f:
            obj = json.load(f)
    except Exception:
        return None

    if isinstance(obj, list):
        if len(obj) == 0:
            return None
        # 默认取最后一个 epoch / 最后一条记录
        obj = obj[-1]

    if not isinstance(obj, dict):
        return None

    auc = find_metric_from_dict(
        obj,
        [
            "test_auc", "final_auc", "best_auc", "balanced_auc",
            "auc", "val_auc", "balanced_early_auc"
        ],
    )
    acc = find_metric_from_dict(
        obj,
        [
            "test_acc", "final_acc", "best_acc",
            "acc", "accuracy", "val_acc", "val_accuracy"
        ],
    )
    f1 = find_metric_from_dict(
        obj,
        [
            "test_f1", "final_f1", "best_f1",
            "f1", "macro_f1", "val_f1"
        ],
    )

    if auc is None and acc is None and f1 is None:
        return None

    return {"auc": auc, "acc": acc, "f1": f1}


def parse_csv_file(path: Path) -> Optional[Dict[str, float]]:
    try:
        with open(path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    if not rows:
        return None

    # 默认取最后一行；如果有 best / test 这种 summary csv，通常也只有一行
    row = rows[-1]

    auc = find_metric_from_dict(
        row,
        [
            "test_auc", "auc_test"
        ],
    )
    acc = find_metric_from_dict(
        row,
        [
            "test_acc", "acc_test"
        ],
    )
    f1 = find_metric_from_dict(
        row,
        [
            "test_f1", "f1_test"
        ],
    )

    if auc is None and acc is None and f1 is None:
        return None

    return {"auc": auc, "acc": acc, "f1": f1}


def parse_log_file(path: Path) -> Optional[Dict[str, float]]:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None

    # 尽量匹配常见格式：
    # auc: 0.83 / AUC=0.83 / test_auc 0.83
    patterns = {
        "auc": [
            r"test[_\s-]*auc\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"auc[_\s-]*test\s*[:=]\s*([0-9]*\.?[0-9]+)",
        ],
        "acc": [
            r"test[_\s-]*acc\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"test[_\s-]*accuracy\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"acc[_\s-]*test\s*[:=]\s*([0-9]*\.?[0-9]+)",
        ],
        "f1": [
            r"test[_\s-]*f1\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"f1[_\s-]*test\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"test[_\s-]*macro[_\s-]*f1\s*[:=]\s*([0-9]*\.?[0-9]+)",
        ],
    }

    result = {}
    for metric, pats in patterns.items():
        values = []
        for pat in pats:
            values += [parse_float(x) for x in re.findall(pat, text, flags=re.IGNORECASE)]
        values = [x for x in values if x is not None]
        result[metric] = values[-1] if values else None

    if result["auc"] is None and result["acc"] is None and result["f1"] is None:
        return None

    return result

def parse_json_file_strict_test(path: Path) -> Optional[Dict[str, float]]:
    try:
        with open(path, "r") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"[parse error] cannot read json: {path}, error={e}")
        return None

    if isinstance(obj, list):
        if len(obj) == 0:
            print(f"[parse error] empty list json: {path}")
            return None
        obj = obj[-1]

    if not isinstance(obj, dict):
        print(f"[parse error] json is not dict/list[dict]: {path}, type={type(obj)}")
        return None

    print(f"[debug] parsing {path}")
    print(f"[debug] available keys: {list(obj.keys())}")

    # 关键修改：优先读取 final_test_metrics.json 里的 test_metrics 子字段
    if "test_metrics" in obj and isinstance(obj["test_metrics"], dict):
        metric_obj = obj["test_metrics"]
        print(f"[debug] using nested test_metrics keys: {list(metric_obj.keys())}")
    else:
        metric_obj = obj
        print("[debug] using top-level metrics")

    auc = find_metric_from_dict(
        metric_obj,
        [
            "test_auc",
            "auc_test",
            "final_test_auc",
            "auc",
            "balanced_auc",
            "roc_auc",
        ],
    )

    acc = find_metric_from_dict(
        metric_obj,
        [
            "test_acc",
            "acc_test",
            "test_accuracy",
            "accuracy_test",
            "final_test_acc",
            "acc",
            "accuracy",
        ],
    )

    f1 = find_metric_from_dict(
        metric_obj,
        [
            "test_f1",
            "f1_test",
            "test_macro_f1",
            "macro_f1_test",
            "final_test_f1",
            "f1",
            "macro_f1",
        ],
    )

    if auc is None or acc is None or f1 is None:
        print(
            f"[parse error] missing metric in {path}: "
            f"auc={auc}, acc={acc}, f1={f1}"
        )
        return None

    return {
        "auc": auc,
        "acc": acc,
        "f1": f1,
    }

def collect_metrics(seed_out_dir: Path, log_path: Path) -> Dict[str, Optional[float]]:
    """
    严格只读取 seed_out_dir/test_metrics.json。
    不做任何 json/csv 兜底，避免误读 best_epoch_selection.json。
    """
    test_path = seed_out_dir / "final_test_metrics.json"

    if not test_path.exists():
        return {
            "auc": None,
            "acc": None,
            "f1": None,
            "source": f"missing: {test_path}",
        }

    parsed = parse_json_file_strict_test(test_path)

    if parsed is None:
        return {
            "auc": None,
            "acc": None,
            "f1": None,
            "source": f"parse_failed: {test_path}",
        }

    parsed["source"] = str(test_path)
    return parsed


def safe_metric(x: Optional[float]) -> float:
    return float(x) if x is not None else 0.0


#def composite_score(auc, acc, f1, auc_w=0.5, acc_w=0.25, f1_w=0.25):
def composite_score(auc, acc, f1, auc_w=0.6, acc_w=0.4, f1_w=0.0):
    return (
        auc_w * safe_metric(auc)
        + acc_w * safe_metric(acc)
        + f1_w * safe_metric(f1)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="1-30",
                        help="Seed 列表，例如 1-50 或 1,2,3,7,11")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--base_out_dir", type=str,
                        default="./results/downstream/BRACS/abmil_dino_moe_transfer_v1")
    parser.add_argument("--skip_existing", action="store_true",
                        help="如果 seed 目录已存在 train.log，则跳过训练，只重新解析结果")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if "-" in args.seeds:
        start, end = args.seeds.split("-")
        seeds = list(range(int(start), int(end) + 1))
    else:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    base_out_dir = Path(args.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in seeds:
        seed_out_dir = base_out_dir / f"seed_{seed}"
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        log_path = seed_out_dir / "train.log"

        cmd = [
            "python", "downstream/train_abmil.py",
            "--slides_csv", "../data/BRACS/bracs_split.csv",
            "--feature_dir", "features/BRACS/dino-moe/pt_files_transfer_v1",
            "--out_dir", str(seed_out_dir),
            "--mil_model", "abmil",
            "--att_dim", "64",
            "--lr", "1e-4",
            "--weight_decay", "1e-4",
            "--shuffle_instances",
            "--epochs", "15",
            "--patience", "10",
            "--batch_size", "1",
            "--num_workers", "8",
            "--monitor", "balanced_early_auc",
            "--early_auc_tol", "0.03",
            "--min_val_sens", "0.30",
            "--min_val_spec", "0.60",
            "--min_val_f1", "1e-8",
            "--min_select_epoch", "8",
            "--pos_weight","1.7",
            "--seed", str(seed),
        ]
        # cmd = [
        #     "python", "downstream/train_abmil.py",
        #     "--slides_csv", "../data/Parotid/parotid_split_seed8.csv",
        #     "--feature_dir", "features/parotid_openclip_b16_moe_feats/v7_adapt_427",
        #     "--out_dir", str(seed_out_dir),
        #     "--mil_model", "transmil",
        #     "--transmil_input_proj_dim", "1024",
        #     "--lr", "5e-5",
        #     "--weight_decay", "1e-5",
        #     "--shuffle_instances",
        #     "--max_instances","512" ,
        #     "--epochs", "25",
        #     "--patience", "10",
        #     "--batch_size", "1",
        #     "--num_workers", "8",
        #     "--monitor", "balanced_early_auc",
        #     "--early_auc_tol", "0.03",
        #     "--min_val_sens", "0.30",
        #     "--min_val_spec", "0.60",
        #     "--min_val_f1", "1e-8",
        #     "--min_select_epoch", "8",
        #     #"--monitor", "auc",
        #     "--seed", str(seed),
        # ]
        # cmd = [
        #     "python", "downstream/train_abmil.py",
        #     "--slides_csv", "../data/Parotid/parotid_split_seed8.csv",
        #     "--feature_dir", "features/parotid_openclip_b16_moe_feats/v7_adapt_427",
        #     "--out_dir", str(seed_out_dir),
        #     "--mil_model", "abmil",
        #     "--att_dim", "128",
        #     "--lr", "5e-4",
        #     "--weight_decay", "1e-3",
        #     "--shuffle_instances",
        #     "--epochs", "20",
        #     "--patience", "20",
        #     "--batch_size", "1",
        #     "--num_workers", "8",
        #     "--monitor", "auc",
        #     "--seed", str(seed),
        # ]

        print("=" * 80)
        print(f"[Seed {seed}] out_dir = {seed_out_dir}")

        if args.dry_run:
            print(" ".join(cmd))
            continue

        if args.skip_existing and log_path.exists():
            print(f"[Seed {seed}] skip existing training, parse previous result.")
        else:
            print(f"[Seed {seed}] running...")
            with open(log_path, "w") as f:
                proc = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            if proc.returncode != 0:
                print(f"[Seed {seed}] failed. Check log: {log_path}")
                all_results.append({
                    "seed": seed,
                    "auc": None,
                    "acc": None,
                    "f1": None,
                    "score": 0.0,
                    "source": str(log_path),
                    "status": "failed",
                })
                continue

        metrics = collect_metrics(seed_out_dir, log_path)
        auc = metrics.get("auc")
        acc = metrics.get("acc")
        f1 = metrics.get("f1")
        score = composite_score(auc, acc, f1)

        result = {
            "seed": seed,
            "auc": auc,
            "acc": acc,
            "f1": f1,
            "score": score,
            "source": metrics.get("source"),
            "status": "ok",
        }
        all_results.append(result)

        print(
            f"[Seed {seed}] "
            f"auc={auc}, acc={acc}, f1={f1}, score={score:.6f}, "
            f"source={metrics.get('source')}"
        )

    all_results = sorted(
        all_results,
        key=lambda x: (
            x["score"],
            safe_metric(x["auc"]),
            safe_metric(x["f1"]),
            safe_metric(x["acc"]),
        ),
        reverse=True,
    )

    result_csv = base_out_dir / "multi_seed_summary.csv"
    with open(result_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "seed", "auc", "acc", "f1", "score", "status", "source"]
        )
        writer.writeheader()
        for i, r in enumerate(all_results, start=1):
            writer.writerow({
                "rank": i,
                "seed": r["seed"],
                "auc": r["auc"],
                "acc": r["acc"],
                "f1": r["f1"],
                "score": r["score"],
                "status": r["status"],
                "source": r["source"],
            })

    print("\n" + "=" * 80)
    print(f"Saved summary to: {result_csv}")
    print(f"Top {args.topk} seeds:")
    print("-" * 80)

    for i, r in enumerate(all_results[:args.topk], start=1):
        print(
            f"Rank {i}: "
            f"seed={r['seed']}, "
            f"auc={r['auc']}, "
            f"acc={r['acc']}, "
            f"f1={r['f1']}, "
            f"score={r['score']:.6f}, "
            f"status={r['status']}"
        )


if __name__ == "__main__":
    main()