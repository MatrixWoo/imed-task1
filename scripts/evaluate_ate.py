from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.pose_metrics import (
    RPE_DELTAS,
    evaluate_split,
    format_rpe_lines,
    format_sequence_line,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pose.txt predictions (Endomapper-style metrics).")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON results path")
    args = parser.parse_args()

    start = time.perf_counter()
    per_seq, global_mean = evaluate_split(args.data_root, args.pred_root, args.split)

    for seq_name, result in per_seq.items():
        print(format_sequence_line(seq_name, result))
        for line in format_rpe_lines(result):
            print(line)

    print("\n" + "=" * 60)
    print("GLOBAL MEAN")
    print(
        f"  mean_ate={global_mean.get('mean_ate', float('nan')):.4f} mm  "
        f"(over {int(global_mean.get('num_sequences', 0))} sequences)"
    )
    print(f"  median_ate (mean of per-seq medians)={global_mean.get('median_ate', float('nan')):.4f} mm")
    for delta in RPE_DELTAS:
        print(
            f"  RPE d={delta:2d}: trans={global_mean.get(f'trans_rpe_{delta}frame', float('nan')):.4f} mm  "
            f"rot={global_mean.get(f'rot_rpe_deg_{delta}frame', float('nan')):.4f} deg"
        )
    print(f"\nEvaluation runtime: {time.perf_counter() - start:.3f}s")

    if args.json_out is not None:
        payload = {
            "split": args.split,
            "data_root": str(args.data_root),
            "pred_root": str(args.pred_root),
            "per_sequence": per_seq,
            "global_mean": global_mean,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON results to: {args.json_out}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"per_sequence": per_seq, "aggregate": aggregate}, indent=2))
        print(f"Wrote metrics to: {args.json_out}")


if __name__ == "__main__":
    main()