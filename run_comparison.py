"""Comparison experiment: SVD-LLM vs Dobi-SVD vs ours, on LLaMA-7B / wikitext-2.

Each (method, ratio) cell runs as its own subprocess so a 7B model is fully torn
down between cells, and each writes a JSON result that this script aggregates
into a CSV and a markdown table. Finished cells are skipped on re-run, so an
interrupted sweep can simply be relaunched (pass `--overwrite` to force).

Example
-------
  python run_comparison.py --ratios 0.9,0.8,0.7 --methods dense,svdllm,dobisvd,ours
  python run_comparison.py --ratios 0.8 --methods ours --extra_ours "--stage2_epochs 4"
"""
import argparse
import csv
import glob
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["dense", "svdllm", "dobisvd", "ours"]
PRETTY = {"dense": "Dense (uncompressed)", "svdllm": "SVD-LLM",
          "dobisvd": "Dobi-SVD", "ours": "Ours (sliding-window)"}


def cell_paths(out_dir, method, ratio):
    tag = f"{method}_r{ratio:g}" if method != "dense" else "dense"
    return (os.path.join(out_dir, f"{tag}.json"),
            os.path.join(out_dir, f"{tag}.log"))


def build_cmd(args, method, ratio, out_json):
    common = [
        "--model_id", args.model_id,
        "--dtype", args.dtype,
        "--seqlen", str(args.seqlen),
        "--device", args.device,
        "--seed", str(args.seed),
        "--data_root", args.data_root,
        "--calib_dataset", args.calib_dataset,
        "--eval_datasets", args.eval_datasets,
        "--nsamples", str(args.nsamples),
        "--out_json", out_json,
    ]
    if method == "dense":
        return [sys.executable, "-u", os.path.join(HERE, "eval_dense.py")] + common
    if method == "ours":
        cmd = [sys.executable, "-u", os.path.join(HERE, "sliding_llm.py")] + common
        cmd += ["--ratio", str(ratio)] + shlex.split(args.extra_ours)
        return cmd
    cmd = [sys.executable, "-u", os.path.join(HERE, "baselines.py"), "--method", method] + common
    cmd += ["--ratio", str(ratio)]
    cmd += shlex.split(args.extra_svdllm if method == "svdllm" else args.extra_dobisvd)
    return cmd


def run_cell(cmd, log_path, dry_run=False):
    print("\n" + "=" * 78)
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    print("=" * 78, flush=True)
    if dry_run:
        return True
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    ok = proc.returncode == 0
    print(f"[cell] {'ok' if ok else 'FAILED (rc=%d)' % proc.returncode} "
          f"in {time.time()-t0:.0f}s — log: {log_path}", flush=True)
    return ok


def collect(out_dir, eval_datasets):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        if os.path.basename(path) in ("summary.json",):
            continue
        with open(path) as f:
            r = json.load(f)
        row = {
            "method": r.get("method", "?"),
            "model": r.get("model_id", ""),
            "target_ratio": r.get("target_ratio", 1.0),
            "achieved_linear_ratio": round(r.get("linear_param_ratio", 1.0), 4),
            "linear_params_B": round(r.get("linear_params", 0) / 1e9, 3),
            "total_params_B": round(r.get("total_params", 0) / 1e9, 3),
            "runtime_sec": round(sum(r.get("timings_sec", {}).values()), 1),
        }
        for ds in eval_datasets:
            v = r.get("ppl", {}).get(ds)
            row[f"ppl_{ds}"] = round(v, 4) if isinstance(v, (int, float)) else ""
        rows.append(row)

    order = {m: i for i, m in enumerate(METHODS)}
    rows.sort(key=lambda r: (-r["target_ratio"], order.get(r["method"], 99)))
    return rows


def write_outputs(out_dir, rows, eval_datasets):
    if not rows:
        print("[summary] no results yet")
        return
    fields = list(rows[0].keys())
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ppl_cols = [f"ppl_{d}" for d in eval_datasets]
    head = ["Method", "Target ratio", "Achieved ratio", "Linear params (B)"] + \
           [f"PPL {d}" for d in eval_datasets] + ["Runtime (s)"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        cells = [PRETTY.get(r["method"], r["method"]),
                 "—" if r["method"] == "dense" else f"{r['target_ratio']:.2f}",
                 f"{r['achieved_linear_ratio']:.4f}",
                 f"{r['linear_params_B']:.3f}"]
        cells += [str(r.get(c, "")) for c in ppl_cols]
        cells += [f"{r['runtime_sec']:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write(f"# LLM SVD compression comparison\n\n{md}\n")
    print("\n" + md)
    print(f"\n[summary] {csv_path}\n[summary] {md_path}")


def main():
    p = argparse.ArgumentParser(description="Run the SVD-LLM / Dobi-SVD / ours comparison")
    p.add_argument("--methods", type=str, default="dense,svdllm,dobisvd,ours")
    p.add_argument("--ratios", type=str, default="0.9,0.8,0.7,0.6")
    p.add_argument("--out_dir", type=str, default=os.path.join(HERE, "results", "comparison"))
    p.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str,
                   default="/home/kangeunjeon/geontack_kairi/data")
    p.add_argument("--calib_dataset", type=str, default="wikitext2")
    p.add_argument("--eval_datasets", type=str, default="wikitext2")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--extra_ours", type=str, default="")
    p.add_argument("--extra_svdllm", type=str, default="")
    p.add_argument("--extra_dobisvd", type=str, default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--keep_going", action="store_true",
                   help="continue the sweep even if a cell fails")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ratios = [float(r) for r in args.ratios.split(",") if r.strip()]
    eval_datasets = [d.strip() for d in args.eval_datasets.split(",") if d.strip()]

    if not args.summary_only:
        cells = []
        if "dense" in methods:
            cells.append(("dense", 1.0))
        for ratio in ratios:
            for m in methods:
                if m != "dense":
                    cells.append((m, ratio))

        for method, ratio in cells:
            out_json, log_path = cell_paths(args.out_dir, method, ratio)
            if os.path.exists(out_json) and not args.overwrite:
                print(f"[skip] {os.path.basename(out_json)} already exists")
                continue
            ok = run_cell(build_cmd(args, method, ratio, out_json), log_path, args.dry_run)
            if not ok and not args.keep_going:
                print("[abort] cell failed; re-run with --keep_going to continue anyway")
                break

    write_outputs(args.out_dir, collect(args.out_dir, eval_datasets), eval_datasets)


if __name__ == "__main__":
    main()
