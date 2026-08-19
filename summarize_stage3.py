#!/usr/bin/env python
"""Collect the Stage-3 experiment arms into one CSV.

Every number is read back out of the result JSON the run wrote, so this stays
honest if an arm is re-run: nothing here is transcribed by hand. Per-pass means
only exist for sequential arms and are scraped from the log, which is the only
place stage3_sliding prints them.
"""
import csv, json, os, re, sys

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results", "s3_summary.csv")

# label | family | result json | log (for per-pass means)
ARMS = [
    ("no stage 3",              "plain SVD student", "r0.6_stage2_baseline.json", None),
    ("rel_lr 3e-5 (reference)", "plain SVD student", "r0.6_rellr.json",           None),
    ("rel_lr 1e-4",             "plain SVD student", "s3_rellr1e-4.json",         "s3_rellr1e-4.log"),
    ("rel_lr 3e-4",             "plain SVD student", "s3_rellr3e-4.json",         "s3_rellr3e-4.log"),
    ("joint full-rank",         "U/V x parameterisation", "r0.6_rellr.json",      None),
    ("sequential full-rank [A]", "U/V x parameterisation", "s3_uvA_seq_full.json",   "s3_uvA_seq_full.log"),
    ("sequential LoRA r=8 [B]",  "U/V x parameterisation", "s3_uvB_seq_lora8.json",  "s3_uvB_seq_lora8.log"),
    ("joint LoRA r=8 [C]",       "U/V x parameterisation", "s3_uvC_joint_lora8.json", "s3_uvC_joint_lora8.log"),
    ("stage3 rel_lr 3e-5",      "massive split", "r0.6_masssplit_rellr.json", "r0.6_masssplit_rellr.log"),
    ("stage3 const lr 1e-4",    "massive split", "s3_masssplit_const.json",   "s3_masssplit_const.log"),
]

PASS_RE = re.compile(r"pass (\d+) mean window mse ([0-9.]+)")


def pass_means(log):
    """{pass index: mean window mse} — empty for joint arms, which print none."""
    p = os.path.join(BASE, "logs", log or "")
    if not log or not os.path.exists(p):
        return {}
    with open(p, "rb") as fh:
        text = fh.read().decode("utf-8", "replace").replace("\r", "\n")
    return {int(i): float(v) for i, v in PASS_RE.findall(text)}


rows = []
for label, family, jname, log in ARMS:
    path = os.path.join(BASE, "results", jname)
    if not os.path.exists(path):
        print(f"[warn] missing {jname} — skipping {label}", file=sys.stderr)
        continue
    d = json.load(open(path))
    c = d["config"]
    pm = pass_means(log)
    stage3 = not c.get("skip_stage3", False)
    rows.append({
        "family": family,
        "arm": label,
        "wikitext2_ppl": round(d["ppl"]["wikitext2"], 4),
        "avg_window_mse": round(d["avg_window_mse"], 6),
        "linear_param_ratio": round(d["linear_param_ratio"], 5),
        "uv_mode": c.get("stage3_uv_mode", "joint") if stage3 else "n/a",
        "lora_r": c.get("stage3_lora_r", 0) if stage3 else "n/a",
        "lr_mode": c["ft_lr_mode"] if stage3 else "n/a",
        "lr": (c["ft_rel_lr"] if c["ft_lr_mode"] == "relative" else c["ft_lr"]) if stage3 else "n/a",
        "stage3_epochs": c["stage3_epochs"] if stage3 else 0,
        "nsamples": c["nsamples"],
        "seqlen": c["seqlen"],
        "pass1_mse": pm.get(1, ""),
        "pass2_mse": pm.get(2, ""),
        "result_json": f"results/{jname}",
        "log": f"logs/{log}" if log else "",
    })

# ppl relative to the un-fine-tuned student of the same family
base = {r["family"]: r["wikitext2_ppl"] for r in rows if r["arm"] == "no stage 3"}
base["U/V x parameterisation"] = base.get("plain SVD student")
base["massive split"] = None          # its no-stage-3 point was never measured
for r in rows:
    b = base.get(r["family"])
    r["delta_ppl_vs_no_stage3"] = round(r["wikitext2_ppl"] - b, 4) if b else ""

# The reading columns only. Everything else the runs recorded — uv_mode, lr,
# per-pass means, source paths — stays in the result JSONs under results/.
cols = ["family", "arm", "wikitext2_ppl", "avg_window_mse", "linear_param_ratio"]
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"wrote {OUT}  ({len(rows)} arms)")
