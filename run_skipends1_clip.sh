#!/bin/bash
# Stage 3: leave out only the first and the last block, with per-window gradient
# clipping. Full pipeline, sw_nsamples=128, epochs 5 / 10 / 2.
# Stage 2 checkpoints every epoch, so an OOM costs one epoch, not the whole run.
cd /home/kangeunjeon/hyunjung/slidingLLM || exit 1
mkdir -p ckpt logs results
set -o pipefail

TAG=r0.6_skipends1_clip1
S2CKPT=ckpt/${TAG}_stage2.pt
LOG=logs/${TAG}.log
say() { echo; echo "===== $* ====="; date '+%F %T'; echo; }

# Pick the card with the most headroom, at launch time. The run peaks around
# 6.3 GiB, so 9 GiB free leaves room for a neighbour to grow a little.
NEED_FREE_MIB=9000
pick_gpu() {
  local best="" bestfree=0 idx free
  while true; do
    best=""; bestfree=0
    while IFS=, read -r idx free; do
      idx=${idx// /}; free=${free// /}
      if [ "$free" -gt "$bestfree" ]; then bestfree=$free; best=$idx; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    if [ -n "$best" ] && [ "$bestfree" -ge "$NEED_FREE_MIB" ]; then
      echo "$best"; return 0
    fi
    echo "  [gpu] best is cuda:$best with ${bestfree}MiB free, need ${NEED_FREE_MIB} — waiting 120s" >&2
    sleep 120
  done
}

# A neighbour landing on the same card is not a bug in the run, so retry;
# Stage 2 picks up from its last epoch checkpoint each time.
for attempt in 1 2 3 4 5; do
  G=$(pick_gpu)
  say "ATTEMPT $attempt  on cuda:$G"
  python -u sliding_llm.py \
      --ratio 0.6 --device "cuda:$G" \
      --nsamples 128 --sw_nsamples 128 \
      --stage1_epochs 5 --stage2_epochs 10 --stage3_epochs 2 \
      --stage3_skip_ends 1 --ft_clip_grad 1.0 \
      --stage2_ckpt "$S2CKPT" \
      --out_json results/${TAG}.json \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ]; then say "FINISHED on attempt $attempt"; break; fi
  echo "attempt $attempt failed (exit $rc)"
  if grep -qa "OutOfMemoryError" "$LOG"; then
    echo "  -> GPU contention; Stage 2 resumes from $S2CKPT"
    sleep 60
  else
    echo "  -> not an OOM, giving up"; break
  fi
done

say "RESULT"
python - <<'PY'
import json, os
p = "results/r0.6_skipends1_clip1.json"
if not os.path.exists(p):
    print("  no result JSON — the run did not finish"); raise SystemExit
d = json.load(open(p)); c = d["config"]; t = d["timings_sec"]
print(f"  wikitext2 ppl  : {d['ppl'].get('wikitext2'):.4f}")
print(f"  param ratio    : {d['linear_param_ratio']:.4f}")
print(f"  avg window mse : {d['avg_window_mse']:.6f}")
print(f"  config         : sw_nsamples={c['sw_nsamples']} "
      f"epochs={c['stage1_epochs']}/{c['stage2_epochs']}/{c['stage3_epochs']} "
      f"skip_ends={c['stage3_skip_ends']} clip={c['ft_clip_grad']} rel_mse={c['rel_mse']}")
print("  timings        : " + "  ".join(f"{k.split('_')[0]} {v/60:.0f}m" for k, v in t.items()))
print()
print("  reference (geontack_kairi, sw_nsamples=32, no clipping):")
for name, ppl in [("no stage 3", 10.74), ("stage3inner", 16.79),
                  ("stage3innerpercent", 17.54), ("stage3 full", 743.87)]:
    print(f"     {name:<22} {ppl:>8.2f}")
PY

echo
echo "  worst Stage-3 task losses:"
grep -aoE "window +[0-9]+ \[[0-9]+:[0-9]+\][^|]*\| mse [0-9.e+-]+" "logs/r0.6_skipends1_clip1.log" 2>/dev/null \
  | awk '{print $NF, $0}' | sort -gr | head -6 | cut -d' ' -f2-
say "DONE"
