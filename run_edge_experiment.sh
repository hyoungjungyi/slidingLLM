#!/bin/bash
# Stage-3 edge-mode comparison.
#   1) build the sliced student once (Stage 0 -> 1 -> 2 -> slice) and cache it
#   2) run both Stage-3 variants from that identical checkpoint, in parallel
# Everything is tee'd to logs/. Safe to detach.
cd /home/kangeunjeon/hyunjung/slidingLLM || exit 1
mkdir -p ckpt logs results
set -o pipefail

BUILD_CKPT=ckpt/r0.6_stage2_sliced.pt
say() { echo; echo "===== $* ====="; date '+%F %T'; echo; }

# Only ever take a GPU that nobody is using. This box is shared, and by the time
# the Stage-3 variants start (hours after the build) the free set will have
# changed, so the choice has to be made at launch time, not up front.
FREE_MIB=1000
pick_gpu() {                      # $1 = space-separated indices to skip
  local skip=" $1 " idx used
  while true; do
    while IFS=, read -r idx used; do
      idx=${idx// /}; used=${used// /}
      case "$skip" in *" $idx "*) continue;; esac
      if [ "$used" -lt "$FREE_MIB" ]; then echo "$idx"; return 0; fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    echo "  [gpu] none free (skip:$1) — waiting 120s" >&2
    sleep 120
  done
}

say "[1/3] BUILD  stage 0 -> 1 -> 2 -> slice   (sw_nsamples=128, ~5.3 h)"
# Never silently reuse: a checkpoint built with --load_ranks looks identical on
# disk but skipped Stage 2, which is the whole thing this comparison rests on.
if [ -f "$BUILD_CKPT" ]; then
  echo "REFUSING to reuse an existing $BUILD_CKPT"
  echo "delete or rename it first if you really want to rebuild"
  exit 1
else
  G=$(pick_gpu "")
  echo "  build -> cuda:$G"
  python -u sliding_llm.py --ratio 0.6 --device cuda:$G --sw_nsamples 128 \
    --skip_stage3 --save_sliced "$BUILD_CKPT" \
    --eval_datasets "" --out_json results/r0.6_stage2_only.json \
    2>&1 | tee logs/1_build.log
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$BUILD_CKPT" ]; then
    echo "BUILD FAILED (exit $rc, checkpoint present: $([ -f "$BUILD_CKPT" ] && echo yes || echo no))"
    echo "aborting — the two variants need the checkpoint"
    exit 1
  fi
fi
ls -la "$BUILD_CKPT"

say "[2/3] STAGE 3 variants, in parallel  (~45 min each)"
GA=$(pick_gpu "")
python -u sliding_llm.py --ratio 0.6 --device cuda:$GA --load_sliced "$BUILD_CKPT" \
  --stage3innerpercent --stage3_edge individual --ft_clip_grad 1.0 \
  --out_json results/r0.6_edge_individual.json \
  > logs/2_edge_individual.log 2>&1 &
PID_A=$!
echo "  A  edge=individual  cuda:$GA  pid $PID_A  -> logs/2_edge_individual.log"

GB=$(pick_gpu "$GA")
python -u sliding_llm.py --ratio 0.6 --device cuda:$GB --load_sliced "$BUILD_CKPT" \
  --stage3innerpercent --ft_clip_grad 1.0 \
  --out_json results/r0.6_edge_skip.json \
  > logs/3_edge_skip.log 2>&1 &
PID_B=$!
echo "  B  edge=skip        cuda:$GB  pid $PID_B  -> logs/3_edge_skip.log"

wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?
echo "  A exit $RC_A | B exit $RC_B"

say "[3/3] RESULT"
python - <<'PY'
import json, os
rows = [("edge=individual (0-5, 28-31 teacher-forced)", "results/r0.6_edge_individual.json"),
        ("edge=skip       (0-5, 28-31 untouched)",      "results/r0.6_edge_skip.json")]
print(f"{'variant':<46} {'wikitext2 ppl':>14} {'ratio':>8} {'avg mse':>10} {'stage3':>9}")
for lab, p in rows:
    if not os.path.exists(p):
        print(f"{lab:<46} {'(missing)':>14}")
        continue
    d = json.load(open(p))
    ppl = d["ppl"].get("wikitext2", float("nan"))
    print(f"{lab:<46} {ppl:>14.4f} {d['linear_param_ratio']:>8.4f} "
          f"{d['avg_window_mse']:>10.5f} {d['timings_sec'].get('stage3_finetune',0)/60:>7.1f}m")
PY
echo
echo "per-task Stage-3 losses:"
for f in logs/2_edge_individual.log logs/3_edge_skip.log; do
  echo "  --- $f ---"
  grep -aoE "(block|window) +[0-9]+ \[[0-9]+:[0-9]+\] \((indi|slid)\)[^|]*\| mse [0-9.e+-]+" "$f" 2>/dev/null | tail -12
done
say "DONE"
