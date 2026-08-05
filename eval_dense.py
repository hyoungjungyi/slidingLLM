"""Uncompressed reference row for the comparison table."""
import argparse

import torch

import data_utils
import rotation
from llm_utils import (Timer, evaluate_ppl, load_llm, param_report,
                       save_result, seed_everything)


def main():
    p = argparse.ArgumentParser(description="Dense (uncompressed) perplexity reference")
    p.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, default=data_utils.DEFAULT_DATA_ROOT)
    p.add_argument("--calib_dataset", type=str, default="wikitext2")  # unused, kept for CLI parity
    p.add_argument("--eval_datasets", type=str, default="wikitext2")
    p.add_argument("--nsamples", type=int, default=128)              # unused, CLI parity
    p.add_argument("--eval_batch_size", type=int, default=1)
    # Rotation is exactly function-preserving, so `--rotate` here is the check
    # that it is: dense ppl must be unchanged up to bf16 rounding.
    p.add_argument("--rotate", type=str, default="none", choices=["none", "hadamard", "random"])
    p.add_argument("--rotate_seed", type=int, default=0)
    p.add_argument("--out_json", type=str, default="")
    args = p.parse_args()

    seed_everything(args.seed)
    model, tokenizer = load_llm(args.model_id, args.dtype, args.seqlen)
    rotation.apply_rotation(model, args.rotate, args.rotate_seed, torch.device(args.device))
    report = param_report(model)

    ppl = {}
    with Timer() as t:
        for ds in args.eval_datasets.split(","):
            ds = ds.strip()
            if not ds:
                continue
            test_ids = data_utils.get_test_input_ids(tokenizer, args.seqlen, ds, args.data_root)
            ppl[ds] = evaluate_ppl(model, test_ids, torch.device(args.device),
                                   args.eval_batch_size, desc=f"ppl:{ds}")
            print(f"[eval] {ds} perplexity: {ppl[ds]:.4f}")

    payload = {
        "method": "dense",
        "model_id": args.model_id,
        "target_ratio": 1.0,
        "ppl": ppl,
        "timings_sec": {"eval": t.elapsed},
        "config": vars(args),
        **report,
    }
    if args.out_json:
        save_result(args.out_json, payload)


if __name__ == "__main__":
    main()
