"""Training loop for one run (Step-1 sanity gate).

Plain-language purpose: train the model, every so often measure validation loss
and print a generated sample so we can literally watch garbled -> coherent text,
and log everything to CSV so we can plot it. Seeded end-to-end for exact
reproduction (PROJECT_BRIEF_PHASE1.md section 9).

Usage:
    python src/train.py --config configs/v0_shakespeare.yaml
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# allow `python src/train.py` to import sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RunConfig, load_config
from data import load_dataset
from ffn import ArborFFN, assert_param_parity, count_params, arbor_report
from model import GPT

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    print("[train] WARNING: CUDA not available, falling back to CPU (slow).")
    return "cpu"


def lr_at(step: int, cfg: RunConfig) -> float:
    """Linear warmup then cosine decay to 10% of peak."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def measure_step_flops(model, dataset, cfg, device, autocast_ctx) -> int:
    """FLOPs of a single forward+backward step (parameter parity is NOT compute
    parity — branched einsums cost differently than a dense matmul, so we report
    learning speed against FLOPs too, not just steps). Measured once via the
    built-in FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode
    x, y = dataset.get_batch("train", cfg.batch_size, cfg.context, device)
    counter = FlopCounterMode(display=False)
    with counter, autocast_ctx:
        _, loss = model(x, y)
        loss.backward()
    flops = counter.get_total_flops()
    model.zero_grad(set_to_none=True)  # discard these grads
    return int(flops)


@torch.no_grad()
def estimate_loss(model, dataset, cfg, device, autocast_ctx):
    """Average loss over a few fixed batches for train and val splits."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = dataset.get_batch(split, cfg.batch_size, cfg.context, device)
            with autocast_ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out["train"], out["val"]


def main():
    # byte-level BPE samples can contain chars the Windows console can't encode;
    # make stdout tolerant so printing a sample never crashes a run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a YAML run config")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the config seed (for multi-seed runs)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override max_steps (e.g. a short Tier-0 smoke run)")
    ap.add_argument("--main-frac", type=float, default=None,
                    help="override main_frac (budget split sweep)")
    ap.add_argument("--branches", type=int, default=None,
                    help="override branches (compartment-count sweep)")
    ap.add_argument("--taps", type=int, default=None,
                    help="override taps (ArborFFN receptive-field size sweep)")
    ap.add_argument("--run-name", type=str, default=None,
                    help="override run_name (so sweep points don't collide)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.main_frac is not None:
        cfg.main_frac = args.main_frac
    if args.branches is not None:
        cfg.branches = args.branches
    if args.taps is not None:
        cfg.taps = args.taps
    if args.run_name is not None:
        cfg.run_name = args.run_name
    set_seed(cfg.seed)

    device = pick_device()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # bf16 autocast on Ada (RTX 4060); no-op context on CPU
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    if use_bf16:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        import contextlib
        autocast_ctx = contextlib.nullcontext()

    dataset = load_dataset(cfg.dataset)
    model = GPT(cfg, dataset.vocab_size).to(device)

    n_backbone = model.num_params()
    n_ffn = sum(count_params(b.ffn) for b in model.blocks)
    print(f"[train] run={cfg.run_name} ffn={cfg.ffn} device={device} "
          f"bf16={use_bf16}")
    print(f"[train] vocab={dataset.vocab_size} params={n_backbone/1e6:.2f}M "
          f"(of which FFN={n_ffn/1e6:.2f}M)")

    # Fairness rule (section 3): assert this FFN is parameter-matched to V0
    # *before* training. The metric checked depends on the variant (active
    # params for the equal variant, allocated otherwise).
    rep = assert_param_parity(cfg, model.blocks[0].ffn)
    print(f"[parity] per-FFN  v0={rep['v0_params']:,}  "
          f"allocated={rep['allocated']:,}  active={rep['active']:,}  "
          f"(matched on {rep['parity_on']}, rel diff {rep['rel_diff']:.3%})")

    # round-3b: report the arbor's actual geometry (per-corner branch width w -- so
    # "learned vs frozen" attribution is read honestly, since learned routing spends
    # params on the route tensor and so must run a narrower w at matched parity).
    addon0 = getattr(model.blocks[0].ffn, "addon", None)
    if isinstance(addon0, ArborFFN):
        main0 = model.blocks[0].ffn.main
        main_h = main0.gate.out_features
        print(f"[arbor]  routing={addon0.routing} nonlin={addon0.branch_nonlin} "
              f"route_norm={addon0.route_norm} route_init={addon0.route_init}  "
              f"B={addon0.B} taps={addon0.k} width={addon0.w}  "
              f"| soma(main) hidden={main_h}  scale_init={model.blocks[0].ffn.scale.item():.2f}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.99),
    )

    # seed-scoped output dir so variants stay grouped: results/<run>/seed<N>/
    run_dir = RESULTS_DIR / cfg.run_name / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "tokens_seen", "wall_clock_s", "flops",
                     "train_loss", "val_loss", "lr"])

    flops_per_step = measure_step_flops(model, dataset, cfg, device, autocast_ctx)
    print(f"[flops] {flops_per_step/1e9:.2f} GFLOP/step (fwd+bwd)")

    # Human-readable log of generated text at each eval, so we can read the
    # garbled -> coherent progression as a single file afterwards.
    samples_path = run_dir / "samples.txt"
    samples_file = open(samples_path, "w", encoding="utf-8")
    samples_file.write(f"# {cfg.run_name}  (ffn={cfg.ffn}, seed={cfg.seed})\n"
                       f"# text samples logged at each eval interval\n")

    # round-3b: log arbor routing entropy + pre-norm activation variance at each
    # eval, so a flat-routing (locality-never-held) run is FLAGGED rather than
    # mis-read as "dendrites don't help", and product blow-up is visible.
    has_arbor = arbor_report(model) is not None
    if has_arbor:
        arbor_file = open(run_dir / "arbor_stats.csv", "w", newline="",
                          encoding="utf-8")
        arbor_writer = csv.writer(arbor_file)
        arbor_writer.writerow(["step", "act_var", "route_entropy", "eff_inputs",
                               "support", "frac_local"])

    tokens_per_step = cfg.batch_size * cfg.context
    start = time.time()
    seed_prompt = torch.zeros((1, 1), dtype=torch.long, device=device)  # newline=0-ish

    for step in range(cfg.max_steps + 1):
        # ---- periodic eval + sample ----
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            tr, va = estimate_loss(model, dataset, cfg, device, autocast_ctx)
            elapsed = time.time() - start
            tokens_seen = step * tokens_per_step
            writer.writerow([step, tokens_seen, f"{elapsed:.1f}",
                             step * flops_per_step,
                             f"{tr:.4f}", f"{va:.4f}", f"{lr_at(step, cfg):.2e}"])
            csv_file.flush()
            print(f"[step {step:>5}] train {tr:.3f} | val {va:.3f} "
                  f"| ppl {math.exp(va):.1f} | {tokens_seen/1e6:.2f}M tok "
                  f"| {elapsed:.0f}s")
            sample = model.generate(seed_prompt, cfg.sample_tokens, top_k=40)[0]
            sample_text = dataset.decode(sample)
            print("  --- sample ---")
            print("  " + sample_text.replace("\n", "\n  "))
            print("  --------------")
            samples_file.write(
                f"\n===== step {step}  (val_loss {va:.3f}, ppl {math.exp(va):.1f}, "
                f"{tokens_seen/1e6:.2f}M tokens) =====\n{sample_text}\n")
            samples_file.flush()

            if has_arbor:
                ar = arbor_report(model)
                arbor_writer.writerow([
                    step, f"{ar['act_var']:.4g}",
                    f"{ar['route_entropy']:.4f}" if ar['learned'] else "",
                    f"{ar['eff_inputs']:.2f}" if ar['learned'] else "",
                    f"{ar['support']:.1f}" if ar['learned'] else "",
                    f"{ar['frac_local']:.3f}" if ar['learned'] else ""])
                arbor_file.flush()
                if ar['learned']:
                    print(f"  [arbor] act_var {ar['act_var']:.3g} | route "
                          f"eff_inputs {ar['eff_inputs']:.1f}/{addon0.d_model} "
                          f"(support {ar['support']:.0f}, frac_local "
                          f"{ar['frac_local']:.2f})")
                else:
                    print(f"  [arbor] act_var {ar['act_var']:.3g} (frozen routing)")

        if step == cfg.max_steps:
            break

        # ---- one optimisation step ----
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, cfg)
        x, y = dataset.get_batch("train", cfg.batch_size, cfg.context, device)
        with autocast_ctx:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    csv_file.close()
    samples_file.close()

    if has_arbor:
        arbor_file.close()
        ar = arbor_report(model)
        if ar['learned']:
            print(f"[arbor] FINAL routing: eff_inputs {ar['eff_inputs']:.1f}/"
                  f"{addon0.d_model}  support {ar['support']:.0f}  "
                  f"frac_local {ar['frac_local']:.2f}  act_var {ar['act_var']:.3g}")
            if ar['frac_local'] < 0.5:
                print("[arbor] WARNING: routing stayed largely GLOBAL "
                      "(frac_local<0.5) -- the locality bias was NOT held, so a null "
                      "here does not test locality (see round-3b notes).")

    # ---- final checkpoint ----
    ckpt = {
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "vocab": {"stoi": dataset.stoi, "itos": dataset.itos},
        "step": cfg.max_steps,
    }
    ckpt_path = run_dir / "ckpt.pt"
    torch.save(ckpt, ckpt_path)
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[train] peak VRAM {peak:.2f} GB")
    print(f"[train] done. metrics -> {csv_path}  checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
