"""Round 3b -- a truer dendritic primitive (learned routing + multiplicative
coincidence), with the review's fixes: enforced/attributable LOCALITY and a
non-XOR DISCRIMINATOR task.

Round 3a reclassified the phase-2 "win" as FFN-internal depth, mostly an XOR
artifact. 3b asks: can a faithful dendrite -- branches that LEARN which input
cluster they read and detect its MULTIPLICATIVE coincidence -- beat a plain deep
SwiGLU stack at matched params, on a task that needs composition but ISN'T the
depth-friendly XOR? Four blocks:

  A. 2x2 ATTRIBUTION (B=4, k=d/B=48, mf 0.5) -- one ArborFFN, four corners:
       fg frozen+gate | fp frozen+product | lg learned+gate | lp learned+product
     on XOR (anchor) and the k-ary TABLE (discriminator). vs controls swiglu /
     deep_swiglu / seq_dense (XOR controls exist from 3a; table controls here).
  B. LOCALITY ABLATION (the big risk: softmax locality is init-only). Same B,k,
     task, seeds; on the truer primitive (learned+product):
       born-local-free  = arbor_lp            (peaked  init, softmax)   [in block A]
       born-global-free = arbor_lp_globalfree (uniform init, softmax)
       born-global-conc = arbor_lp_concentr   (uniform init, sparsemax) -> prunes local
     If concentrating beats free-global, locality matters; if not, it's the
     bilinear bottleneck. Routing entropy is logged so a flat-row null is flagged.
  C. CONNECTIVE-TISSUE regime -- the truer primitive as MANY thin coincidence
     detectors (B=16, k=4) + a FAT soma, main_frac swept {0.5,0.75,0.9}. Tests the
     thesis: big neuron + thin learned dendritic wiring vs splitting into depth.
  D. BPE headline -- fit/transfer trade for the primitive.

Serial, one process per run (clean VRAM). Blocks A+B are the CORE (the review's
must-haves); C+D are secondary.

Usage:
    python experiments/run_round3b.py                 # all blocks, seeds 0 1 2
    python experiments/run_round3b.py --only core     # A+B only
    python experiments/run_round3b.py --only attrib|locality|tissue|bpe
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = str(ROOT / "src" / "train.py")

# Block A: 2x2 attribution + table controls (XOR controls exist from 3a).
ATTRIB = [
    "configs/arbor_fg_pcfg.yaml", "configs/arbor_fp_pcfg.yaml",
    "configs/arbor_lg_pcfg.yaml", "configs/arbor_lp_pcfg.yaml",
    "configs/arbor_fg_table.yaml", "configs/arbor_fp_table.yaml",
    "configs/arbor_lg_table.yaml", "configs/arbor_lp_table.yaml",
    "configs/swiglu_table.yaml", "configs/deep_swiglu_table.yaml",
    "configs/seq_dense_table.yaml",
]
# Block B: the two extra locality cells (born-local-free = arbor_lp is in ATTRIB).
LOCALITY = [
    "configs/arbor_lp_globalfree_pcfg.yaml", "configs/arbor_lp_concentr_pcfg.yaml",
    "configs/arbor_lp_globalfree_table.yaml", "configs/arbor_lp_concentr_table.yaml",
]
# Block D: BPE headline.
BPE = ["configs/arbor_lp_bpe.yaml"]

# Block C: connective-tissue regime -- arbor_lp as B=16,k=4, mf swept, via overrides.
TISSUE_BASES = {"pcfg": "configs/arbor_lp_pcfg.yaml",
                "table": "configs/arbor_lp_table.yaml"}
TISSUE_MFS = [0.5, 0.75, 0.9]
TISSUE_B, TISSUE_K = 16, 4


def run(cfg: str, seed: int, extra=None):
    cmd = [sys.executable, TRAIN, "--config", str(ROOT / cfg), "--seed", str(seed)]
    cmd += extra or []
    print(f"\n========== {cfg}  seed={seed}  {' '.join(extra or [])} ==========",
          flush=True)
    subprocess.run(cmd, check=True)


def run_list(tag, cfgs, seeds):
    total = len(cfgs) * len(seeds)
    i = 0
    for cfg in cfgs:
        for seed in seeds:
            i += 1
            print(f"\n###### {tag} [{i}/{total}] ######", flush=True)
            run(cfg, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--only",
                    choices=["core", "attrib", "locality", "tissue", "bpe"],
                    default=None)
    args = ap.parse_args()
    o = args.only
    if o in (None, "core", "attrib"):
        run_list("attrib", ATTRIB, args.seeds)
    if o in (None, "core", "locality"):
        run_list("locality", LOCALITY, args.seeds)
    if o in (None, "tissue"):
        jobs = []
        for sfx, base in TISSUE_BASES.items():
            for mf in TISSUE_MFS:
                name = f"arbor_lp_{sfx}_b{TISSUE_B}k{TISSUE_K}_mf{int(round(mf*100)):03d}"
                jobs.append((base, name, mf))
        total = len(jobs) * len(args.seeds)
        i = 0
        for base, name, mf in jobs:
            for seed in args.seeds:
                i += 1
                print(f"\n###### tissue [{i}/{total}] ######", flush=True)
                run(base, seed, ["--branches", str(TISSUE_B), "--taps", str(TISSUE_K),
                                 "--main-frac", str(mf), "--run-name", name])
    if o in (None, "bpe"):
        run_list("bpe", BPE, args.seeds)

    print("\n===ROUND 3b RUNS DONE===")
    print("Evaluate:")
    print("  # 2x2 + controls on the XOR anchor")
    print("  python src/eval_generalization.py results/analysis/round3b_pcfg.png "
          "swiglu_pcfg deep_swiglu_pcfg seq_dense_pcfg "
          "arbor_fg_pcfg arbor_fp_pcfg arbor_lg_pcfg arbor_lp_pcfg")
    print("  # 2x2 + controls on the TABLE discriminator (the decisive one)")
    print("  python src/eval_generalization.py results/analysis/round3b_table.png "
          "swiglu_table deep_swiglu_table seq_dense_table "
          "arbor_fg_table arbor_fp_table arbor_lg_table arbor_lp_table")
    print("  # locality 3-cell (XOR then table)")
    print("  python src/eval_generalization.py results/analysis/round3b_loc_pcfg.png "
          "arbor_lp_pcfg arbor_lp_globalfree_pcfg arbor_lp_concentr_pcfg")
    print("  python src/eval_generalization.py results/analysis/round3b_loc_table.png "
          "arbor_lp_table arbor_lp_globalfree_table arbor_lp_concentr_table")
    print("  # connective-tissue sweep vs deep_swiglu")
    print("  python src/eval_generalization.py results/analysis/round3b_tissue_table.png "
          "deep_swiglu_table arbor_lp_table_b16k4_mf050 arbor_lp_table_b16k4_mf075 "
          "arbor_lp_table_b16k4_mf090")
    print("  # BPE fit/transfer")
    print("  python src/eval_generalization.py results/analysis/round3b_bpe.png "
          "swiglu_bpe deep_swiglu_bpe seq_dend_bpe arbor_lp_bpe")


if __name__ == "__main__":
    main()
