"""eval_checkpoints.py

evaluation of saved VariBAD gridworld_rl2 checkpoints on ALL 21 goals (no 16-random-task noise)
uses repo'sn evaluate() with 1 process and a patched GridNavi.reset_task that pins the goal.
writes csv - eval_checkpoints.csv

additional: --stoch 20   
(for stochastic-policy repeats per goal; default is deterministic = same as the logged eval)

use from repo root:
  python eval_checkpoints.py --repo C:\Users\raosh\Documents\PESU\Capstone\varibad  "logs_gridworld/logs_GridNavi-v0/rl2_*"

"""

import argparse, glob, os, re, sys, json, csv, types
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help="run dirs (globs ok), each with config.json and models/")
ap.add_argument("--repo", default=".", help="path to the varibad repo root")
ap.add_argument("--stoch", type=int, default=0, help="N stochastic repeats per goal (0 = skip)")
cli = ap.parse_args()
sys.path.insert(0, os.path.abspath(cli.repo))

import torch
import environments                                   # registers GridNavi-v0
from environments.navigation.gridworld import GridNavi
from utils.evaluation import evaluate, device as DEV

# ---- pin the goal: GridNavi only supports random goals unless we patch reset_task ----
FIXED = {"goal": None}
_orig_reset_task = GridNavi.reset_task
def _pinned(self, task=None):
    return _orig_reset_task(self, task=FIXED["goal"])
GridNavi.reset_task = _pinned
GOALS = [tuple(int(v) for v in g) for g in GridNavi().possible_goals]          # the 21 goals
REF = {"explore->exploit (hand-coded)": [4.00, 10.76, 11.07, 11.07], "oracle": [11.07] * 4}

def tload(p):
    try: return torch.load(p, map_location=DEV, weights_only=False)
    except TypeError: return torch.load(p, map_location=DEV)       # old torch

class StochPolicy:                                    # force sampling instead of mode
    def __init__(self, p): self.p = p
    def act(self, *a, **k): k["deterministic"] = False; return self.p.act(*a, **k)

def run_goal(ns, policy, encoder, goal, seed_idx):
    FIXED["goal"] = np.array(goal)
    with torch.no_grad():
        r = evaluate(ns, policy, ret_rms=None, iter_idx=seed_idx, tasks=None, encoder=encoder)
    return r.cpu().numpy()[0]                         # [episodes]

def grid(vals, ep):                                   # 5x5 text map of per-goal return for episode ep
    out = []
    for y in range(4, -1, -1):
        out.append(" ".join(f"{vals[(x, y)][ep]:5.1f}" if (x, y) in vals else "  .  " for x in range(5)))
    return "\n".join(out)

rows = []
runs = [r for a in cli.runs for r in sorted(glob.glob(a))]
for run in runs:
    name = os.path.basename(run.rstrip("/\\"))
    ns = types.SimpleNamespace(**json.load(open(os.path.join(run, "config.json"))))
    ns.num_processes = 1                               # DummyVecEnv, in-process (needed for the patch)
    mdir = os.path.join(run, "models")
    ckpts = sorted(glob.glob(os.path.join(mdir, "policy*.pt")),
                   key=lambda p: int(re.findall(r"policy(\d*)\.pt", p)[0] or 10**9))
    if not ckpts: print(f"{name}: no models/policy*.pt"); continue
    print(f"\n=== {name}: {len(ckpts)} checkpoint(s) ===")
    for pp in ckpts:
        lab = re.findall(r"policy(\d*)\.pt", pp)[0]
        ep = os.path.join(mdir, f"encoder{lab}.pt")
        policy, encoder = tload(pp), tload(ep)
        det = {g: run_goal(ns, policy, encoder, g, 0) for g in GOALS}
        D = np.array([det[g] for g in GOALS])          # [21, episodes]
        mean = D.mean(0)
        # noise in ONE logged eval point (16 random goals drawn with replacement)
        rng = np.random.default_rng(0)
        boot = np.array([D[rng.integers(0, len(GOALS), 16)].mean(0) for _ in range(2000)])
        tag = f"iter {lab}" if lab else "latest (policy.pt)"
        print(f"-- {tag}  [deterministic, exact over 21 goals]")
        print("   mean return per episode : " + "  ".join(f"ep{k+1} {m:6.2f}" for k, m in enumerate(mean)))
        print("   SD of a logged 16-task eval : " + "  ".join(f"ep{k+1} {s:5.2f}" for k, s in enumerate(boot.std(0))))
        print("   goals with ep4 return < 5 : " + (", ".join(str(g) for g in GOALS if det[g][-1] < 5) or "none"))
        print("   per-goal ep4 return (x right, y up):\n" + "\n".join("      " + l for l in grid(det, -1).split("\n")))
        for g in GOALS: rows.append([name, lab or "latest", "det", g[0], g[1], *det[g].tolist()])
        if cli.stoch > 0:
            S = []
            for g in GOALS:
                S.append(np.mean([run_goal(ns, StochPolicy(policy), encoder, g, 1 + i) for i in range(cli.stoch)], 0))
            S = np.array(S)
            print(f"-- {tag}  [stochastic, {cli.stoch} repeats/goal]")
            print("   mean return per episode : " + "  ".join(f"ep{k+1} {m:6.2f}" for k, m in enumerate(S.mean(0))))
            for g, s in zip(GOALS, S): rows.append([name, lab or "latest", "stoch", g[0], g[1], *s.tolist()])

print("\nreference: " + " | ".join(f"{k}: {v}" for k, v in REF.items()))
with open("eval_checkpoints.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run", "ckpt", "mode", "goal_x", "goal_y", "ep1", "ep2", "ep3", "ep4"]); w.writerows(rows)
print("saved eval_checkpoints.csv")
