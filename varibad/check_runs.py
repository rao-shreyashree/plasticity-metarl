"""
to check the 3 things on VariBAD gridworld_rl2 TensorBoard runs

usage:  python check_runs.py logs/logs_GridNavi-v0/rl2__1* logs/logs_GridNavi-v0/rl2__2* ...

needs: pip install tensorboard matplotlib numpy
writes check_runs.png
"""
import sys, os, json, glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIN = 1e6            # frames window for "still rising?" check
RISE_THRESH = 0.3    # return gain (last WIN vs previous WIN) above which we call it still rising
ENT_LOW = 0.3        # dist_entropy below this = near-deterministic policy (max is ln5=1.61)
Z_THRESH = 2.0       # |ep4-ep3| / SE beyond this = real difference, not eval noise
EPS = 4

def load(run):
    ea = EventAccumulator(run, size_guidance={"scalars": 0}); ea.Reload()
    tags = ea.Tags()["scalars"]
    def get(tag):
        if tag not in tags: return None, None
        ev = ea.Scalars(tag); return np.array([e.step for e in ev]), np.array([e.value for e in ev])
    d = {}
    for k in range(1, EPS + 1):
        d[f"m{k}"] = get(f"return_avg_per_frame/episode_{k}")
        d[f"s{k}"] = get(f"return_std_per_frame/episode_{k}")
    d["ent"] = get("policy_losses/dist_entropy")
    d["vl"] = get("policy_losses/value_loss")
    cfg = {}
    p = os.path.join(run, "config.json")
    if os.path.exists(p): cfg = json.load(open(p))
    d["nproc"] = cfg.get("num_processes", 16)
    d["upd_frames"] = None
    return d

def window_mean(x, y, lo, hi):
    m = (x >= lo) & (x < hi)
    return y[m].mean() if m.any() else np.nan

def main(runs):
    fig, ax = plt.subplots(len(runs), 3, figsize=(15, 3.6 * len(runs)), squeeze=False)
    for i, run in enumerate(runs):
        name = os.path.basename(run.rstrip("/"))
        d = load(run)
        if d["m1"][0] is None: print(f"\n{name}: no return_avg_per_frame tags found, skipping"); continue
        fx = d["m1"][0]; fend = fx.max()
        print(f"\n=== {name}  (last eval at {fend/1e6:.2f}M frames, {len(fx)} evals) ===")

        # --- CHECK 1: still rising? ---
        print("1) still rising?  (mean of last 1M frames vs previous 1M)")
        for k in (1, 2, 3, 4):
            x, y = d[f"m{k}"]
            last, prev = window_mean(x, y, fend - WIN, fend + 1), window_mean(x, y, fend - 2 * WIN, fend - WIN)
            sl = np.polyfit(x[x >= fend - WIN] / 1e6, y[x >= fend - WIN], 1)[0] if (x >= fend - WIN).sum() > 2 else np.nan
            verdict = "RISING" if (last - prev) > RISE_THRESH else "flat"
            print(f"   ep{k}: prev={prev:6.2f} last={last:6.2f} delta={last-prev:+.2f} slope/M={sl:+.2f}  -> {verdict}")

        # --- CHECK 2: entropy / value loss ---
        print("2) entropy / value loss (per update; max entropy = ln5 = 1.61)")
        for key, label in (("ent", "entropy"), ("vl", "value_loss")):
            x, y = d[key]
            if x is None: print(f"   {label}: tag missing"); continue
            n = len(y); q = lambda a, b: y[int(n*a):max(int(n*b), int(n*a)+1)].mean()
            print(f"   {label}: start={q(0,.1):.3f} mid={q(.45,.55):.3f} end={q(.9,1):.3f}", end="")
            if key == "ent": print("  -> " + ("COLLAPSED (low entropy)" if q(.9,1) < ENT_LOW else "ok"))
            else: print()

        # --- CHECK 3: ep4 < ep3 noise or real? ---
        x3, m3 = d["m3"]; x4, m4 = d["m4"]; _, s3 = d["s3"]; _, s4 = d["s4"]
        n = min(len(m3), len(m4)); m3, m4, s3, s4, xx = m3[:n], m4[:n], s3[:n], s4[:n], x3[:n]
        diff = m4 - m3; se = np.sqrt(s3**2 + s4**2) / np.sqrt(d["nproc"]); z = diff / np.maximum(se, 1e-8)
        late = xx >= fend - WIN
        print("3) ep4 vs ep3 (eval-noise check)")
        print(f"   all evals : ep4<ep3 in {np.mean(diff<0)*100:.0f}% | mean diff {diff.mean():+.2f}")
        print(f"   last 1M   : ep4<ep3 in {np.mean(diff[late]<0)*100:.0f}% | mean diff {diff[late].mean():+.2f} | "
              f"|z|>{Z_THRESH:.0f} in {np.mean(np.abs(z[late])>Z_THRESH)*100:.0f}% of evals")
        consistent = np.mean(diff[late] < 0) > 0.8 and np.mean(z[late] < -Z_THRESH) > 0.5
        print("   -> " + ("CONSISTENT ep4<ep3 (not noise; GRU may lose context over later episodes)" if consistent
                          else "looks like eval noise (sign flips / small vs SE)"))

        # plots
        for k, c in zip(range(1, 5), ["C0", "C1", "C2", "C3"]):
            x, y = d[f"m{k}"]; ax[i][0].plot(x/1e6, y, c, label=f"ep{k}")
        ax[i][0].set_title(f"{name}: return per episode"); ax[i][0].set_xlabel("M frames"); ax[i][0].legend()
        if d["ent"][0] is not None: ax[i][1].plot(d["ent"][1]); ax[i][1].set_title("dist_entropy (per update)")
        ax[i][2].plot(xx/1e6, diff, "C3"); ax[i][2].fill_between(xx/1e6, -2*se, 2*se, alpha=.2, color="gray")
        ax[i][2].axhline(0, c="k", lw=.5); ax[i][2].set_title("ep4-ep3 (gray = +-2 SE)"); ax[i][2].set_xlabel("M frames")
    plt.tight_layout(); plt.savefig("check_runs.png", dpi=110); print("\nsaved check_runs.png")

if __name__ == "__main__":
    runs = [r for a in sys.argv[1:] for r in glob.glob(a)]
    if not runs: sys.exit("usage: python check_runs.py <run_dir> [<run_dir> ...]")
    main(runs)