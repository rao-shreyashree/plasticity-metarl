"""
gridworld_baselines.py

Reference returns for the VariBAD GridNavi (5x5, 15 steps/ep, 4 eps/task) used by gridworld_rl2.
Dynamics copied from environments/navigation/gridworld.py (reward +1 on goal cell, -0.1 elsewhere, episode = 15 steps, state resets to (0,0) each episode, goal fixed across the 4 episodes).

usage:  python gridworld_baselines.py (no torch/gym needed)
"""
import itertools, random
import numpy as np

N, T, EPS = 5, 15, 4
GOALS = [g for g in itertools.product(range(N), repeat=2) if g not in [(0,0),(0,1),(1,1),(1,0)]]
MOVES = {0:(0,0), 1:(0,1), 2:(1,0), 3:(0,-1), 4:(-1,0)}  # noop, up, right, down, left

def step(pos, a):
    dx, dy = MOVES[a]
    return (min(max(pos[0]+dx,0),N-1), min(max(pos[1]+dy,0),N-1))

def run(policy_factory, goal):
    """returns per-episode returns [EPS] for one task. policy_factory() -> policy object (fresh per task)."""
    pol = policy_factory()
    rets = []
    for ep in range(EPS):
        pos = (0,0); R = 0.0; pol.start_episode(ep)
        for t in range(T):
            a = pol.act(pos, t)
            pos = step(pos, a)
            r = 1.0 if pos == goal else -0.1
            pol.observe(pos, r)
            R += r
        rets.append(R)
    return rets

# policies
class Random: # uniform over 5 actions, no memory
    def start_episode(self, ep): pass
    def act(self, pos, t): return random.randrange(5)
    def observe(self, pos, r): pass

class SnakeSweep: # NON-adaptive: same fixed open-loop sweep every episode, ignores reward
    PATH = []
    def __init__(self):
        if not SnakeSweep.PATH:
            cells = []
            for y in range(N):
                xs = range(N) if y % 2 == 0 else range(N-1,-1,-1)
                cells += [(x,y) for x in xs]
            acts, cur = [], (0,0)
            for c in cells[1:]:
                while cur != c:
                    for a in (1,2,3,4):
                        if step(cur,a) == c: acts.append(a); cur = c; break
            SnakeSweep.PATH = acts
    def start_episode(self, ep): pass
    def act(self, pos, t): return SnakeSweep.PATH[t] if t < len(SnakeSweep.PATH) else 0
    def observe(self, pos, r): pass

class Oracle: # knows goal: shortest path then stay (upper bound)
    def __init__(self, goal): self.g = goal
    def start_episode(self, ep): pass
    def act(self, pos, t):
        if pos[0] < self.g[0]: return 2
        if pos[0] > self.g[0]: return 4
        if pos[1] < self.g[1]: return 1
        if pos[1] > self.g[1]: return 3
        return 0
    def observe(self, pos, r): pass

class ExploreExploit: # hand-coded ADAPTIVE: visit nearest unvisited cell until reward seen, then go to goal + stay
    def __init__(self): self.visited = {(0,0)}; self.goal = None
    def start_episode(self, ep): pass
    def act(self, pos, t):
        if self.goal is not None: return Oracle(self.goal).act(pos, t)
        cand = [c for c in GOALS if c not in self.visited]
        tgt = min(cand, key=lambda c: abs(c[0]-pos[0])+abs(c[1]-pos[1]))
        return Oracle(tgt).act(pos, t)
    def observe(self, pos, r):
        self.visited.add(pos)
        if r > 0: self.goal = pos

def avg(factory, mc=1):
    out = np.zeros(EPS)
    for g in GOALS:
        out += np.mean([run((lambda: factory(g)), g) for _ in range(mc)], axis=0)
    return out / len(GOALS)

if __name__ == "__main__":
    random.seed(0)
    res = {
        "random (uniform)":              avg(lambda g: Random(), mc=2000),
        "non-adaptive snake sweep":      avg(lambda g: SnakeSweep()),
        "hand-coded explore->exploit":   avg(lambda g: ExploreExploit()),
        "oracle (knows goal)":           avg(lambda g: Oracle(g)),
    }
    print(f"{'policy':32s}" + "".join(f"ep{k+1:>6d}" for k in range(EPS)))
    for k, v in res.items():
        print(f"{k:32s}" + "".join(f"{x:8.2f}" for x in v))