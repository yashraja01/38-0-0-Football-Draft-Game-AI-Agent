"""Evaluate the agent: greedy vs learned vs the hindsight-optimal ceiling, on
identical draw sequences, plus a respin sweep showing the road to 38-0-0.

    python -m road38.evaluate
"""
from __future__ import annotations
import argparse
import os
import random
import statistics as st

from .data import build_pool
from .simulator import Game, GameState, project_record, team_strength, slot_value, play_game
from .engine import (GreedyAgent, ReservationAgent, learn_value_function,
                     load_V, replay_hindsight)
from .positions import formation_slots


def play_fixed(game: Game, agent, squads) -> float:
    """Run an agent over a FIXED ordered list of squads (no respins). Returns
    final team strength. Lets us compare policies on identical luck."""
    state = game.reset()
    for sq in squads:
        # disable respins for the fair comparison
        state.respins_left = 0
        mv = agent.act(game, state, sq)
        _, player, idx = mv
        game.apply(state, player, idx)
        if state.is_done():
            break
    return team_strength(state)


def compare(pool, formation="4-3-3", trials=3000, seed=123):
    game = Game(pool=pool, formation=formation, seed=1)
    n = len(formation_slots(formation))

    # train fresh (fast) so eval is self-contained
    V, respin_gain = learn_value_function(game, iterations=5, games_per_iter=2500,
                                          verbose=False)
    greedy = GreedyAgent()
    smart = ReservationAgent(V)

    rng = random.Random(seed)
    res = {"greedy": [], "smart": [], "ceiling": []}
    for _ in range(trials):
        game.rng.seed(rng.randrange(1 << 30))
        squads = [game.spin() for _ in range(n)]
        res["greedy"].append(play_fixed(game, greedy, squads))
        res["smart"].append(play_fixed(game, smart, squads))
        res["ceiling"].append(replay_hindsight(squads, formation))

    def summarize(xs):
        xs = sorted(xs)
        perfect = sum(1 for s in xs if project_record(s)["W"] == 38) / len(xs)
        return {"mean": st.mean(xs), "min": xs[0], "p10": xs[len(xs)//10],
                "max": xs[-1], "perfect%": perfect * 100}

    print(f"\n=== {formation}: {trials} games on identical draw sequences (no respins) ===")
    print(f"{'policy':10s} {'mean':>7s} {'min':>7s} {'p10':>7s} {'max':>7s} {'gap→ceil':>9s}")
    ceil_mean = st.mean(res["ceiling"])
    for name in ("greedy", "smart", "ceiling"):
        s = summarize(res[name])
        gap = ceil_mean - s["mean"]
        print(f"{name:10s} {s['mean']:7.2f} {s['min']:7.2f} {s['p10']:7.2f} "
              f"{s['max']:7.2f} {gap:9.2f}")
    # how much of the greedy->ceiling gap does the smart agent close?
    g = summarize(res["greedy"])["mean"]; sm = summarize(res["smart"])["mean"]
    if ceil_mean - g > 1e-6:
        print(f"\nSmart agent closes {(sm-g)/(ceil_mean-g)*100:.0f}% of the gap "
              f"between greedy and the hindsight-optimal ceiling.")
    return res, V, respin_gain


def respin_sweep(pool, V, respin_gain, formation="4-3-3", trials=1500, seed=7):
    """With respins enabled (watch-ad), how the 38-0-0 rate climbs.
    Aggressive mode refuses any slot below `target` and keeps spinning."""
    target = 95.0
    print(f"\n=== respin sweep ({formation}): aggressive 38-0-0 chase (target OVR {target:.0f}) ===")
    print(f"{'respins':>8s} {'mean':>7s} {'min':>7s} {'38-0-0 rate':>12s}")
    for r in (0, 5, 20, 60, 150):
        game = Game(pool=pool, formation=formation, respins=r, seed=seed)
        agent = ReservationAgent(V, respin_gain, use_respins=(r > 0),
                                 target_ovr=target if r > 0 else None)
        strengths = []
        for _ in range(trials):
            _, rec = play_game(game, agent)
            strengths.append(rec["strength"])
        perfect = sum(1 for s in strengths if project_record(s)["W"] == 38) / len(strengths)
        print(f"{r:8d} {st.mean(strengths):7.2f} {min(strengths):7.2f} {perfect*100:11.1f}%")


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = [x / 2 for x in range(166, 196)]   # 83..97.5
    colors = {"greedy": "#e07a5f", "smart": "#3d9970", "ceiling": "#274690"}
    labels = {"greedy": "Greedy", "smart": "Learned agent", "ceiling": "Hindsight optimum"}
    for name in ("greedy", "smart", "ceiling"):
        ax.hist(res[name], bins=bins, alpha=0.55, label=labels[name], color=colors[name])
    ax.axvline(95.5, ls="--", color="#444", lw=1)
    ax.text(95.6, ax.get_ylim()[1]*0.9, "≈38-0-0 threshold", fontsize=9)
    ax.set_xlabel("Final XI strength (mean in-position rating)")
    ax.set_ylabel("Games")
    ax.set_title("38-0-0 agent: final-XI strength over 3000 identical-luck games (no respins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nSaved distribution plot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formation", default="4-3-3")
    ap.add_argument("--trials", type=int, default=3000)
    args = ap.parse_args()
    pool = build_pool()
    res, V, respin_gain = compare(pool, args.formation, trials=args.trials)
    respin_sweep(pool, V, respin_gain, args.formation)
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "strength_distribution.png"))
    plot(res, out)


if __name__ == "__main__":
    main()
