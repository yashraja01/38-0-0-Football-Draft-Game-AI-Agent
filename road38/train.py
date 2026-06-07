"""Train the agent's value function by Monte-Carlo self-play and save it.

    python -m road38.train --formation 4-3-3 --iters 6 --games 4000
    python -m road38.train --fifa data/players.csv     # train on real data
"""
from __future__ import annotations
import argparse
import os

from .data import build_pool, load_fifa_csv
from .simulator import Game
from .engine import learn_value_function, save_V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formation", default="4-3-3")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--fifa", default=None, help="path to real FIFA CSV (optional)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pool = load_fifa_csv(args.fifa) if args.fifa else build_pool()
    print(f"Pool: {len(pool)} club-seasons | formation {args.formation}")
    game = Game(pool=pool, formation=args.formation, seed=1)

    print("Learning value function (self-play)...")
    V, respin_gain = learn_value_function(game, iterations=args.iters,
                                          games_per_iter=args.games)

    out = args.out or os.path.join(os.path.dirname(__file__), "..", "models",
                                   f"V_{args.formation}.json")
    out = os.path.abspath(out)
    save_V(V, respin_gain, out)
    print(f"\nSaved value function -> {out}")

    # show the learned reservation curve for a few positions
    print("\nLearned reservation value V[pos][k]  (k = rounds remaining):")
    for pos in ("ST", "CB", "GK", "RW", "CM"):
        if pos in V:
            curve = "  ".join(f"{V[pos][k]:5.1f}" for k in range(1, len(V[pos])))
            print(f"  {pos:3s}: {curve}")


if __name__ == "__main__":
    main()
