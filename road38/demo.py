"""Play one full game out loud, so you can watch the agent reason pick-by-pick.

    python -m road38.demo
    python -m road38.demo --chase --respins 60     # chase a 38-0-0
"""
from __future__ import annotations
import argparse
import os

from .data import build_pool
from .simulator import Game, team_strength, project_record, slot_value
from .engine import ReservationAgent, GreedyAgent, load_V, learn_value_function, save_V
from .positions import formation_slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formation", default="4-3-3")
    ap.add_argument("--chase", action="store_true")
    ap.add_argument("--respins", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    pool = build_pool()
    model = os.path.join(os.path.dirname(__file__), "..", "models",
                         f"V_{args.formation}.json")
    if os.path.exists(model):
        V, respin_gain = load_V(model)
    else:
        g0 = Game(pool=pool, formation=args.formation, seed=1)
        V, respin_gain = learn_value_function(g0, iterations=5, games_per_iter=2500,
                                              verbose=False)
        save_V(V, respin_gain, model)

    game = Game(pool=pool, formation=args.formation,
                respins=args.respins, seed=args.seed)
    agent = ReservationAgent(V, respin_gain, use_respins=args.respins > 0,
                             target_ovr=95.0 if args.chase else None)
    slots = formation_slots(args.formation)
    state = game.reset()

    print(f"Formation {args.formation}: slots = {slots}")
    print(f"Respins available: {args.respins}\n")
    rnd = 0
    while not state.is_done():
        squad = game.spin()
        rnd += 1
        while True:
            mv = agent.act(game, state, squad)
            if mv[0] == "respin" and state.respins_left > 0:
                state.respins_left -= 1
                print(f"  round {rnd}: re-spin (drew {squad.label}, nothing good enough)")
                squad = game.spin()
                continue
            _, player, idx = mv
            v = slot_value(player, slots[idx])
            print(f"R{rnd:2d}  drew {squad.label:18s} -> draft {player.name:14s}"
                  f" ({player.pos:3s} {player.ovr}) into slot {idx:2d} {slots[idx]:3s}"
                  f"   in-position value {v:4.1f}")
            game.apply(state, player, idx)
            break

    print("\nFinal XI:")
    for i, p in enumerate(state.filled):
        print(f"  {slots[i]:3s}: {p.name:14s} {p.pos:3s} {p.ovr}"
              f"  (fit-weighted {slot_value(p, slots[i]):.1f})")
    rec = project_record(team_strength(state))
    print(f"\nTeam strength {rec['strength']}  ->  {rec['W']}-{rec['D']}-{rec['L']}"
          f"  ·  {rec['pts']} pts  ·  {rec['tier']}")


if __name__ == "__main__":
    main()
