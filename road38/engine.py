"""
The brain. A pure decision layer shared by both the offline simulator and the
live website bot.

Two policies:

  GreedyAgent        — always take the highest in-position value on offer.
                       Simple, but wastes versatile stars in easy slots.

  ReservationAgent   — the good one. Holds a learned value function
                       V[position][k] = the in-position rating a slot of this
                       position is *expected* to end up filled with, when k
                       draft rounds remain. Each spin it drafts the player into
                       the open slot where they most exceed that slot's
                       continuation value V[pos][k-1] (their "surplus"). This is
                       the optimal structure for a sequential stochastic
                       assignment problem: spend a star where it beats the
                       future by the most, not just where it scores highest now.

V is learned by Monte-Carlo self-play to a fixed point (rational-expectations:
the reservation you hold equals what you actually end up getting). No labels, no
gradient descent — approximate dynamic programming, which is the correct tool.

The same `decide()` function drives the live bot: feed it the open slots and the
squad scraped from the page, get back the slot+player to click.
"""

from __future__ import annotations
import json
import random
from collections import defaultdict
from dataclasses import dataclass

from .data import ClubSeason, Player
from .simulator import Game, GameState, slot_value, play_game, team_strength, project_record
from .positions import formation_slots, fit

try:
    from scipy.optimize import linear_sum_assignment
    import numpy as _np
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# --- a draftable move -------------------------------------------------------
@dataclass
class Move:
    kind: str                 # "pick" or "respin"
    player: Player | None = None
    slot_index: int | None = None
    value: float = 0.0
    surplus: float = 0.0


# ---------------------------------------------------------------------------
class GreedyAgent:
    def __init__(self, respin_floor: float = -1.0):
        self.respin_floor = respin_floor   # respin if best value below this

    def act(self, game: Game, state: GameState, squad: ClubSeason):
        moves = game.draftable(squad, state)
        if not moves:
            # nothing eligible in this squad — must respin if allowed, else
            # the round is dead (shouldn't happen with full squads)
            return ("respin",) if state.respins_left > 0 else self._forced(state)
        p, idx, v = max(moves, key=lambda m: m[2])
        if state.respins_left > 0 and v < self.respin_floor:
            return ("respin",)
        return ("pick", p, idx)

    @staticmethod
    def _forced(state):
        idx = state.open_indices[0]
        return ("pick", Player("(empty)", state.slots[idx], 0), idx)


class ReservationAgent:
    """Uses a learned value table V[pos][k]. `use_respins` enables skipping
    below-average draws when respins are available."""

    def __init__(self, V: dict[str, list[float]], respin_gain: dict[int, float] | None = None,
                 use_respins: bool = False, target_ovr: float | None = None):
        self.V = V
        self.respin_gain = respin_gain or {}
        self.use_respins = use_respins
        self.target_ovr = target_ovr   # if set: refuse to fill a slot below this

    def continuation(self, pos: str, k: int) -> float:
        col = self.V.get(pos)
        if not col or k <= 0:
            return 0.0
        return col[min(k, len(col) - 1)]

    def best_move(self, game: Game, state: GameState, squad: ClubSeason) -> Move | None:
        k = len(state.open_indices)        # rounds remaining (incl. this one)
        best: Move | None = None
        for p, idx, v in game.draftable(squad, state):
            cont = self.continuation(state.slots[idx], k - 1)
            s = v - cont
            if best is None or s > best.surplus:
                best = Move("pick", p, idx, v, s)
        return best

    def act(self, game: Game, state: GameState, squad: ClubSeason):
        best = self.best_move(game, state, squad)
        if best is None:
            return ("respin",) if state.respins_left > 0 else GreedyAgent._forced(state)
        if state.respins_left > 0:
            k = len(state.open_indices)
            if self.target_ovr is not None and best.value < self.target_ovr:
                return ("respin",)
            if self.use_respins and best.surplus < self.respin_gain.get(k, 0.0):
                return ("respin",)
        return ("pick", best.player, best.slot_index)


# --- the same decision, exposed as a plain function for the live bot --------
def decide(open_slots: list[tuple[int, str]],
           squad_players: list[Player],
           V: dict[str, list[float]] | None = None,
           target_ovr: float | None = None) -> dict | None:
    """Pure decision used by the live website bot.

    open_slots:    [(slot_index, position_code), ...] still to fill
    squad_players: players available from the current spin
    V:             learned value table (None => greedy)
    target_ovr:    if set, returns {"respin": True} when no pick reaches this
                   in-position value (used for the aggressive "go 38-0-0" mode)

    Returns {"slot_index", "player", "value"} or {"respin": True} or None.
    """
    k = len(open_slots)
    best = None
    for slot_index, pos in open_slots:
        for p in squad_players:
            v = slot_value(p, pos)
            if v <= 0:
                continue
            cont = 0.0
            if V and V.get(pos) and k - 1 > 0:
                col = V[pos]
                cont = col[min(k - 1, len(col) - 1)]
            surplus = v - cont
            if best is None or surplus > best["surplus"]:
                best = {"slot_index": slot_index, "player": p, "value": v, "surplus": surplus}
    if best is None:
        return {"respin": True} if target_ovr else None
    if target_ovr is not None and best["value"] < target_ovr:
        return {"respin": True}
    return best


# ---------------------------------------------------------------------------
def learn_value_function(game: Game, iterations: int = 6, games_per_iter: int = 4000,
                         damping: float = 0.5, seed: int = 0, verbose: bool = True
                         ) -> tuple[dict[str, list[float]], dict[int, float]]:
    """Monte-Carlo self-play fixed point for V[pos][k] and respin thresholds."""
    slots = formation_slots(game.formation)
    positions = sorted(set(slots))
    n = len(slots)
    # V[pos] indexed by k = 0..n ; V[pos][0] = 0 (a slot with 0 rounds left)
    V = {pos: [0.0] * (n + 1) for pos in positions}

    # --- myopic warm start: best eligible value per position in one draw ----
    warm = game.rng
    samp = defaultdict(list)
    for _ in range(3000):
        sq = game.spin()
        for pos in positions:
            best = max((slot_value(p, pos) for p in sq.players), default=0.0)
            samp[pos].append(best)
    for pos in positions:
        m = sum(samp[pos]) / len(samp[pos])
        for k in range(1, n + 1):
            V[pos][k] = m

    rng = random.Random(seed)
    respin_gain: dict[int, float] = {}

    for it in range(iterations):
        agent = ReservationAgent(V)
        acc_val = defaultdict(lambda: defaultdict(list))   # acc_val[pos][k] -> [final values]
        acc_surplus = defaultdict(list)                    # acc_surplus[k] -> [chosen surplus]
        for g in range(games_per_iter):
            game.rng.seed(rng.randrange(1 << 30))
            state = game.reset()
            # record, per round, the slots open and at what k
            open_log = []   # list of (k, [slot_index...])
            while not state.is_done():
                squad = game.spin()
                k = len(state.open_indices)
                open_log.append((k, list(state.open_indices)))
                mv = agent.best_move(game, state, squad)
                acc_surplus[k].append(mv.surplus)
                game.apply(state, mv.player, mv.slot_index)
            final_val = [slot_value(p, state.slots[i]) for i, p in enumerate(state.filled)]
            for k, open_idx in open_log:
                for i in open_idx:
                    acc_val[state.slots[i]][k].append(final_val[i])

        newV = {pos: list(V[pos]) for pos in positions}
        for pos in positions:
            for k in range(1, n + 1):
                vals = acc_val[pos][k]
                if vals:
                    target = sum(vals) / len(vals)
                    newV[pos][k] = (1 - damping) * V[pos][k] + damping * target
        V = newV
        respin_gain = {k: (sum(v) / len(v)) for k, v in acc_surplus.items() if v}

        if verbose:
            # report expected final strength under current policy
            ev = _quick_eval(game, ReservationAgent(V), n_games=2000)
            print(f"  iter {it+1}/{iterations}: mean strength {ev['mean']:.2f}  "
                  f"min {ev['min']:.2f}  38-0-0 rate {ev['perfect']*100:.1f}%")
    return V, respin_gain


def _quick_eval(game: Game, agent, n_games: int = 2000) -> dict:
    strengths = []
    for _ in range(n_games):
        st, rec = play_game(game, agent)
        strengths.append(rec["strength"])
    strengths.sort()
    perfect = sum(1 for s in strengths if project_record(s)["W"] == 38) / len(strengths)
    return {"mean": sum(strengths) / len(strengths), "min": strengths[0],
            "max": strengths[-1], "p10": strengths[len(strengths) // 10],
            "perfect": perfect}


# --- hindsight oracle: best XI obtainable from the exact draws you saw ------
def replay_hindsight(squads: list[ClubSeason], formation: str) -> float:
    """Best strength achievable assigning one player from each of the given
    squads to the 11 slots (one player per spin), via Hungarian assignment."""
    slots = formation_slots(formation)
    n = len(slots)
    assert len(squads) >= n
    # cost matrix: squad i -> slot j, using that squad's best eligible player
    M = [[0.0] * n for _ in range(len(squads))]
    for i, sq in enumerate(squads):
        for j, pos in enumerate(slots):
            M[i][j] = max((slot_value(p, pos) for p in sq.players), default=0.0)
    if _HAVE_SCIPY:
        arr = _np.array(M)
        r, c = linear_sum_assignment(-arr)          # maximise
        total = arr[r, c].sum()
    else:  # greedy fallback
        used_i, used_j, total = set(), set(), 0.0
        cells = sorted(((M[i][j], i, j) for i in range(len(squads)) for j in range(n)),
                       reverse=True)
        for val, i, j in cells:
            if i in used_i or j in used_j:
                continue
            used_i.add(i); used_j.add(j); total += val
            if len(used_j) == n:
                break
    return total / n


def save_V(V, respin_gain, path):
    with open(path, "w") as f:
        json.dump({"V": V, "respin_gain": {str(k): v for k, v in respin_gain.items()}}, f, indent=2)


def load_V(path):
    with open(path) as f:
        d = json.load(f)
    return d["V"], {int(k): v for k, v in d.get("respin_gain", {}).items()}
