"""
A self-contained model of the 38-0-0 game loop, used to train and validate the
agent offline.

Each round the game spins a random club-season (stronger pools weighted more
heavily) and shows its squad. You draft ONE eligible player into ONE open slot.
After 11 rounds the XI's strength -> a projected 38-game W/D/L record.

The exact site formula is private, but it is monotonic in "in-position OVR"
(better fitted players => better record). We use a transparent monotonic model
calibrated so an all-elite XI projects to 38-0-0. Because the mapping is
monotonic, the OPTIMAL POLICY IS THE SAME whatever the exact constants are:
maximise total in-position rating. That's what the agent optimises, so it stays
correct even where our record curve only approximates the real one.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field

from .data import ClubSeason, Player, build_pool
from .positions import formation_slots, fit


@dataclass
class GameState:
    formation: str
    slots: list[str]                       # slot position codes (len 11)
    filled: list[Player | None]            # player per slot index, or None
    round: int = 0
    respins_left: int = 0

    @property
    def open_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.filled) if p is None]

    def open_slot_positions(self) -> list[str]:
        return [self.slots[i] for i in self.open_indices]

    def is_done(self) -> bool:
        return all(p is not None for p in self.filled)


def slot_value(player: Player, slot_pos: str) -> float:
    """A player's contribution to a slot: OVR * fit, 0 if ineligible."""
    f = fit(player.pos, slot_pos)
    return player.ovr * f if f > 0 else 0.0


# ---- strength -> projected record -----------------------------------------
# team strength = mean in-position rating across the XI (0..~99).
# Per-game win probability rises steeply once the XI is genuinely elite.
def _per_game_probs(strength: float) -> tuple[float, float, float]:
    # logistic anchors: ~83 title challenger, ~88 champions(~100pts),
    # ~92 invincible-class, ~95 the flawless 38-0-0.
    win = 1.0 / (1.0 + math.exp(-(strength - 85.0) / 2.15))
    loss = 1.0 / (1.0 + math.exp((strength - 70.0) / 3.0))
    win = min(win, 1.0)
    loss = min(loss, max(0.0, 1.0 - win))
    draw = max(0.0, 1.0 - win - loss)
    return win, draw, loss


def project_record(strength: float, games: int = 38) -> dict:
    w, d, l = _per_game_probs(strength)
    W, D, L = round(w * games), round(d * games), round(l * games)
    # reconcile rounding so W + D + L == games (adjust draws first, then losses)
    diff = games - (W + D + L)
    if diff > 0:
        D += diff
    elif diff < 0:
        take = min(D, -diff); D -= take; diff += take
        L = max(0, L + diff)
    pts = W * 3 + D
    if W == games:
        tier = "38-0-0  — THE INVINCIBLE+"
    elif L == 0 and W >= 34:
        tier = "Invincible-class"
    elif pts >= 90:
        tier = "Champions"
    elif pts >= 74:
        tier = "Title challengers"
    elif pts >= 60:
        tier = "Top four"
    elif pts >= 45:
        tier = "Mid-table"
    else:
        tier = "Relegation battle"
    return {"W": W, "D": D, "L": L, "pts": pts, "tier": tier, "strength": round(strength, 2)}


def team_strength(state: GameState) -> float:
    vals = [slot_value(p, state.slots[i]) for i, p in enumerate(state.filled) if p]
    return sum(vals) / len(state.slots) if vals else 0.0


# ---- the environment -------------------------------------------------------
class Game:
    def __init__(self, pool: list[ClubSeason] | None = None,
                 formation: str = "4-3-3", respins: int = 0, seed: int | None = None):
        self.pool = pool if pool is not None else build_pool()
        self.formation = formation
        self.respins = respins
        self.rng = random.Random(seed)
        self._weights = [c.tier for c in self.pool]

    def reset(self) -> GameState:
        slots = formation_slots(self.formation)
        return GameState(self.formation, slots, [None] * len(slots),
                         round=0, respins_left=self.respins)

    def spin(self) -> ClubSeason:
        """Draw a club-season, weighted toward stronger pools (as the site does)."""
        return self.rng.choices(self.pool, weights=self._weights, k=1)[0]

    def draftable(self, squad: ClubSeason, state: GameState) -> list[tuple[Player, int, float]]:
        """All legal (player, slot_index, value) moves for the current open slots."""
        moves = []
        for idx in state.open_indices:
            spos = state.slots[idx]
            for p in squad.players:
                v = slot_value(p, spos)
                if v > 0:
                    moves.append((p, idx, v))
        return moves

    def apply(self, state: GameState, player: Player, slot_index: int) -> None:
        assert state.filled[slot_index] is None, "slot already filled"
        state.filled[slot_index] = player
        state.round += 1


def play_game(game: Game, agent, seed: int | None = None) -> tuple[GameState, dict]:
    """Run one full game with an agent. Agent must implement .act(...)."""
    if seed is not None:
        game.rng.seed(seed)
    state = game.reset()
    while not state.is_done():
        squad = game.spin()
        while True:
            move = agent.act(game, state, squad)   # ("pick", player, idx) | ("respin",)
            if move[0] == "respin" and state.respins_left > 0:
                state.respins_left -= 1
                squad = game.spin()
                continue
            _, player, idx = move
            game.apply(state, player, idx)
            break
    return state, project_record(team_strength(state))


if __name__ == "__main__":
    for s in (60, 75, 83, 88, 92, 96):
        print(f"strength {s:4.0f} -> {project_record(s)}")
