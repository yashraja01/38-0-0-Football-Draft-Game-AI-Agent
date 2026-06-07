"""
Position model: which players can fill which formation slots, and how well.

The live game says: "Players can only fill positions they genuinely play —
wing-backs cover full-back, wide midfielders cover the wings ... your projected
record is based on the overall ratings of your XI, *weighted by how well each
player fits the position you put them in.*"

So a player contributes  OVR * fit(player_pos, slot_pos)  to the slot.
We model `fit` from pitch geometry: every position sits at an (x, y) point
(x = defence->attack, y = left->right). Fit decays with distance, the wrong
flank is penalised, and anything below a floor is simply not eligible. This
reproduces the real rules (RWB covers RB, RM covers RW, ST covers CF, etc.)
without hand-listing hundreds of pairs.
"""

from __future__ import annotations
import math
from functools import lru_cache

# (x: defence 0 -> attack 1,  y: left -1 -> right +1)
COORDS: dict[str, tuple[float, float]] = {
    "GK":  (0.00,  0.00),
    "CB":  (0.12,  0.00),
    "RB":  (0.20,  0.85),
    "LB":  (0.20, -0.85),
    "RWB": (0.35,  0.95),
    "LWB": (0.35, -0.95),
    "CDM": (0.35,  0.00),
    "CM":  (0.50,  0.00),
    "CAM": (0.68,  0.00),
    "RM":  (0.55,  0.85),
    "LM":  (0.55, -0.85),
    "RW":  (0.82,  0.85),
    "LW":  (0.82, -0.85),
    "CF":  (0.85,  0.00),
    "ST":  (0.92,  0.00),
}

ALL_POSITIONS = list(COORDS.keys())

_DECAY = 0.44       # distance -> fit slope (neighbours land near ~0.9)
_Y_WEIGHT = 0.5     # wrong-flank penalty is real but not fatal
_FIT_FLOOR = 0.70   # below this a player is *not eligible* for the slot


@lru_cache(maxsize=None)
def fit(player_pos: str, slot_pos: str) -> float:
    """Fit weight in [0, 1].  Exact match = 1.0.  0.0 means ineligible."""
    player_pos = player_pos.upper()
    slot_pos = slot_pos.upper()
    if player_pos == slot_pos:
        return 1.0
    # Goalkeeping is its own world.
    if player_pos == "GK" or slot_pos == "GK":
        return 0.0
    (x1, y1), (x2, y2) = COORDS[player_pos], COORDS[slot_pos]
    dist = math.hypot(x1 - x2, _Y_WEIGHT * (y1 - y2))
    f = 1.0 - _DECAY * dist
    return round(f, 4) if f >= _FIT_FLOOR else 0.0


def eligible(player_pos: str, slot_pos: str) -> bool:
    return fit(player_pos, slot_pos) > 0.0


# ---- Formations: each is an ordered list of 11 slot positions -------------
# Slot codes repeat where a formation needs several of the same role.
FORMATIONS: dict[str, list[str]] = {
    "4-3-3": ["GK", "RB", "CB", "CB", "LB", "CDM", "CM", "CM", "RW", "ST", "LW"],
    "4-4-2": ["GK", "RB", "CB", "CB", "LB", "RM", "CM", "CM", "LM", "ST", "ST"],
    "4-2-4": ["GK", "RB", "CB", "CB", "LB", "CM", "CM", "RW", "ST", "ST", "LW"],
    "3-4-3": ["GK", "CB", "CB", "CB", "RM", "CM", "CM", "LM", "RW", "ST", "LW"],
    "3-5-2": ["GK", "CB", "CB", "CB", "RWB", "LWB", "CM", "CM", "CAM", "ST", "ST"],
    "5-3-2": ["GK", "RWB", "CB", "CB", "CB", "LWB", "CM", "CM", "CAM", "ST", "ST"],
    "5-4-1": ["GK", "RWB", "CB", "CB", "CB", "LWB", "RM", "CM", "CM", "LM", "ST"],
}


def formation_slots(name: str) -> list[str]:
    if name not in FORMATIONS:
        raise KeyError(f"Unknown formation {name!r}. Options: {list(FORMATIONS)}")
    return list(FORMATIONS[name])


if __name__ == "__main__":  # quick sanity print
    for a in ("RB", "RM", "ST", "CM"):
        row = {b: fit(a, b) for b in ALL_POSITIONS if fit(a, b) > 0}
        print(f"{a:4s} fits -> {row}")
