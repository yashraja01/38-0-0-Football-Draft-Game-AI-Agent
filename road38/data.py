"""
Player data.

For OFFLINE training/eval we generate a realistic pool of "club-seasons", each
a squad of rated players. The live agent doesn't use this at all — it reads the
real squad straight from the website each spin. We only need a representative
*distribution* of draws to learn a good policy, and the policy depends on
relative values, which transfer.

To train on the real distribution instead, drop a Kaggle-style FIFA CSV in and
call `load_fifa_csv(path)`. Expected columns (case-insensitive, flexible):
    short_name / name, club_name / club, player_positions / position,
    overall, and optionally a year/version column.
"""

from __future__ import annotations
import csv
import random
from dataclasses import dataclass, field

from .positions import ALL_POSITIONS


@dataclass(frozen=True)
class Player:
    name: str
    pos: str          # primary position code, e.g. "ST"
    ovr: int          # overall rating 0-99


@dataclass
class ClubSeason:
    label: str                 # e.g. "Manchester City 2018"
    players: list[Player]
    tier: float = 1.0          # draw weight (stronger pools drawn more often)


# Positions a generated squad needs to cover, with how many of each.
_SQUAD_TEMPLATE = {
    "GK": 2, "CB": 4, "RB": 1, "LB": 1, "RWB": 1, "LWB": 1,
    "CDM": 2, "CM": 3, "CAM": 1, "RM": 1, "LM": 1, "RW": 1, "LW": 1,
    "ST": 2, "CF": 1,
}

_CLUBS = [
    ("Man City", 0.95), ("Liverpool", 0.93), ("Arsenal", 0.88), ("Chelsea", 0.86),
    ("Man United", 0.85), ("Tottenham", 0.83), ("Real Madrid", 0.96),
    ("Barcelona", 0.94), ("Bayern", 0.93), ("PSG", 0.90), ("Juventus", 0.88),
    ("Inter", 0.85), ("Milan", 0.84), ("Napoli", 0.84), ("Dortmund", 0.83),
    ("Atletico", 0.85), ("Newcastle", 0.78), ("Aston Villa", 0.76),
    ("Leicester", 0.75), ("Everton", 0.72), ("West Ham", 0.74),
    ("Sevilla", 0.78), ("Roma", 0.80), ("Leipzig", 0.81),
]
_SEASONS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]


def _gen_squad(rng: random.Random, club: str, strength: float, year: int) -> ClubSeason:
    """A squad whose ratings centre on the club's strength for that season."""
    base = 60 + strength * 28          # ~60 (weak) .. ~88 (elite) mean OVR
    players: list[Player] = []
    n = 0
    for pos, count in _SQUAD_TEMPLATE.items():
        for _ in range(count):
            n += 1
            # star players are rarer; right tail via max-of-two draws
            r = max(rng.gauss(base, 6.0), rng.gauss(base, 6.0))
            ovr = int(max(48, min(97, round(r))))
            players.append(Player(f"{club[:3].upper()}-{year%100:02d}-{pos}{n}", pos, ovr))
    # draw weight: stronger pools appear more often (matches the live game)
    tier = 0.4 + strength ** 2
    return ClubSeason(f"{club} {year}", players, tier)


def build_pool(seed: int = 7, n_seasons_per_club: int = 3) -> list[ClubSeason]:
    rng = random.Random(seed)
    pool: list[ClubSeason] = []
    for club, strength in _CLUBS:
        years = rng.sample(_SEASONS, n_seasons_per_club)
        for y in years:
            # season-to-season wobble around the club's baseline
            s = max(0.45, min(0.99, strength + rng.uniform(-0.08, 0.05)))
            pool.append(_gen_squad(rng, club, s, y))
    return pool


# ---------------------------------------------------------------------------
def load_fifa_csv(path: str) -> list[ClubSeason]:
    """Load real player data so the policy is trained on the true distribution."""
    def pick(row, *keys):
        for k in keys:
            for actual in row:
                if actual.lower().replace(" ", "_") == k:
                    return row[actual]
        return None

    buckets: dict[str, list[Player]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = pick(row, "short_name", "name", "long_name")
            club = pick(row, "club_name", "club", "team")
            posraw = pick(row, "player_positions", "position", "positions", "club_position")
            ovr = pick(row, "overall", "ovr", "rating")
            year = pick(row, "fifa_version", "year", "version") or ""
            if not (name and club and posraw and ovr):
                continue
            pos = posraw.split(",")[0].strip().upper()
            if pos not in ALL_POSITIONS:
                continue
            try:
                ovr_i = int(float(ovr))
            except ValueError:
                continue
            buckets.setdefault(f"{club} {year}".strip(), []).append(
                Player(name, pos, ovr_i))

    pool = []
    for label, players in buckets.items():
        if len(players) < 11:
            continue
        mean = sum(p.ovr for p in players) / len(players)
        strength = max(0.45, min(0.99, (mean - 55) / 33))
        pool.append(ClubSeason(label, players, tier=0.4 + strength ** 2))
    return pool


if __name__ == "__main__":
    pool = build_pool()
    pool.sort(key=lambda c: -max(p.ovr for p in c.players))
    print(f"{len(pool)} club-seasons generated. Strongest squads by best player:")
    for c in pool[:6]:
        top = sorted(c.players, key=lambda p: -p.ovr)[:3]
        print(f"  {c.label:22s} tier={c.tier:.2f}  top={[ (p.pos,p.ovr) for p in top ]}")
