# 38-0-0 drafting agent

An agent that plays **[38-0-0.com](https://38-0-0.com/)** — the football game where you
spin a random Premier League club-season each round, draft one player into an open slot,
repeat for 11 rounds, and get a projected 38-game record driven by the in-position ratings
of your XI. The goal is the legendary **38-0-0**: win all 38, no draws, no losses.

This repo contains the **brain** (a trained decision engine), an **offline simulator** to
train and validate it, and a **browser bot** that drives the real website with that brain.

---

## What kind of problem is this, really?

It looks like a game for a neural net, but it isn't. Each round you see a *random* squad and
must commit one player to one slot, never knowing what later rounds will offer. That's a
textbook **sequential stochastic assignment problem**. The optimal structure is known: hold a
*reservation value* for each open slot — the rating you expect to fill it with later — and
assign each arriving player to the slot where they most **exceed** that expectation (their
*surplus*). Spend a star where it beats the future by the most, not merely where it scores
highest right now.

So the "model" here is a **value function** `V[position][rounds_remaining]` learned by
**Monte-Carlo self-play to a fixed point** (approximate dynamic programming). No labels, no
gradient descent — just the right tool for the job. A deep RL policy would be slower, less
interpretable, and no better on a problem whose optimal policy is this well-structured.

The learned reservations come out reading like real football sense:

```
V[pos][k]  (k = draft rounds remaining)        k=1                         k=11
  ST :  89.7  90.7  91.5  92.0 ... 93.8     (scarce-ish, climbs as you can afford to wait)
  CB :  91.1  91.6  92.0  92.3 ... 93.3     (plentiful — high floor even late)
  RW :  86.9  88.2  89.0  89.5 ... 91.5     (wingers are rare → lowest reservation)
```

---

## Results (offline, 3000 games on identical draw sequences)

| policy | mean strength | 10th-pct (floor) | gap to ceiling |
|---|---|---|---|
| Greedy (take highest value now) | 92.74 | 91.38 | 1.08 |
| **Learned agent** | **92.98** | **91.70** | **0.83** |
| Hindsight optimum (perfect foresight) | 93.81 | — | 0.00 |

The learned agent beats greedy on both the mean **and the floor** (more consistent good
results) and closes ~23% of the greedy→optimal gap. See `strength_distribution.png`.

**Reaching a true 38-0-0** needs re-spins (the game gates them behind watch-an-ad). With the
aggressive "chase" policy:

| re-spins allowed | 38-0-0 rate |
|---|---|
| 0 | ~9% |
| 5 | ~35% |
| 20 | ~92% |
| 60 | ~99.8% |
| 150 | 100% |

This matches how the site frames it: "achievable but punishing — you'll need elite players in
every position and very little luck to spare."

---

## Quick start

```bash
pip install -r requirements.txt

python -m road38.train          # learn the value function (saved to models/)
python -m road38.evaluate       # policy comparison + ceiling + respin sweep + plot
python -m road38.demo --seed 42 # watch one game drafted pick-by-pick
python -m road38.demo --chase --respins 60   # watch it chase a 38-0-0
```

### Play the live website

```bash
playwright install chromium
python -m road38.play_live --snapshot    # 1) spin once, dump page.html + inventory.json
python -m road38.play_live --dry-run     # 2) scrape + decide, no clicking
python -m road38.play_live               # 3) play for real
python -m road38.play_live --chase --max-respins 80   # chase a perfect season
```

The bot does **not** rely on the site's CSS class names (those are hidden behind the JS app and
change over time — guessing them is what makes these bots silently fail). Instead it finds
controls by their visible text ("Classic", "SPIN", "Re-spin", "4-3-3", …) and finds player cards
by **content**: a small injected script tags every on-screen element that holds a football
position *and* a player name, then the bot clicks the one `decide()` chooses. If detection ever
comes up empty, `--snapshot` writes the live page and an element inventory to `./live_snapshot/`
so the exact controls can be pinned down.

---

## Project layout

```
road38/
  positions.py   pitch geometry -> who can play where, and the fit weight
  data.py        synthetic club-season pool (+ real FIFA-CSV loader)
  simulator.py   the game loop: weighted draws, scoring, strength->record
  engine.py      the brain: greedy + learned reservation agent, self-play learner,
                 hindsight oracle, and decide() (the shared decision function)
  train.py       learn + save the value function
  evaluate.py    compare policies vs the ceiling; respin sweep; plot
  demo.py        play one game out loud, pick by pick
  play_live.py   Playwright bot that drives 38-0-0.com using decide()
models/          saved value functions, e.g. V_4-3-3.json
```

All seven site formations (4-3-3, 4-4-2, 4-2-4, 3-4-3, 3-5-2, 5-3-2, 5-4-1) are supported:
`--formation 3-5-2`, etc.

---

## Honest limitations

- **The exact site scoring formula is private.** I use a transparent, monotonic
  strength→record curve calibrated so an all-elite XI tips into 38-0-0. Because the mapping is
  monotonic, **the optimal policy is identical regardless of the exact constants** — maximise
  total in-position rating — so the agent stays correct even where the record curve only
  approximates the real one.
- **Training uses a synthetic player pool.** The policy depends on *relative* values, which
  transfer, but to train on the true distribution drop a Kaggle FIFA CSV in and run
  `python -m road38.train --fifa players.csv`. The live bot doesn't use this pool at all — it
  reads the real squad from the page each spin.
- **The learned agent's edge over greedy grows with draw heterogeneity and tighter re-spin
  budgets.** On a very strong, uniform pool, greedy is already close to optimal.
- **The live bot is the only part that touches the website, and it can't be verified without a
  network connection to the live page.** It avoids brittle CSS selectors by finding controls via
  their visible text and player cards via their content (position + name), which is robust to the
  site's markup changing. If the site significantly restructures its draft UI and detection comes
  up empty, run `--snapshot` and the saved `page.html` / `inventory.json` make it a quick fix.
- **Re-spins are ad-gated.** A bot can't reliably watch a video ad, so on the live site
  `--chase` only re-spins where no ad wall blocks it; a true live 38-0-0 usually needs a human
  to sit through the ads. The strategy is identical either way.
```
