# 38-0-0 drafting agent

An agent that plays **[38-0-0.com](https://38-0-0.com/)**, a football game where you
spin a random Premier League club-season each round, draft one player into an open slot,
repeat for 11 rounds, and get a projected 38-game record driven by the mean in-position ratings
of your XI. The goal is the legendary **38-0-0**: win all 38, no draws, no losses(real invincibles, sorry gunners). 
It's basically a mean 86 squad, very difficult to do, trust me :')

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
highest right now(this is not just a greedy algo with fancy words, i swear!).

So the "model" here is a **value function** `V[position][rounds_remaining]` learned by
**Monte-Carlo self-play to a fixed point** (approximate dynamic programming). No labels, no
gradient descent, just the right and most efficient(imo) tool for the job. A deep RL policy would be slower, less
interpretable, and no better on a problem whose optimal policy is this well-structured. But lowkey I just don't have 
the resources to be training a deep RL kind of model rn so here we are! It does the job well, i guess that's enough 
for a vibe-coded draft game website haha.

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

**Reaching a true 38-0-0** needs re-spins (first attempt 38-0-0 is just unrealistic!)

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

## how to get started:

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

## what is here exactly:

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
`--formation 3-5-2`, etc. just edit it on the play_live.py file in the setup fn!

There are till quite a few bugs/quality of life upgrades I shall be uploading whenever I 
get some spare time from my DSA struggles :')
Feel free to contact me if you have any ideas!

---

## stuff I need to update/fix:

```
->Positions
1) CF allowed in ST
2) LM/RM in LW and RW
3) CAM/CDM in CM
4) LWB/RWB in LB and RB
Fix: simple couple lines of code, will make a huge difference imo!

->Repetition of the same Player

Like eg: if it picks VVD from 2019 liverpool and in the next spins, 2020 liverpool shows up and 
it has a CB position left to fill, it might try to pick VVD again, cause it treats the new VVD as 
a different player altogether.
Fix: not yet decided

->Last player greedy

Sometimes (very rare) it doesn't choose highest rated player at the end cause it doesn't know that 
it's over(?). (Very rare and mind boggling as it knows the amount of remaining players and uses that 
in choice making !)
Fix: Can just hardcode it, lowkey should fix it.

->Screenshotting of every team after it's made
Fix:easy couple lines in code, needed to study last landing page format to detect it easily and take 
a screenshot for the times someone wants to keep this playing in the background for longer sessions.

->recognising last landing page after team is made
So it can go and take the screenshot ofcourse but also:
1 choose "build another" by itself 
2 remove pop up
3 restart team building

===
