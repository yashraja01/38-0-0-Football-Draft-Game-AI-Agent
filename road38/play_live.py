"""
Live bot: drives https://38-0-0.com/ in a real browser and drafts each pick with
the trained decision engine (road38/engine.py).

    pip install playwright
    playwright install chromium

    python -m road38.play_live --snapshot         # 1) dump the page so selectors can be verified
    python -m road38.play_live --dry-run          # 2) scrape + decide, NO clicking
    python -m road38.play_live                     # 3) actually play (4-3-3, classic)
    python -m road38.play_live --chase --max-respins 80   # chase a perfect 38-0-0

────────────────────────────────────────────────────────────────────────────
WHY THIS REWRITE FIXES "the agent can't choose the players"
The site is a JavaScript app whose CSS class names are not visible from outside
and change over time, so guessing classes (the old approach) made scraping
return nothing. This version does NOT depend on class names:

  • Controls (mode / formation / SPIN / Re-spin / Start) are found by their
    VISIBLE TEXT, which is stable: "Classic", "SPIN", "Re-spin", "4-3-3", ...
  • Player cards are found by CONTENT: an element is a player card if its text
    contains a football position (GK/CB/.../ST) and, in Classic mode, an OVR
    rating. We inject a small JS scanner that finds the *minimal* such elements
    (a card, not the container holding all cards), tags each with
    data-bot-card="i", and returns its parsed name/pos/ovr. The bot then clicks
    by that tag — no fragile selectors.

If the auto-detector ever comes up empty, run `--snapshot`: it writes the live
page's HTML and a JSON inventory of every clickable element + card candidate to
./live_snapshot/. Send me those two files and I'll wire in exact selectors.

Re-spins are gated behind watch-an-ad. A bot can't watch a video ad, so --chase
only re-spins where the page allows it without an ad wall; otherwise it takes
the best available pick. A true live 38-0-0 usually needs a human for the ads.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time

from .data import Player
from .positions import ALL_POSITIONS, formation_slots
from .engine import decide, load_V

# ───────────────────────── known, stable VISIBLE TEXT ──────────────────────
# These are read off the live page and are far more stable than CSS classes.
TEXT = {
    "cookie_accept":  ["Accept", "Reject non-essential", "Got it", "Agree"],
    "game_epl":       ["EPL", "Build a 38-0-0"],
    "mode_classic":   ["Classic"],
    "mode_expert":    ["Expert"],
    "start":          ["Start drafting", "Start draft", "Start"],
    "spin":           ["SPIN", "Spin"],
    "respin":         ["Re-spin", "Respin", "Don't like them"],
    "build_another":  ["Build another", "Play again"],
}
# the column labels a Classic player card shows — used to recognise cards
STAT_LABELS = ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY", "OVR"]

# map the site's position labels -> our internal codes
_POS_ALIASES = {
    "GK": "GK", "GOALKEEPER": "GK",
    "RB": "RB", "LB": "LB", "CB": "CB",
    "RWB": "RWB", "LWB": "LWB", "WB": "RWB",
    "CDM": "CDM", "DM": "CDM", "CM": "CM", "CAM": "CAM", "AM": "CAM",
    "RM": "RM", "LM": "LM", "RW": "RW", "LW": "LW",
    "CF": "CF", "ST": "ST", "STRIKER": "ST", "FW": "ST",
}
_POS_TOKENS = sorted(_POS_ALIASES.keys(), key=len, reverse=True)


def _norm_pos(raw: str) -> str | None:
    raw = (raw or "").strip().upper()
    if raw in _POS_ALIASES:
        return _POS_ALIASES[raw]
    for token in re.split(r"[,/\s]+", raw):
        if token in _POS_ALIASES:
            return _POS_ALIASES[token]
    return raw if raw in ALL_POSITIONS else None


# ───────────────────────── tiny Playwright helpers ─────────────────────────
def click_text(page, texts, timeout=3500, exact=False) -> bool:
    """Click the first visible CONTROL whose text matches any of `texts`.
    Tries an accessible button first, then a short text element. Skips long
    paragraph matches (e.g. the 'Spin to draw a club...' hint) so we hit the
    real button, not instructional copy."""
    if isinstance(texts, str):
        texts = [texts]
    for t in texts:
        # 1) a real button by accessible name (most reliable)
        try:
            loc = page.get_by_role("button", name=t, exact=exact)
            for i in range(min(loc.count(), 5)):
                el = loc.nth(i)
                if el.is_visible():
                    el.click(timeout=timeout)
                    return True
        except Exception:
            pass
        # 2) any short visible element whose text is essentially this label
        try:
            loc = page.get_by_text(t, exact=exact)
            for i in range(min(loc.count(), 8)):
                el = loc.nth(i)
                try:
                    if not el.is_visible():
                        continue
                    own = (el.inner_text(timeout=600) or "").strip()
                    # a control label is short; skip sentences/paragraphs
                    if len(own) > max(24, len(t) + 8):
                        continue
                    el.click(timeout=timeout)
                    return True
                except Exception:
                    continue
        except Exception:
            pass
    return False


def visible_text_present(page, texts) -> bool:
    if isinstance(texts, str):
        texts = [texts]
    for t in texts:
        try:
            if page.get_by_text(t).first.is_visible():
                return True
        except Exception:
            continue
    return False


# ───────── content-based card scanner (injected JS, class-agnostic) ─────────
# Finds the MINIMAL elements that look like a single player card and tags them
# data-bot-card="i". Returns the parsed cards. Works without knowing any class.
_SCANNER = r"""
(posTokens) => {
  const reWordPos = new RegExp('\\b(' + posTokens.join('|') + ')\\b');
  // a "name word" = a token containing a lowercase letter (positions/stats are
  // all-caps, numbers have none) -> distinguishes a real card from a bare chip.
  const hasName = (t) => /[A-Za-z]*[a-z][A-Za-z'\-]*/.test(t);
  const isVisible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 24 || r.height < 16) return false;
    const s = window.getComputedStyle(el);
    return s.display !== 'none' && s.visibility !== 'hidden' && s.opacity !== '0';
  };
  // a card = visible element whose text has a position token AND a name word,
  // bounded length (one player, not a section of prose or a list of cards).
  const isCard = (el) => {
    const t = (el.innerText || '').trim();
    return t && t.length <= 200 && reWordPos.test(t) && hasName(t) && isVisible(el);
  };
  const cands = Array.from(document.querySelectorAll('body *')).filter(isCard);
  // keep only MINIMAL cards: drop any that contains another card element
  const minimal = cands.filter(el => !cands.some(c => c !== el && el.contains(c)));
  const out = [];
  minimal.forEach((el, i) => {
    el.setAttribute('data-bot-card', String(i));
    out.push({ idx: i, text: el.innerText.trim() });
  });
  return out;
}
"""


def parse_player_from_text(text: str) -> Player | None:
    """Pull (name, position, OVR) out of a card's text blob.
    Classic cards read like: 'Bukayo Saka  RW  PAC 86 ... OVR 88'.
    Expert cards may be just: 'Bukayo Saka  RW' (OVR hidden -> 0)."""
    pos = None
    for token in re.findall(r"[A-Z]{2,3}", text.upper()):
        if token in _POS_ALIASES and token not in STAT_LABELS:
            pos = _POS_ALIASES[token]
            break
    if pos is None:
        return None
    # OVR: prefer explicit 'OVR <n>'; else the largest standalone 2-digit 40-99
    m = re.search(r"OVR\D{0,4}(\d{2})", text, re.I)
    if m:
        ovr = int(m.group(1))
    else:
        nums = [int(n) for n in re.findall(r"\b(\d{2})\b", text) if 40 <= int(n) <= 99]
        ovr = max(nums) if nums else 0   # 0 in Expert mode (hidden)
    # name = leading tokens up to the first position / stat label / number
    name_tokens = []
    for tok in re.split(r"\s+", text.strip()):
        base = tok.strip(",.|·-").strip()
        if not base:
            continue
        if _norm_pos(base) is not None or base.upper() in STAT_LABELS or re.fullmatch(r"\d+", base):
            break
        name_tokens.append(tok)
    name = " ".join(name_tokens)[:40] or "?"
    return Player(name, pos, ovr)


def scan_cards(page) -> list[tuple[int, Player]]:
    """Return [(data_bot_card_index, Player)] for the cards currently on screen."""
    try:
        raw = page.evaluate(_SCANNER, _POS_TOKENS)
    except Exception as e:
        print(f"   (card scan error: {e})")
        return []
    out = []
    for item in raw or []:
        p = parse_player_from_text(item["text"])
        if p:
            out.append((item["idx"], p))
    return out

def parse_prow(text: str):
    """Parse a .prow row like 'NK N. Kanté CDM/CMAge 30 · #7 78 66 75 ... 90'."""
    flat = re.sub(r"\s+", " ", text).strip()
    if not flat:
        return None
    mnum = re.search(r"\d", flat)
    head = flat[:mnum.start()] if mnum else flat        # 'RL R. Lukaku STAge '

    # --- position: match a known position token immediately followed by 'Age'
    #     (handles 'STAge', 'CBAge', 'CDM/CMAge' without splitting letters)
    pos = None
    mpos = re.search(r"(GK|RWB|LWB|CDM|CAM|RB|LB|CB|CM|RM|LM|RW|LW|CF|ST)Age", head)
    if mpos:
        pos = _norm_pos(mpos.group(1))
    if pos is None:                                      # fallback: any spaced token
        for tk in re.findall(r"\b[A-Za-z]{2,3}\b", head):
            np = _norm_pos(tk.upper())
            if np:
                pos = np
    if pos is None:
        return None

    # --- OVR: last number in the row
    nums = re.findall(r"\d{1,2}", flat)
    ovr = int(nums[-1]) if nums else 0

    # --- name: strip the trailing position/Age block, drop the initials chip
    name_part = re.sub(r"\s*[A-Z][A-Za-z/]*Age.*$", "", head)     # remove 'STAge ...' tail
    name_part = re.sub(r"\s*[A-Z]{2,3}(?:/[A-Z]{2,3})*$", "", name_part).strip()
    toks = name_part.split(" ")
    if toks and toks[0].isupper() and 1 <= len(toks[0]) <= 3:
        toks = toks[1:]                                  # drop 'RL' / 'NK' / 'TS' chip
    name = " ".join(toks).strip() or "?"
    return Player(name, pos, ovr)

def _iter_frames(page):
    seen = []
    def walk(fr):
        seen.append(fr)
        for c in fr.child_frames:
            walk(c)
    walk(page.main_frame)
    return seen

def _safe_count(fr, sel):
    try:
        return fr.evaluate(f"() => document.querySelectorAll({sel!r}).length")
    except Exception:
        return 0

def read_offered_players(page):
    """Read .prow squad rows, tagging each for clicking. Verbose on failure."""
    rows = page.evaluate(
        "() => Array.from(document.querySelectorAll('.prow')).map((el,i) => {"
        "  el.setAttribute('data-bot-card', String(i));"
        "  return { i, text: (el.innerText||'').replace(/\\s+/g,' ').trim() };"
        "})"
    )
    players, index_of = [], {}
    if not rows:
        return players, index_of
    fails = 0
    for r in rows:
        p = parse_prow(r["text"])
        if p:
            players.append(p)
            index_of[p.name] = r["i"]
        else:
            fails += 1
            if fails <= 3:
                print(f"   [read] could not parse row {r['i']}: {r['text'][:80]!r}")
    print(f"   [read] {len(rows)} rows, parsed {len(players)}, failed {fails}")
    return players, index_of


def read_open_slots(page, formation: str, filled: set[int]) -> list[tuple[int, str]]:
    """We track fills ourselves (reliable); the page just confirms positions."""
    slots = formation_slots(formation)
    return [(i, slots[i]) for i in range(len(slots)) if i not in filled]


def read_round(page) -> int | None:
    """Read the site's own 'Round X / 11' counter — ground truth for progress."""
    try:
        body = page.evaluate("() => document.body.innerText")
    except Exception:
        return None
    m = re.search(r"Round\s+(\d+)\s*/\s*\d+", body or "", re.I)
    return int(m.group(1)) if m else None


def squad_signature(players: list[Player]) -> tuple:
    return tuple(sorted((p.name, p.pos, p.ovr) for p in players))


def read_final_record(page) -> str:
    for t in ("PROJECTED RECORD", "Projected record", "38 – 0 – 0", "pts"):
        try:
            loc = page.get_by_text(t).first
            if loc.is_visible():
                # grab a chunk of surrounding text
                return loc.evaluate(
                    "el => (el.closest('section,div,main')||el).innerText").strip()[:400]
        except Exception:
            continue
    return "(record element not found — run --snapshot and share live_snapshot/)"

def click_card(page, player_name, want_pos=None):
    """Two-step draft: click the player's row (matched by name, fresh), then
    click the 'armed' pitch slot for our chosen position."""
    # 1) select the row by matching its visible name text — re-resolved live so
    #    it can't be a stale/detached element after the list re-renders.
    try:
        rows = page.locator(".prow")
        n = rows.count()
        target = None
        for i in range(n):
            try:
                txt = rows.nth(i).inner_text(timeout=800)
            except Exception:
                continue
            if player_name and player_name.split()[-1].lower() in txt.lower():
                target = rows.nth(i)
                break
        if target is None and n:
            target = rows.first
        if target is None:
            return False
        target.scroll_into_view_if_needed(timeout=1500)
        target.click(timeout=2500)
        time.sleep(0.5)
    except Exception as e:
        print(f"   [click] couldn't select the row: {str(e)[:60]}")
        return False

    # 2) click an 'armed' (eligible) pitch slot, preferring our chosen position
    js = """
    (wantPos) => {
      const slots = Array.from(document.querySelectorAll('.slot'));
      const glow = slots.filter(s => /\\barmed\\b/.test(s.className));
      const pool = glow.length ? glow : slots.filter(s => s.className.includes('open'));
      const label = s => (s.innerText||'').trim().toUpperCase();
      let pick = null;
      if (wantPos) pick = pool.find(s => label(s) === wantPos) || pool.find(s => label(s).startsWith(wantPos));
      if (!pick) pick = pool[0];
      if (!pick) return {ok:false};
      pick.setAttribute('data-bot-slot','1');
      return {ok:true};
    }"""
    try:
        info = page.evaluate(js, want_pos)
        if not info.get("ok"):
            print("   [click] no eligible pitch slot lit up")
            return False
        page.locator("[data-bot-slot='1']").first.click(timeout=2500)
        page.evaluate("() => document.querySelectorAll('[data-bot-slot]').forEach(e=>e.removeAttribute('data-bot-slot'))")
        time.sleep(0.5)
        return True
    except Exception as e:
        print(f"   [click] couldn't click the pitch slot: {str(e)[:60]}")
        return False
    
def spin(page):
    """Wait for SPIN to be enabled, click it, wait for the squad to render."""
    dismiss_popups(page)                                  # <-- ADD THIS LINE
    btn = page.locator("button.spin-btn")
    # wait until the button exists AND is not disabled (it's locked until you draft)
    for _ in range(40):                                   # up to ~12s
        try:
            if btn.count() and btn.first.is_enabled() and btn.first.is_visible():
                break
        except Exception:
            pass
        time.sleep(0.3)
    try:
        btn.first.click(timeout=4000)
    except Exception as e:
        print(f"   [spin] could not click SPIN (still disabled?): {str(e)[:60]}")
        return
    for _ in range(33):                                   # wait for rows
        if page.evaluate("() => document.querySelectorAll('.prow').length"):
            break
        time.sleep(0.3)
    time.sleep(0.4)





# ───────────────────────────── setup / flow ────────────────────────────────
def setup_game(page, formation: str, mode: str = "classic"):
    click_text(page, TEXT["cookie_accept"], timeout=2500)
    time.sleep(0.3)
    click_text(page, TEXT["game_epl"], timeout=2500)
    time.sleep(0.3)
    click_text(page, TEXT["mode_classic"] if mode == "classic" else TEXT["mode_expert"],
               timeout=2500)
    time.sleep(0.3)
    # formation label e.g. "4-3-3"
    click_text(page, [formation], timeout=2500, exact=True)
    time.sleep(0.3)
    click_text(page, TEXT["start"], timeout=2500)
    time.sleep(0.8)


# ───────────────────────────── snapshot mode ───────────────────────────────
def snapshot(page, outdir="live_snapshot"):
    os.makedirs(outdir, exist_ok=True)
    # full HTML
    try:
        html = page.content()
        with open(os.path.join(outdir, "page.html"), "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        print("could not save HTML:", e)
    # inventory of clickable elements + card candidates
    inv = page.evaluate(r"""
      (posTokens) => {
        const reP = new RegExp('\\b('+posTokens.join('|')+')\\b');
        const vis = el => { const r = el.getBoundingClientRect();
          const s = getComputedStyle(el);
          return r.width>4 && r.height>4 && s.display!=='none' && s.visibility!=='hidden'; };
        const clickables = [];
        document.querySelectorAll("button,[role=button],a,[onclick]").forEach(el=>{
          if(!vis(el)) return;
          clickables.push({tag:el.tagName, text:(el.innerText||'').trim().slice(0,50),
                           cls:el.className, id:el.id});
        });
        const cards = [];
        const isCard = el => { const t=(el.innerText||'').trim();
          return t && t.length<=200 && reP.test(t) && /[a-z]/.test(t) && vis(el); };
        const all = Array.from(document.querySelectorAll('body *')).filter(isCard);
        const minimal = all.filter(el => !all.some(c => c!==el && el.contains(c)));
        minimal.forEach(el => cards.push({text:el.innerText.trim().slice(0,160),
                                          tag:el.tagName, cls:el.className}));
        return {clickables, cards};
      }
    """, _POS_TOKENS)
    with open(os.path.join(outdir, "inventory.json"), "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2, ensure_ascii=False)
    print(f"\nSaved snapshot to ./{outdir}/")
    print(f"  page.html        ({len(inv['clickables'])} clickable elements)")
    print(f"  inventory.json   ({len(inv['cards'])} player-card candidates detected)")
    print("\nFirst few card candidates the scanner sees:")
    for c in inv["cards"][:8]:
        print("   -", c["text"].replace("\n", " | ")[:90])
    if not inv["cards"]:
        print("   (none — the squad list may not be open; the snapshot was taken")
        print("    after a SPIN so this is unexpected — share the files and I'll fix it.)")


# ───────────────────────────── main loop ───────────────────────────────────
def dismiss_popups(page):
    """Close the Ko-fi donate prompt (and similar) that steals the next click."""
    # most reliable: click by the known ids
    for sel in ("#donateLater", "#donateX"):
        try:
            loc = page.locator(sel)
            if loc.count() and loc.first.is_visible():
                loc.first.click(timeout=1500)
                time.sleep(0.3)
                return
        except Exception:
            pass
    # generic fallback for any future dialog
    for label in ["Maybe later", "No thanks", "Not now", "Later", "Dismiss", "Close", "×", "✕"]:
        try:
            loc = page.get_by_text(label, exact=False)
            for i in range(min(loc.count(), 3)):
                el = loc.nth(i)
                if el.is_visible():
                    el.click(timeout=1500)
                    time.sleep(0.3)
                    return
        except Exception:
            pass

def play(args):
    from playwright.sync_api import sync_playwright

    V = None
    model = os.path.join(os.path.dirname(__file__), "..", "models", f"V_{args.formation}.json")
    if os.path.exists(model):
        V, _ = load_V(model)
        print(f"Loaded value function {model}")

    target = args.target if args.chase else None
    slots = formation_slots(args.formation)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        best_record = None
        for attempt in range(1, args.max_attempts + 1):
            print(f"\n================  ATTEMPT {attempt}/{args.max_attempts}  ================")
            page.goto(args.url, wait_until="domcontentloaded")
            time.sleep(1.2)
            dismiss_popups(page)
            setup_game(page, args.formation, args.mode)

            filled = set()
            picks = []
            guard = 0
            while len(filled) < len(slots) and guard < 60:
                guard += 1
                dismiss_popups(page)
                spin(page)
                dismiss_popups(page)
                players, _ = read_offered_players(page)
                if not players:
                    print("   [play] no players read; retrying"); continue
                open_slots = [(i, slots[i]) for i in range(len(slots)) if i not in filled]
                choice = decide(open_slots, players, V=V, target_ovr=target)
                # chase: respin if below target and respins available
                if choice and choice.get("respin"):
                    if click_text(page, TEXT["respin"], 2500):
                        time.sleep(1.2); continue
                    choice = decide(open_slots, players, V=V, target_ovr=None)
                if not choice or choice.get("respin"):
                    choice = decide(open_slots, players, V=V, target_ovr=None)
                if not choice:
                    print("   [play] no eligible player; respinning")
                    click_text(page, TEXT["respin"], 2500); time.sleep(1.0); continue

                p, idx = choice["player"], choice["slot_index"]
                before = read_round(page)
                ok = click_card(page, p.name, want_pos=slots[idx])
                time.sleep(0.8)
                after = read_round(page)
                # confirm the draft registered: round counter advanced or a slot filled
                advanced = (after is not None and before is not None and after > before)
                if ok and advanced:
                    filled.add(idx); picks.append((slots[idx], p))
                    print(f"  drafted {len(filled)}/11: {p.name} ({p.pos} {p.ovr}) -> {slots[idx]}")
                else:
                    print(f"  retry: draft of {p.name} didn't register (round {before}->{after})")

            rec = read_final_record(page)
            print("\nRESULT:", rec)
            try:
                shot = os.path.abspath(f"road38_attempt_{attempt:02d}.png")
                page.screenshot(path=shot, full_page=True)
                print(f"  saved screenshot -> {shot}")
            except Exception as e:
                print("  screenshot failed:", e)
            is_perfect = ("38" in rec and "0 – 0" in rec) or ("38-0-0" in rec) or ("38–0–0" in rec)
            best_record = rec
            if is_perfect or len(filled) < len(slots):
                if is_perfect:
                    print("\n🏆 38-0-0 ACHIEVED!")
                break
            print("Not 38-0-0 — restarting...")
            click_text(page, ["Build another", "Play again"], 3000)
            time.sleep(1.5)

        # screenshot + keep open
        try:
            shot = os.path.abspath("road38_result.png")
            page.screenshot(path=shot, full_page=True)
            print(f"\nSaved screenshot -> {shot}")
        except Exception as e:
            print("screenshot failed:", e)
        if not args.headless:
            print(f"\nLeaving the browser open for {args.keep_open}s...")
            time.sleep(args.keep_open)
        browser.close()


def main():
    ap = argparse.ArgumentParser(description="Play 38-0-0.com with the trained agent")
    ap.add_argument("--url", default="https://38-0-0.com/")
    ap.add_argument("--formation", default="4-3-3")
    ap.add_argument("--mode", default="classic", choices=["classic", "expert"])
    ap.add_argument("--snapshot", action="store_true",
                    help="spin once, then dump page.html + inventory.json for selector fixing")
    ap.add_argument("--inspect", action="store_true", help="alias for --snapshot")
    ap.add_argument("--dry-run", action="store_true", help="scrape + decide, no clicks")
    ap.add_argument("--chase", action="store_true", help="respin toward 38-0-0 where allowed")
    ap.add_argument("--target", type=float, default=85.0, help="min in-position value when chasing")
    ap.add_argument("--max-respins", type=int, default=10)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=1, help="restart until 38-0-0 (or this many tries)")
    ap.add_argument("--keep-open", type=int, default=60, help="seconds to leave the browser open at the end")
    args = ap.parse_args()
    play(args)


if __name__ == "__main__":
    main()
