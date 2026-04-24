"""
IMM AGI Hub — Terminal Dashboard
Live monitoring display using ANSI escape codes.
No external dependencies — works in any terminal.

Run standalone:  python dashboard.py --url http://localhost:8000
Or auto-launched by main.py when --dashboard flag is set.
"""
from __future__ import annotations
import asyncio
import aiohttp
import argparse
import os
import sys
import time
from datetime import datetime


# ── ANSI helpers ─────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
CYAN   = "\033[36m"
WHITE  = "\033[37m"
BG_DK  = "\033[40m"

def clr(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


# ── Phase colour coding ───────────────────────────────────────────────────────

PHASE_COLORS = {1: GREEN, 2: YELLOW, 3: RED + BOLD}
PHASE_LABELS = {1: "I  COHERENT", 2: "II  COMPRESS", 3: "III  REVIVAL⚡"}

def phase_str(p: int) -> str:
    return clr(f"Ph{PHASE_LABELS.get(p, '?')}", PHASE_COLORS.get(p, WHITE))


# ── Bars ──────────────────────────────────────────────────────────────────────

def bar(value: float, max_val: float, width: int = 12,
        color_ok: str = GREEN, color_warn: str = YELLOW,
        color_crit: str = RED, warn_thresh: float = 0.6,
        crit_thresh: float = 0.85) -> str:
    frac   = min(value / max(max_val, 1e-9), 1.0)
    filled = int(frac * width)
    empty  = width - filled
    col    = (color_crit if frac >= crit_thresh else
              color_warn if frac >= warn_thresh else color_ok)
    return clr("█" * filled, col) + clr("░" * empty, DIM)


# ── Main dashboard renderer ───────────────────────────────────────────────────

class Dashboard:

    def __init__(self, hub_url: str = "http://localhost:8000"):
        self.hub_url  = hub_url.rstrip("/")
        self._data    = {}
        self._last_ok = 0.0

    async def run(self, refresh: float = 3.0):
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(
                        f"{self.hub_url}/status", timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            self._data    = await resp.json()
                            self._last_ok = time.time()
                except Exception:
                    pass  # render stale data with warning

                self._render()
                await asyncio.sleep(refresh)

    # ── Render ────────────────────────────────────────────────────────

    def _render(self):
        _clear()
        hub   = self._data.get("hub",  {})
        risk  = hub.get("risk",   {})
        phases= hub.get("phases", {})
        ngrok = hub.get("ngrok_url", "")
        eas   = self._data.get("connected_eas", [])
        ts    = self._data.get("timestamp", "")
        stale = time.time() - self._last_ok > 10

        # ── Header ────────────────────────────────────────────────────
        print(clr("╔══════════════════════════════════════════════════════╗", CYAN))
        print(clr("║", CYAN)
              + clr("     IMM AGI TRADING HUB  —  MANIFOLD OBSERVER       ", BOLD + WHITE)
              + clr("║", CYAN))
        print(clr("╚══════════════════════════════════════════════════════╝", CYAN))

        status_str = clr("● LIVE", GREEN) if not stale else clr("⚠ STALE", YELLOW)
        dry_run    = clr("[DRY RUN]", YELLOW) if hub.get("dry_run") else ""
        print(f"  {status_str}  {clr(ts[:19], DIM)}  {dry_run}")
        if ngrok:
            print(f"  {clr('ngrok:', DIM)} {clr(ngrok, CYAN)}")
        print()

        # ── Risk gauge ─────────────────────────────────────────────────
        daily_used  = float(risk.get("daily_dd_used",  0))
        daily_hard  = float(risk.get("daily_dd_hard",  500))
        max_used    = float(risk.get("max_dd_used",    0))
        max_hard    = float(risk.get("max_dd_hard",   1000))
        locked      = risk.get("session_locked", False)
        theta       = float(risk.get("theta_trade", -15))
        open_trades = int(risk.get("open_trades",   0))
        acct_size   = float(risk.get("account_size", 10000))

        print(clr("  ── RISK GATE ──────────────────────────────────────", DIM))
        print(f"  Daily DD  {bar(daily_used, daily_hard)}  "
              f"${daily_used:.0f} / ${daily_hard:.0f}  "
              + (clr("LOCKED", RED + BOLD) if locked else ""))
        print(f"  Max DD    {bar(max_used,   max_hard)}  "
              f"${max_used:.0f} / ${max_hard:.0f}")
        print(f"  θ_trade  {clr(f'{theta:.1f}', YELLOW if theta > -20 else GREEN)}"
              f"   Open trades: {clr(str(open_trades), WHITE)}"
              f"   Account: ${acct_size:,.0f}")
        print()

        # ── EA connections ─────────────────────────────────────────────
        print(clr("  ── EA CONNECTIONS ─────────────────────────────────", DIM))
        if eas:
            for ea in eas:
                print(f"  {clr('●', GREEN)}  {ea}")
        else:
            print(f"  {clr('○', RED)}  No EAs connected")
        print()

        # ── Instrument phases ──────────────────────────────────────────
        print(clr("  ── INSTRUMENT PHASES ──────────────────────────────", DIM))
        if phases:
            items = sorted(phases.items())
            # Two-column layout
            col1 = items[:len(items)//2 + 1]
            col2 = items[len(items)//2 + 1:]
            max_rows = max(len(col1), len(col2))
            for i in range(max_rows):
                left  = ""
                right = ""
                if i < len(col1):
                    sym, p = col1[i]
                    left  = f"  {clr(sym.ljust(10), WHITE)}  {phase_str(p)}"
                if i < len(col2):
                    sym, p = col2[i]
                    right = f"  {clr(sym.ljust(10), WHITE)}  {phase_str(p)}"
                print(f"{left:<45}{right}")
        else:
            print(f"  {clr('Waiting for market data...', DIM)}")
        print()

        # ── Signal alert area ─────────────────────────────────────────
        # (populated from phase III events via status endpoint)
        recent_signals = hub.get("recent_signals", [])
        if recent_signals:
            print(clr("  ── RECENT SIGNALS ─────────────────────────────────", DIM))
            for sig in recent_signals[-5:]:
                sym  = sig.get("symbol", "")
                dire = sig.get("direction", "")
                rev  = sig.get("revival_strength", 0)
                S    = sig.get("stability", 0)
                col  = GREEN if dire == "BUY" else RED
                print(
                    f"  {clr('⚡', YELLOW)}  {clr(sym.ljust(8), WHITE)}  "
                    f"{clr(dire, col)}  "
                    f"revival={clr(f'{rev:.2f}', CYAN)}  "
                    f"S={S:.1f}"
                )
            print()

        # ── Footer ────────────────────────────────────────────────────
        print(clr("  ── IMM V3 | doi.org/10.5281/zenodo.19075097 ────────", DIM))
        print()


# ── Standalone entry point ────────────────────────────────────────────────────

async def _main():
    parser = argparse.ArgumentParser(description="IMM AGI Hub Dashboard")
    parser.add_argument("--url",     default="http://localhost:8000")
    parser.add_argument("--refresh", default=3.0, type=float)
    args = parser.parse_args()

    dash = Dashboard(hub_url=args.url)
    print(f"Connecting to hub at {args.url} ...")
    await dash.run(refresh=args.refresh)


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nDashboard closed.")
