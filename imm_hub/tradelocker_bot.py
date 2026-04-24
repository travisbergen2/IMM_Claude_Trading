"""
IMM AGI — TradeLocker Bot Studio
Backtrader strategy — execution hands for the IMM hub brain.

WHY IT WASN'T TRADING (fixed here):
  1. Hub needs ~25 bars of history before Phase II/III can be detected.
     The bot now sends OHLCV data so the hub has a real price history
     to compute spectral gap, entropy, and coherence cycle from.

  2. The Gemini version stripped trade_id, entry_price, current_stop
     from tick posts. Hub uses these to confirm trades and manage exits.
     Restored here.

  3. The poll response was being checked but commands not being consumed
     correctly. The hub returns {"commands": [...]} — now handled properly.

  4. Added debug logging so you can see exactly what the hub is saying.

TIMEFRAME: Use M1 or M5. The hub's phase cycle is 20-35 bars.
  At M1: a Phase II → Phase III cycle takes 20-35 minutes.
  At M5: it takes 100-175 minutes.
  M1 is correct for XAUUSD scalping as designed.

SETUP:
  1. pip install backtrader requests
  2. Set HUB_BASE_URL to your ngrok URL from deploy.py
  3. Run: python tradelocker_bot.py
     Or load IMMStrategy into TradeLocker Bot Studio directly.
"""
from __future__ import annotations
import os
import time
import math
import logging
import requests
import backtrader as bt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("imm_tl")

# ── CONFIG — set these ─────────────────────────────────────────────────────
# Priority order for hub URL:
#   1. IMM_HUB_URL environment variable (set this in Bot Studio / your shell)
#   2. Hardcoded fallback below (update after each deploy.py run)
HUB_BASE_URL = os.environ.get(
    "IMM_HUB_URL",
    "https://unsaid-arvilla-unseeded.ngrok-free.dev"   # update after deploy.py
)
BOT_ID       = os.environ.get("IMM_BOT_ID",  "remora")
SYMBOL       = os.environ.get("IMM_SYMBOL",  "XAUUSD")
POLL_EVERY   = 3    # poll every 3 bars
POST_EVERY   = 1    # post every bar (hub needs dense data to detect phase)
HTTP_TIMEOUT = 3
MIN_LOTS     = 0.01
MAX_LOTS     = 10.0
DEBUG        = True   # set False once trading correctly
# ──────────────────────────────────────────────────────────────────────────


def _post(url: str, data: dict) -> dict | None:
    try:
        r = requests.post(url, json=data, timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        if DEBUG:
            log.warning(f"POST {url} → {r.status_code}: {r.text[:120]}")
    except Exception as e:
        if DEBUG:
            log.debug(f"POST failed {url}: {e}")
    return None


def _get(url: str) -> dict | None:
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        if DEBUG:
            log.debug(f"GET failed {url}: {e}")
    return None


class IMMStrategy(bt.Strategy):
    """
    IMM backtrader strategy for TradeLocker Bot Studio.
    Hub = brain.  This = hands.

    The strategy sends OHLCV + state data every bar so the hub can
    maintain a full spectral gap and coherence cycle computation.
    It polls for ENTER/CLOSE/TRAIL commands and executes them exactly.
    """

    params = dict(
        hub_url    = HUB_BASE_URL,
        bot_id     = BOT_ID,
        symbol     = SYMBOL,
        poll_every = POLL_EVERY,
        post_every = POST_EVERY,
    )

    def __init__(self):
        self.tick_url  = f"{self.p.hub_url}/ea/tick/{self.p.bot_id}"
        self.poll_url  = f"{self.p.hub_url}/ea/poll/{self.p.bot_id}"
        self.event_url = f"{self.p.hub_url}/ea/event/{self.p.bot_id}"
        self.hb_url    = f"{self.p.hub_url}/ea/heartbeat/{self.p.bot_id}"

        self._bar     = 0
        self._last_hb = 0.0

        # Trade state — all fields the hub expects
        self.trade_id     = ""
        self.in_trade     = False
        self.direction    = 0        # 1=BUY  -1=SELL
        self.entry_price  = 0.0
        self.stop_price   = 0.0
        self.exit_wall_up = 0.0
        self.exit_wall_dn = 0.0
        self.order        = None

        # XAUUSD vs forex pip sizing
        self.is_gold = "XAU" in self.p.symbol.upper()
        self.pip     = 0.01 if self.is_gold else 0.0001
        self.decimals = 2  if self.is_gold else 5

        log.info(f"[{self.p.bot_id}] Strategy ready")
        log.info(f"[{self.p.bot_id}] Hub: {self.p.hub_url}")
        log.info(f"[{self.p.bot_id}] Symbol: {self.p.symbol}")
        _get(self.hb_url)

    def start(self):
        log.info(f"[{self.p.bot_id}] Backtrader started — "
                 f"registering with hub at {self.p.hub_url}")
        resp = _get(self.hb_url)
        if resp:
            log.info(f"[{self.p.bot_id}] Hub acknowledged registration")
        else:
            log.warning(f"[{self.p.bot_id}] Hub not reachable — "
                        f"check HUB_BASE_URL and that deploy.py is running")

    # ── Main loop ──────────────────────────────────────────────────────
    def next(self):
        self._bar += 1

        # Post tick + OHLCV to hub every bar (hub needs this for phase calc)
        if self._bar % self.p.post_every == 0:
            self._post_tick()

        # Poll hub for commands
        if self._bar % self.p.poll_every == 0:
            self._poll_commands()

        # Heartbeat every 30 seconds
        now = time.time()
        if now - self._last_hb > 30:
            self._last_hb = now
            _get(self.hb_url)

        # Wall-based exit check
        if self.in_trade:
            self._check_walls()

    # ── Tick post ──────────────────────────────────────────────────────
    def _post_tick(self):
        price = self.data.close[0]
        bid   = round(price, self.decimals)
        ask   = round(price + self.pip * 2, self.decimals)

        # Unrealised PnL
        pnl = 0.0
        if self.in_trade and self.entry_price > 0:
            pnl = (price - self.entry_price) * self.direction * self._pos_size() * 100

        # Synthetic L2 depth (real depth from broker API would go here)
        bids = [[round(bid - i * self.pip, self.decimals), 1000.0]
                for i in range(1, 16)]
        asks = [[round(ask + i * self.pip, self.decimals), 1000.0]
                for i in range(1, 16)]

        # Volume: guard against NaN/inf that some feeds return when unavailable
        raw_vol = float(self.data.volume[0])
        volume  = raw_vol if math.isfinite(raw_vol) and raw_vol > 0 else 1000.0

        payload = {
            "type":          "tick_update",
            "symbol":        self.p.symbol,
            "bid":           bid,
            "ask":           ask,
            # OHLCV — hub uses these for spectral gap and phase detection
            "open":          round(float(self.data.open[0]),  self.decimals),
            "high":          round(float(self.data.high[0]),  self.decimals),
            "low":           round(float(self.data.low[0]),   self.decimals),
            "close":         round(float(self.data.close[0]), self.decimals),
            "volume":        volume,
            "time":          int(time.time()),
            # Trade state — hub needs these to manage exits
            "in_trade":      self.in_trade,
            "trade_id":      self.trade_id,
            "entry_price":   round(self.entry_price,  self.decimals),
            "current_stop":  round(self.stop_price,   self.decimals),
            "lots":          self._pos_size(),
            "open_pnl":      round(pnl, 2),
            # Depth
            "bids":          bids,
            "asks":          asks,
        }

        resp = _post(self.tick_url, payload)
        if DEBUG and self._bar % 10 == 0:
            phase = resp.get("phase", "?") if resp else "no_resp"
            log.info(f"[{self.p.bot_id}] Bar {self._bar} | "
                     f"price={bid} | hub_phase={phase} | "
                     f"in_trade={self.in_trade}")

    # ── Command poll ───────────────────────────────────────────────────
    def _poll_commands(self):
        resp = _get(self.poll_url)
        if not resp:
            return

        commands = resp.get("commands", [])
        if DEBUG and commands:
            log.info(f"[{self.p.bot_id}] Received {len(commands)} command(s): "
                     f"{[c.get('type') for c in commands]}")

        for cmd in commands:
            self._handle_command(cmd)

    def _handle_command(self, cmd: dict):
        t = cmd.get("type", "")
        if   t == "ENTER":           self._enter(cmd)
        elif t == "CLOSE_ALL":       self._close_all(cmd.get("reason", "hub"))
        elif t == "CLOSE_PARTIAL":   self._close_partial(float(cmd.get("close_fraction", 0.5)))
        elif t in ("TRAIL_STOP", "MOVE_TO_BE_PLUS"):
                                     self._trail(float(cmd.get("new_stop", 0)))
        elif DEBUG:
            log.debug(f"[{self.p.bot_id}] Unknown command type: {t}")

    # ── Order handlers ─────────────────────────────────────────────────
    def _enter(self, cmd: dict):
        if self.in_trade or self.order:
            log.info(f"[{self.p.bot_id}] ENTER ignored — already in trade")
            return

        direction = cmd.get("direction", "BUY")
        lots      = max(MIN_LOTS, min(MAX_LOTS, float(cmd.get("lots", 0.01))))
        sl_pips   = int(cmd.get("initial_sl_pips", 18))
        price     = float(self.data.close[0])

        self.trade_id     = cmd.get("trade_id", f"t{self._bar}")
        self.exit_wall_up = float(cmd.get("exit_target_up", 0))
        self.exit_wall_dn = float(cmd.get("exit_target_dn", 0))
        self.direction    = 1 if direction == "BUY" else -1
        self.stop_price   = (price - sl_pips * self.pip if self.direction == 1
                             else price + sl_pips * self.pip)

        log.info(f"[{self.p.bot_id}] ENTERING {direction} {lots} lots @ ~{price:.{self.decimals}f} | "
                 f"SL={self.stop_price:.{self.decimals}f} | "
                 f"revival={cmd.get('revival_strength',0):.2f} | "
                 f"S={cmd.get('receiver_stability',0):.1f}")

        if direction == "BUY":
            self.order = self.buy(size=lots)
        else:
            self.order = self.sell(size=lots)

    def _close_all(self, reason: str = "hub"):
        if not self.in_trade:
            return
        cp = round(float(self.data.close[0]), self.decimals)
        log.info(f"[{self.p.bot_id}] CLOSING @ {cp} | reason={reason}")
        self.close()
        _post(self.event_url, {
            "type":        "trade_closed",
            "trade_id":    self.trade_id,
            "close_price": cp,
            "reason":      reason,
        })
        self._reset()

    def _close_partial(self, fraction: float):
        if not self.in_trade:
            return
        size   = self._pos_size()
        reduce = max(MIN_LOTS, round(size * fraction, 2))
        log.info(f"[{self.p.bot_id}] PARTIAL CLOSE {fraction:.0%} ({reduce} lots)")
        if self.direction == 1:
            self.sell(size=reduce)
        else:
            self.buy(size=reduce)

    def _trail(self, new_stop: float):
        if not self.in_trade or new_stop <= 0:
            return
        if (self.direction == 1 and new_stop > self.stop_price) or \
           (self.direction == -1 and new_stop < self.stop_price):
            self.stop_price = new_stop
            log.info(f"[{self.p.bot_id}] TRAIL stop → {new_stop:.{self.decimals}f}")

    def _check_walls(self):
        price = float(self.data.close[0])
        if self.direction == 1 and self.exit_wall_up > 0:
            if price >= self.exit_wall_up * 0.999:
                self._close_all("wall_resistance_hit")
        elif self.direction == -1 and self.exit_wall_dn > 0:
            if price <= self.exit_wall_dn * 1.001:
                self._close_all("wall_support_hit")

    # ── Backtrader order/trade notifications ───────────────────────────
    def notify_order(self, order):
        if order.status == order.Completed:
            self.in_trade    = True
            self.entry_price = round(order.executed.price, self.decimals)
            self.order       = None
            log.info(f"[{self.p.bot_id}] ORDER FILLED @ {self.entry_price} | "
                     f"size={order.executed.size:.2f}")
            _post(self.event_url, {
                "type":        "trade_opened",
                "trade_id":    self.trade_id,
                "symbol":      self.p.symbol,
                "entry_price": self.entry_price,
                "lots":        order.executed.size,
            })

        elif order.status in (order.Canceled, order.Rejected, order.Margin):
            log.warning(f"[{self.p.bot_id}] Order {order.getstatusname()} — "
                        f"reason: {order.info.get('reject_reason','unknown')}")
            self.order = None
            self._reset()

    def notify_trade(self, trade):
        if trade.isclosed:
            log.info(f"[{self.p.bot_id}] Trade PnL gross={trade.pnl:.2f} "
                     f"net={trade.pnlcomm:.2f}")

    def stop(self):
        log.info(f"[{self.p.bot_id}] Backtrader ended | "
                 f"Final value: {self.broker.getvalue():.2f}")

    # ── Helpers ────────────────────────────────────────────────────────
    def _pos_size(self) -> float:
        pos = self.broker.getposition(self.data)
        return abs(float(pos.size)) if pos else 0.0

    def _reset(self):
        self.in_trade     = False
        self.trade_id     = ""
        self.direction    = 0
        self.entry_price  = 0.0
        self.stop_price   = 0.0
        self.exit_wall_up = 0.0
        self.exit_wall_dn = 0.0


# ── Standalone run for testing ─────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from backtrader.feeds import PandasData

    print(f"\n  IMM TradeLocker Bot — backtrader")
    print(f"  Hub:    {HUB_BASE_URL}")
    print(f"  Bot ID: {BOT_ID}")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Debug:  {DEBUG}")
    print()
    print("  Checking hub connection...")
    hb = _get(f"{HUB_BASE_URL}/ea/heartbeat/{BOT_ID}")
    if hb:
        print(f"  Hub CONNECTED: {hb}")
    else:
        print(f"  Hub NOT REACHABLE — start deploy.py first, then run this bot")
    print()

    # Synthetic M1 data — replace with your TradeLocker live feed
    rng = np.random.default_rng(42)
    n   = 500
    px  = 3300.0 + np.cumsum(rng.standard_normal(n) * 1.5)
    df  = pd.DataFrame({
        "open":   px + rng.uniform(-0.5, 0.5, n),
        "high":   px + rng.uniform(0, 3, n),
        "low":    px - rng.uniform(0, 3, n),
        "close":  px,
        "volume": rng.uniform(200, 2000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="1min"))

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        IMMStrategy,
        hub_url    = HUB_BASE_URL,
        bot_id     = BOT_ID,
        symbol     = SYMBOL,
        poll_every = POLL_EVERY,
        post_every = POST_EVERY,
    )
    cerebro.adddata(PandasData(dataname=df), name=SYMBOL)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)

    print("  Running 500 bars of synthetic M1 data...")
    print("  Watch for Phase III signals in the hub console.\n")
    cerebro.run()
    print(f"\n  Final portfolio value: {cerebro.broker.getvalue():.2f}")
