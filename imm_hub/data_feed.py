"""
IMM AGI Hub — Data Feed Manager
Handles live price data for forex pairs that yfinance covers poorly.

Priority order per symbol:
  1. EA tick feed (pushed from MT5 via WebSocket — most accurate)
  2. yfinance (5-min bars — acceptable for manifold-level analysis)
  3. Alpha Vantage free API (backup, requires free API key)

For production at scale, replace with broker's REST/WS API.
"""
from __future__ import annotations
import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Deque, Optional, Tuple
import numpy as np

log = logging.getLogger("data_feed")

# yfinance ticker symbols for forex pairs
YF_FOREX_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "USDCHF": "CHF=X",
    "NZDUSD": "NZDUSD=X",
    "GBPJPY": "GBPJPY=X",
    "EURJPY": "EURJPY=X",
    "AUDJPY": "AUDJPY=X",
    "CADJPY": "CADJPY=X",
    "XAUUSD": "GC=F",       # Gold futures
    "XAGUSD": "SI=F",       # Silver futures
    "DXY":    "DX-Y.NYB",
    "DX-Y.NYB": "DX-Y.NYB",
    "SPX":    "^GSPC",
    "^GSPC":  "^GSPC",
    "TLT":    "TLT",
    "^TNX":   "^TNX",
    "USOIL":  "CL=F",
    "UKOIL":  "BZ=F",
}

# How frequently to fetch (seconds) — yfinance rate limit aware
FETCH_INTERVAL: Dict[str, float] = {
    "default": 60.0,  # 1 min for most
    "macro":   120.0, # 2 min for macro anchors (slower moving)
}

MACRO_ANCHORS = {"DXY", "DX-Y.NYB", "^GSPC", "SPX", "TLT", "^TNX"}


class OHLCVBar:
    __slots__ = ["t", "o", "h", "l", "c", "v"]
    def __init__(self, t, o, h, l, c, v):
        self.t = t; self.o = o; self.h = h
        self.l = l; self.c = c; self.v = v


class SymbolFeed:
    """Rolling OHLCV buffer for one symbol."""

    def __init__(self, symbol: str, maxlen: int = 200):
        self.symbol     = symbol
        self.bars: Deque[OHLCVBar] = deque(maxlen=maxlen)
        self.last_fetch: float = 0.0
        self.last_tick:  float = 0.0  # last EA tick timestamp
        self.tick_bid:   float = 0.0
        self.tick_ask:   float = 0.0
        self.tick_count: int   = 0
        self._lock = asyncio.Lock()

    # -- EA tick injection (real-time) --------------------------------

    def inject_tick(self, bid: float, ask: float):
        self.tick_bid   = bid
        self.tick_ask   = ask
        self.last_tick  = time.time()
        self.tick_count += 1

    @property
    def mid(self) -> float:
        if self.tick_bid > 0 and self.tick_ask > 0:
            return (self.tick_bid + self.tick_ask) / 2.0
        if self.bars:
            return float(self.bars[-1].c)
        return 0.0

    # -- Numpy arrays for analysis ------------------------------------

    def closes(self) -> np.ndarray:
        return np.array([b.c for b in self.bars], dtype=float)

    def opens(self) -> np.ndarray:
        return np.array([b.o for b in self.bars], dtype=float)

    def highs(self) -> np.ndarray:
        return np.array([b.h for b in self.bars], dtype=float)

    def lows(self) -> np.ndarray:
        return np.array([b.l for b in self.bars], dtype=float)

    def volumes(self) -> np.ndarray:
        return np.array([b.v for b in self.bars], dtype=float)

    def returns(self) -> np.ndarray:
        c = self.closes()
        if len(c) < 2:
            return np.zeros(1)
        return np.diff(np.log(c + 1e-12))

    def has_data(self, min_bars: int = 10) -> bool:
        return len(self.bars) >= min_bars

    def add_bars_from_df(self, df):
        """Load pandas DataFrame with OHLCV columns into buffer."""
        for _, row in df.iterrows():
            try:
                bar = OHLCVBar(
                    t=row.name,
                    o=float(row.get("Open",  row.get("open",  0))),
                    h=float(row.get("High",  row.get("high",  0))),
                    l=float(row.get("Low",   row.get("low",   0))),
                    c=float(row.get("Close", row.get("close", 0))),
                    v=float(row.get("Volume",row.get("volume",0))),
                )
                if bar.c > 0:
                    self.bars.append(bar)
            except Exception:
                pass


class DataFeedManager:
    """
    Central data manager.
    Provides unified access to OHLCV data regardless of source.
    """

    def __init__(self, symbols: list, lookback: int = 150):
        self.lookback = lookback
        self.feeds: Dict[str, SymbolFeed] = {
            sym: SymbolFeed(sym, maxlen=lookback)
            for sym in symbols
        }
        self._fetch_tasks_started = False

    # -- Public interface ---------------------------------------------

    def get_feed(self, symbol: str) -> Optional[SymbolFeed]:
        return self.feeds.get(symbol)

    def inject_tick(self, symbol: str, bid: float, ask: float):
        if symbol in self.feeds:
            self.feeds[symbol].inject_tick(bid, ask)

    def all_feeds_with_data(self, min_bars: int = 15) -> Dict[str, SymbolFeed]:
        return {s: f for s, f in self.feeds.items() if f.has_data(min_bars)}

    def summary(self) -> dict:
        return {
            sym: {
                "bars": len(f.bars),
                "ticks": f.tick_count,
                "last_price": round(f.mid, 5),
                "has_ea_feed": f.last_tick > 0,
            }
            for sym, f in self.feeds.items()
        }

    # -- Background fetch loop ----------------------------------------

    async def start_background_fetch(self):
        """Launch async fetch tasks for all symbols."""
        if self._fetch_tasks_started:
            return
        self._fetch_tasks_started = True
        for sym in self.feeds:
            asyncio.create_task(self._fetch_loop(sym))
        log.info(f"Data feed started for {len(self.feeds)} symbols.")

    async def _fetch_loop(self, symbol: str):
        """Continuously refresh OHLCV data for one symbol."""
        interval = (FETCH_INTERVAL["macro"]
                    if symbol in MACRO_ANCHORS
                    else FETCH_INTERVAL["default"])
        while True:
            try:
                await self._fetch_yfinance(symbol)
            except Exception as e:
                log.debug(f"fetch_loop({symbol}): {e}")
            await asyncio.sleep(interval)

    async def _fetch_yfinance(self, symbol: str):
        """Fetch 5-minute OHLCV bars via yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            log.debug("yfinance not available")
            return

        ticker = YF_FOREX_MAP.get(symbol, symbol)
        loop   = asyncio.get_event_loop()

        def _dl():
            return yf.download(
                ticker, period="2d", interval="5m",
                progress=False, auto_adjust=True
            )

        df = await loop.run_in_executor(None, _dl)
        if df is None or df.empty:
            return

        feed = self.feeds.get(symbol)
        if feed:
            feed.add_bars_from_df(df.tail(self.lookback))
            feed.last_fetch = time.time()
            log.debug(
                f"[{symbol}] fetched {len(df)} bars  "
                f"last={float(df['Close'].iloc[-1]):.5f}"
            )

    # -- One-shot initial load ----------------------------------------

    async def load_initial_data(self):
        """
        Fetch initial historical data for all symbols on startup.
        Blocks until all symbols have at least min_bars.
        """
        log.info("Loading initial market data...")
        tasks = [self._fetch_yfinance(sym) for sym in self.feeds]
        await asyncio.gather(*tasks, return_exceptions=True)

        ready = sum(1 for f in self.feeds.values() if f.has_data(15))
        total = len(self.feeds)
        log.info(f"Initial data: {ready}/{total} symbols ready.")

    # -- Synthetic data generator (for dry-run / testing) -------------

    @staticmethod
    def generate_synthetic_bars(
        n: int = 150,
        start_price: float = 1.1000,
        volatility: float = 0.0008,
        drift: float = 0.0,
        seed: int = 42
    ) -> list:
        """
        Generate synthetic OHLCV bars for testing without live data.
        Includes mean-reversion and occasional phase-transition clusters.
        """
        rng  = np.random.default_rng(seed)
        bars = []
        price = start_price

        for i in range(n):
            # Regime switching: every ~30 bars introduce a compression/revival
            phase_mod = np.sin(i / 30.0 * np.pi)
            local_vol = volatility * (1.0 + 0.5 * abs(phase_mod))

            ret   = drift + local_vol * rng.standard_normal()
            price = price * np.exp(ret)

            h = price * (1 + local_vol * 0.5)
            l = price * (1 - local_vol * 0.5)
            o = price * (1 + local_vol * 0.2 * rng.standard_normal())
            v = max(1000 * (1 + rng.exponential(0.5)), 100)

            bars.append(OHLCVBar(
                t=datetime.utcnow() - timedelta(minutes=(n - i) * 5),
                o=float(o), h=float(h), l=float(l), c=float(price), v=float(v)
            ))
        return bars

    def inject_synthetic_data(self, symbol: str,
                               n: int = 150,
                               start_price: float = 1.1000,
                               volatility: float = 0.0008,
                               seed: int = 42):
        """Load synthetic data into a feed (useful for --dry-run testing)."""
        feed = self.feeds.get(symbol)
        if feed is None:
            return
        bars = self.generate_synthetic_bars(n, start_price, volatility, seed)
        for bar in bars:
            feed.bars.append(bar)
        log.debug(f"[{symbol}] injected {n} synthetic bars")
