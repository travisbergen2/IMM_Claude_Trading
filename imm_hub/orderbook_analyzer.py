"""
IMM AGI Hub — Order Book Analyzer
Multi-resolution order book analysis encoding Travis's manual scanning process.

What this module does:
  1. Reads order book at multiple granularities (1.0, 0.1, 0.01 tick resolution)
  2. Detects real walls vs spoofed walls (persistence tracking)
  3. Estimates κ (re-coupling strength) from bid/ask imbalance directly
  4. Detects liquidation clusters (stop-hunt targets)
  5. Identifies whale positioning vs retail FOMO patterns
  6. Scores which timeframe has the most legible Phase III setup

The order book is a higher-dimensional projection of M than price alone.
bid/ask depth at multiple resolutions gives direct access to ρ_env structure.

Travis's process encoded as algorithm:
  Step 1: 1h/4h → macro direction
  Step 2: Order book direction weighting → confirm direction
  Step 3: Zoom 1.0 → 0.1 → 0.01 → find persistent walls
  Step 4: Filter false-flag (spoofed) walls
  Step 5: Enter after liquidation flush, exit at opposite real wall
"""
from __future__ import annotations
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
import numpy as np

log = logging.getLogger("orderbook")

# Granularity levels to analyze (price bucket sizes)
GRANULARITIES = [1.0, 0.1, 0.01]

# A wall is "real" if it persists for this many snapshots when price is within N% of it
SPOOF_PERSISTENCE_REQUIRED = 4    # snapshots
SPOOF_PROXIMITY_PCT         = 0.002  # within 0.2% of price

# Volume threshold to qualify as a whale wall (multiple of average depth volume)
WHALE_WALL_MULTIPLIER = 8.0

# Liquidation cluster: a dense band of stops below/above a recent spike
LIQUIDATION_BAND_WIDTH_PCT = 0.003  # 0.3% band


@dataclass
class OrderBookLevel:
    """Single price level in the order book."""
    price: float
    volume: float
    side: str        # "bid" or "ask"
    granularity: float
    first_seen: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)
    snapshots_present: int = 1
    snapshots_near_price: int = 0   # times price was within proximity
    max_volume: float = 0.0
    disappeared_near_price: bool = False   # spoof indicator


@dataclass
class WallAnalysis:
    """Analysis of a significant order book wall."""
    price: float
    volume: float
    side: str            # "bid" = support wall, "ask" = resistance wall
    granularity: float
    is_real: bool        # passed spoof persistence test
    is_whale: bool       # volume >> average
    strength: float      # 0-1 normalized wall strength
    distance_pct: float  # distance from current price as %
    is_liquidation_target: bool = False  # near a cluster of stops


@dataclass
class OrderBookState:
    """
    Complete order book analysis snapshot.
    This is the high-dimensional projection of M that price alone can't give.
    """
    symbol: str
    timestamp: float
    mid_price: float
    spread: float

    # Direction signal (IMM: which way is ρ_env collapsing)
    bid_ask_imbalance: float   # +1 = all bids, -1 = all asks, 0 = balanced
    imbalance_at_01:   float   # imbalance at 0.1 granularity
    imbalance_at_001:  float   # imbalance at 0.01 granularity (most granular)
    weighted_direction: float  # combined direction signal -1 to +1

    # κ estimate from order book (replaces volume-based approximation)
    kappa_ob: float    # re-coupling strength from imbalance acceleration

    # Wall analysis
    nearest_bid_wall: Optional[WallAnalysis] = None
    nearest_ask_wall: Optional[WallAnalysis] = None
    real_walls_bid:   List[WallAnalysis] = field(default_factory=list)
    real_walls_ask:   List[WallAnalysis] = field(default_factory=list)

    # Whale / manipulation signals
    whale_bid_detected: bool  = False
    whale_ask_detected: bool  = False
    spoof_bid_detected: bool  = False
    spoof_ask_detected: bool  = False
    stop_hunt_setup:    bool  = False   # liquidation cluster visible below/above

    # Cipolla layer detection
    retail_fomo_signal: bool  = False   # thin order book + price momentum = FOMO
    liquidation_target_up:   float = 0.0   # price where retail longs get wiped
    liquidation_target_down: float = 0.0   # price where retail shorts get wiped

    # Phase cycle contribution from order book
    ob_phase_vote: int    = 1    # 1/2/3 — OB's vote on current phase
    ob_confidence: float  = 0.5

    # Trading signals
    entry_signal:   float = 0.0   # -1 to +1: negative=sell, positive=buy
    exit_target_up: float = 0.0   # resistance wall to target for long exit
    exit_target_dn: float = 0.0   # support wall to target for short exit


class OrderBookSnapshot:
    """One point-in-time snapshot of the order book at one granularity."""

    def __init__(self, bids: List[Tuple[float, float]],
                 asks: List[Tuple[float, float]],
                 granularity: float):
        """
        bids: list of (price, volume) sorted descending
        asks: list of (price, volume) sorted ascending
        """
        self.bids        = bids
        self.asks        = asks
        self.granularity = granularity
        self.timestamp   = time.time()

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        return (bb + ba) / 2.0 if bb and ba else 0.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid if self.best_bid and self.best_ask else 0.0

    def total_bid_volume(self, depth: int = 20) -> float:
        return sum(v for _, v in self.bids[:depth])

    def total_ask_volume(self, depth: int = 20) -> float:
        return sum(v for _, v in self.asks[:depth])

    def imbalance(self, depth: int = 20) -> float:
        """
        Bid/ask imbalance ratio.
        +1 = all volume on bid side (buy pressure)
        -1 = all volume on ask side (sell pressure)
        0  = balanced
        """
        bv = self.total_bid_volume(depth)
        av = self.total_ask_volume(depth)
        total = bv + av
        if total < 1e-9:
            return 0.0
        return (bv - av) / total

    def find_walls(self, min_vol_multiple: float = 3.0,
                   n_levels: int = 30) -> List[Tuple[str, float, float]]:
        """
        Find significant walls (large volume clusters).
        Returns: list of (side, price, volume)
        """
        all_levels = (
            [("bid", p, v) for p, v in self.bids[:n_levels]] +
            [("ask", p, v) for p, v in self.asks[:n_levels]]
        )
        if not all_levels:
            return []
        avg_vol = np.mean([v for _, _, v in all_levels])
        return [
            (side, price, vol)
            for side, price, vol in all_levels
            if vol >= avg_vol * min_vol_multiple
        ]


class OrderBookAnalyzer:
    """
    Multi-resolution order book analyzer.
    Encodes the manual scanning process described:
    Macro direction (1h/4h) → OB direction → zoom 1.0→0.1→0.01 →
    wall persistence → spoof filter → entry/exit levels.
    """

    def __init__(self, symbol: str, history_len: int = 30):
        self.symbol      = symbol
        self.history_len = history_len

        # History of snapshots per granularity
        self._snapshots: Dict[float, Deque[OrderBookSnapshot]] = {
            g: deque(maxlen=history_len) for g in GRANULARITIES
        }

        # Wall persistence tracker: (side, price_bucket) → OrderBookLevel
        self._wall_tracker: Dict[Tuple[str, float], OrderBookLevel] = {}

        # Imbalance history for κ estimation
        self._imbalance_history: Deque[float] = deque(maxlen=20)
        self._price_history:     Deque[float] = deque(maxlen=50)

        # Liquidation cluster estimates
        self._recent_spike_high: float = 0.0
        self._recent_spike_low:  float = 0.0
        self._spike_timestamp:   float = 0.0

        self._last_state: Optional[OrderBookState] = None

    # ── Public ───────────────────────────────────────────────────────

    def update(self,
               snapshots: Dict[float, OrderBookSnapshot]) -> OrderBookState:
        """
        Process new order book snapshots at all available granularities.
        snapshots: dict of {granularity: OrderBookSnapshot}
        Returns full OrderBookState.
        """
        for g, snap in snapshots.items():
            if g in self._snapshots:
                self._snapshots[g].append(snap)

        # Use 0.01 (finest) for price tracking, fall back to coarser
        primary = (snapshots.get(0.01) or
                   snapshots.get(0.1)  or
                   snapshots.get(1.0))
        if primary is None:
            return self._empty_state()

        mid = primary.mid
        if mid > 0:
            self._price_history.append(mid)

        # ── Direction analysis ────────────────────────────────────────
        imb_10  = snapshots.get(1.0, primary).imbalance()
        imb_01  = snapshots.get(0.1, primary).imbalance()
        imb_001 = snapshots.get(0.01, primary).imbalance()

        # Weight finer granularity more (it shows committed near-price flow)
        weighted_dir = (0.20 * imb_10 + 0.35 * imb_01 + 0.45 * imb_001)

        self._imbalance_history.append(weighted_dir)

        # ── κ from imbalance acceleration ─────────────────────────────
        kappa_ob = self._estimate_kappa_from_imbalance()

        # ── Wall analysis ─────────────────────────────────────────────
        self._update_wall_tracker(snapshots, mid)
        real_walls_bid, real_walls_ask = self._classify_walls(mid)

        nearest_bid = (max(real_walls_bid, key=lambda w: w.price, default=None)
                       if real_walls_bid else None)
        nearest_ask = (min(real_walls_ask, key=lambda w: w.price, default=None)
                       if real_walls_ask else None)

        # ── Spoof detection ───────────────────────────────────────────
        spoof_bid, spoof_ask = self._detect_spoofs(mid)

        # ── Whale detection ───────────────────────────────────────────
        whale_bid = any(w.is_whale for w in real_walls_bid)
        whale_ask = any(w.is_whale for w in real_walls_ask)

        # ── Liquidation cluster detection ─────────────────────────────
        liq_up, liq_dn, stop_hunt = self._detect_liquidation_clusters(mid)

        # ── Retail FOMO signal ────────────────────────────────────────
        # Thin near-price order book + recent price acceleration = FOMO setup
        retail_fomo = self._detect_retail_fomo(primary, mid)

        # ── Phase vote from order book ────────────────────────────────
        ob_phase, ob_conf = self._ob_phase_vote(
            weighted_dir, kappa_ob, spoof_bid or spoof_ask,
            whale_bid or whale_ask, stop_hunt
        )

        # ── Entry and exit levels ─────────────────────────────────────
        entry_signal  = self._compute_entry_signal(
            weighted_dir, ob_phase, spoof_bid, spoof_ask,
            whale_bid, whale_ask, stop_hunt
        )
        exit_up = nearest_ask.price if nearest_ask else 0.0
        exit_dn = nearest_bid.price if nearest_bid else 0.0

        state = OrderBookState(
            symbol=self.symbol,
            timestamp=time.time(),
            mid_price=mid,
            spread=primary.spread,
            bid_ask_imbalance=imb_10,
            imbalance_at_01=imb_01,
            imbalance_at_001=imb_001,
            weighted_direction=round(weighted_dir, 4),
            kappa_ob=round(kappa_ob, 4),
            nearest_bid_wall=nearest_bid,
            nearest_ask_wall=nearest_ask,
            real_walls_bid=real_walls_bid[:5],
            real_walls_ask=real_walls_ask[:5],
            whale_bid_detected=whale_bid,
            whale_ask_detected=whale_ask,
            spoof_bid_detected=spoof_bid,
            spoof_ask_detected=spoof_ask,
            stop_hunt_setup=stop_hunt,
            retail_fomo_signal=retail_fomo,
            liquidation_target_up=liq_up,
            liquidation_target_down=liq_dn,
            ob_phase_vote=ob_phase,
            ob_confidence=ob_conf,
            entry_signal=round(entry_signal, 4),
            exit_target_up=exit_up,
            exit_target_dn=exit_dn,
        )

        self._last_state = state
        return state

    def last(self) -> Optional[OrderBookState]:
        return self._last_state

    # ── Wall tracking ─────────────────────────────────────────────────

    def _update_wall_tracker(self, snapshots: Dict[float, OrderBookSnapshot],
                              current_price: float):
        """
        Track walls across snapshots to test persistence.
        A wall that disappears when price approaches it is a spoof.
        """
        seen_keys = set()

        for g, snap in snapshots.items():
            walls = snap.find_walls(min_vol_multiple=3.0)
            avg_vol = np.mean([v for _, v in snap.bids[:20]] +
                               [v for _, v in snap.asks[:20]]) if snap.bids else 1.0

            for side, price, vol in walls:
                bucket = round(price / g) * g  # snap to granularity
                key    = (side, bucket)
                seen_keys.add(key)

                proximity = abs(price - current_price) / max(current_price, 1e-9)
                is_near   = proximity < SPOOF_PROXIMITY_PCT

                if key in self._wall_tracker:
                    lvl = self._wall_tracker[key]
                    lvl.last_seen          = time.time()
                    lvl.snapshots_present += 1
                    lvl.max_volume         = max(lvl.max_volume, vol)
                    if is_near:
                        lvl.snapshots_near_price += 1
                else:
                    self._wall_tracker[key] = OrderBookLevel(
                        price=price, volume=vol, side=side,
                        granularity=g,
                        max_volume=vol,
                    )

        # Mark walls that disappeared while near price (spoof signature)
        for key, lvl in list(self._wall_tracker.items()):
            if key not in seen_keys:
                proximity = abs(lvl.price - current_price) / max(current_price, 1e-9)
                if proximity < SPOOF_PROXIMITY_PCT:
                    lvl.disappeared_near_price = True
                # Expire old entries
                if time.time() - lvl.last_seen > 120:  # 2 minutes
                    del self._wall_tracker[key]

    def _classify_walls(self, current_price: float
                         ) -> Tuple[List[WallAnalysis], List[WallAnalysis]]:
        """
        Classify tracked walls as real vs spoofed, whale vs normal.
        Returns (bid_walls, ask_walls) sorted by proximity.
        """
        bid_walls = []
        ask_walls = []

        # Average wall volume for this symbol (for whale detection)
        all_vols = [lvl.max_volume for lvl in self._wall_tracker.values()]
        avg_wall_vol = np.mean(all_vols) if all_vols else 1.0

        for key, lvl in self._wall_tracker.items():
            # Spoof: disappeared when price got close
            if lvl.disappeared_near_price:
                continue

            # Real wall: persisted when price was near OR hasn't been tested yet
            is_real = (lvl.snapshots_present >= SPOOF_PERSISTENCE_REQUIRED or
                       (lvl.snapshots_near_price >= 2 and not lvl.disappeared_near_price))

            is_whale = lvl.max_volume >= avg_wall_vol * WHALE_WALL_MULTIPLIER

            dist_pct = abs(lvl.price - current_price) / max(current_price, 1e-9)
            strength = float(np.clip(
                lvl.max_volume / (avg_wall_vol * WHALE_WALL_MULTIPLIER), 0.0, 1.0
            ))

            wall = WallAnalysis(
                price=lvl.price, volume=lvl.max_volume,
                side=lvl.side, granularity=lvl.granularity,
                is_real=is_real, is_whale=is_whale,
                strength=strength, distance_pct=dist_pct,
            )

            if lvl.side == "bid" and lvl.price < current_price:
                bid_walls.append(wall)
            elif lvl.side == "ask" and lvl.price > current_price:
                ask_walls.append(wall)

        # Sort by proximity
        bid_walls.sort(key=lambda w: w.distance_pct)
        ask_walls.sort(key=lambda w: w.distance_pct)

        return bid_walls, ask_walls

    # ── Spoof detection ───────────────────────────────────────────────

    def _detect_spoofs(self, current_price: float) -> Tuple[bool, bool]:
        """
        Returns (spoof_on_bid, spoof_on_ask).
        Spoof = large wall that disappeared when price was within proximity.
        """
        spoof_bid = False
        spoof_ask = False
        for lvl in self._wall_tracker.values():
            if lvl.disappeared_near_price:
                if lvl.side == "bid":
                    spoof_bid = True
                else:
                    spoof_ask = True
        return spoof_bid, spoof_ask

    # ── κ estimation ──────────────────────────────────────────────────

    def _estimate_kappa_from_imbalance(self) -> float:
        """
        κ (re-coupling strength) from imbalance acceleration.
        When imbalance is increasing in one direction and accelerating,
        the market is building toward a forced revival.

        High |d²(imbalance)/dt²| → κ is high → revival imminent.
        """
        if len(self._imbalance_history) < 4:
            return 0.05

        imb = np.array(list(self._imbalance_history))

        # First derivative: imbalance velocity
        velocity = np.diff(imb)
        # Second derivative: imbalance acceleration
        accel    = np.diff(velocity)

        recent_accel = float(np.mean(np.abs(accel[-3:]))) if len(accel) >= 3 else 0.0
        # Sustained directional imbalance = strong κ
        directional  = float(abs(np.mean(imb[-5:]))) if len(imb) >= 5 else 0.0

        kappa = float(np.clip(0.02 + recent_accel * 2.0 + directional * 0.3, 0.01, 0.80))
        return kappa

    # ── Liquidation cluster detection ────────────────────────────────

    def _detect_liquidation_clusters(self, current_price: float
                                      ) -> Tuple[float, float, bool]:
        """
        Detect likely liquidation clusters from price history.

        The stop-hunt pattern:
        1. Price spikes to a new high (triggers FOMO longs with leverage)
        2. Slow bleed back (building unrealised loss on those longs)
        3. Drop just below the FOMO entry = liquidation cluster is just below spike

        Returns (liq_target_up, liq_target_down, stop_hunt_setup)
        """
        if len(self._price_history) < 10:
            return 0.0, 0.0, False

        prices = np.array(list(self._price_history))

        # Find recent swing high and low
        recent = prices[-20:] if len(prices) >= 20 else prices
        swing_high = float(np.max(recent))
        swing_low  = float(np.min(recent))

        # Liquidation target for longs: just below the FOMO entry
        # = just below the spike high (where retail entered long)
        liq_up = swing_high * (1 - LIQUIDATION_BAND_WIDTH_PCT * 2)

        # Liquidation target for shorts: just above the panic sell low
        liq_dn = swing_low * (1 + LIQUIDATION_BAND_WIDTH_PCT * 2)

        # Stop hunt setup: price is currently between the two levels
        # and has recently visited both (compression between them)
        range_pct = (swing_high - swing_low) / max(swing_low, 1e-9)
        price_in_range = swing_low < current_price < swing_high
        compression = range_pct < 0.015  # within 1.5% — tight compression

        # Check if price recently spiked (1m-type stop hunt precursor)
        if len(prices) >= 5:
            recent_move = abs(prices[-1] - prices[-5]) / max(prices[-5], 1e-9)
            spike_then_bleed = recent_move < 0.003 and range_pct > 0.005
        else:
            spike_then_bleed = False

        stop_hunt = price_in_range and (compression or spike_then_bleed)

        return liq_up, liq_dn, stop_hunt

    # ── Retail FOMO detection ─────────────────────────────────────────

    def _detect_retail_fomo(self, snap: OrderBookSnapshot,
                             current_price: float) -> bool:
        """
        Retail FOMO: thin near-price order book + recent price momentum upward.
        Thin book = low resistance = price can be pushed easily.
        Momentum = retail sees it moving and jumps in.
        This is the Cipolla stupid-actor entry signal — FADE this, don't follow it.
        """
        if not snap.bids or not snap.asks:
            return False

        # Near-price depth: top 5 levels each side
        near_bid_vol = sum(v for _, v in snap.bids[:5])
        near_ask_vol = sum(v for _, v in snap.asks[:5])
        avg_near_vol = (near_bid_vol + near_ask_vol) / 2.0

        # Total depth
        total_bid_vol = snap.total_bid_volume(20)
        total_ask_vol = snap.total_ask_volume(20)
        avg_total = (total_bid_vol + total_ask_vol) / 2.0

        # Thin near-price: top 5 levels have much less volume than average
        thin_book = avg_near_vol < avg_total * 0.15 if avg_total > 0 else False

        # Recent momentum (price acceleration)
        if len(self._price_history) >= 6:
            recent_ret = (float(self._price_history[-1]) -
                          float(self._price_history[-6])) / max(float(self._price_history[-6]), 1e-9)
            momentum_up = recent_ret > 0.003  # 0.3% move in last 5 bars
        else:
            momentum_up = False

        return thin_book and momentum_up

    # ── Phase vote ────────────────────────────────────────────────────

    def _ob_phase_vote(self, direction: float, kappa: float,
                        spoof_active: bool, whale_active: bool,
                        stop_hunt: bool) -> Tuple[int, float]:
        """
        Order book's vote on which Phase we're in.

        Phase I: balanced/clear direction, no manipulation, whale confirming
        Phase II: spoof walls active, stop hunt setup, direction unclear
        Phase III: stop hunt just resolved, direction strong, κ high
        """
        if len(self._imbalance_history) < 5:
            return 1, 0.5

        # Phase III signals
        if kappa > 0.25 and abs(direction) > 0.30 and not spoof_active:
            # Strong directional flow after manipulation cleared
            return 3, float(np.clip(kappa * abs(direction) * 2, 0.5, 1.0))

        # Phase II signals
        if spoof_active or stop_hunt:
            # Active manipulation = compression building
            conf = 0.6 + (0.2 if stop_hunt else 0) + (0.1 if spoof_active else 0)
            return 2, min(conf, 0.95)

        # Phase I: clean directional flow
        if abs(direction) > 0.15 and whale_active:
            return 1, float(np.clip(abs(direction) + 0.3, 0.4, 0.9))

        return 1, 0.4

    # ── Entry signal ──────────────────────────────────────────────────

    def _compute_entry_signal(self, direction: float,
                               phase: int, spoof_bid: bool, spoof_ask: bool,
                               whale_bid: bool, whale_ask: bool,
                               stop_hunt: bool) -> float:
        """
        Entry signal combining all OB analysis.
        +1 = strong buy, -1 = strong sell, 0 = no signal.

        Logic:
        - Phase III + strong direction = primary signal
        - Whale confirmed direction = boost
        - Spoof on opposite side = boost (the fake wall will be pulled = clear path)
        - NEVER enter into a stop hunt setup (we want to enter AFTER the flush)
        - Retail FOMO on same side = reduce (crowded, about to be harvested)
        """
        if phase != 3:
            return 0.0

        signal = direction * 0.6  # base from weighted imbalance

        # Whale on same side as direction = institutional confirmation
        if direction > 0 and whale_bid:
            signal += 0.2
        if direction < 0 and whale_ask:
            signal += 0.2  # note: signal is negative, this adds magnitude

        # Spoof on opposite side = path is clearing
        if direction > 0 and spoof_ask:
            signal += 0.15   # the ask wall will be pulled, buy signal
        if direction < 0 and spoof_bid:
            signal -= 0.15   # the bid wall will be pulled, sell signal

        # Stop hunt active = NOT the right entry time, wait for flush
        if stop_hunt:
            signal *= 0.3   # suppress signal until flush is complete

        return float(np.clip(signal, -1.0, 1.0))

    # ── Helpers ───────────────────────────────────────────────────────

    def _empty_state(self) -> OrderBookState:
        return OrderBookState(
            symbol=self.symbol,
            timestamp=time.time(),
            mid_price=0.0, spread=0.0,
            bid_ask_imbalance=0.0,
            imbalance_at_01=0.0,
            imbalance_at_001=0.0,
            weighted_direction=0.0,
            kappa_ob=0.05,
            ob_phase_vote=1, ob_confidence=0.5,
            entry_signal=0.0,
        )

    def summary(self) -> str:
        s = self._last_state
        if not s:
            return f"[{self.symbol}] No OB data"
        whale = "🐋" if (s.whale_bid_detected or s.whale_ask_detected) else ""
        spoof = "⚠" if (s.spoof_bid_detected or s.spoof_ask_detected) else ""
        hunt  = "🎯" if s.stop_hunt_setup else ""
        fomo  = "🐑" if s.retail_fomo_signal else ""
        return (
            f"[{self.symbol}] dir={s.weighted_direction:+.3f}  "
            f"κ={s.kappa_ob:.3f}  ph={s.ob_phase_vote}  "
            f"sig={s.entry_signal:+.3f}  {whale}{spoof}{hunt}{fomo}"
        )


    def build_synthetic_snapshot(self, symbol: str, mid: float,
                                   direction_bias: float = 0.0,
                                   regime: str = "normal"
                                   ):
        """Build synthetic order book snapshots (convenience method on analyzer)."""
        rng = np.random.default_rng(int(mid * 1000) % 2**31)
        result = {}
        for g in GRANULARITIES:
            n_levels = 30
            base_vol = 1000.0 / g
            bids = []
            for i in range(n_levels):
                price = round(mid - (i + 1) * g, max(2, len(str(g).split(".")[-1])))
                vol = max(base_vol * (1 + 0.3 * rng.standard_normal()), 10.0)
                if direction_bias > 0 and i < 10:
                    vol *= (1 + direction_bias * 0.5)
                if regime == "whale_bid" and i == 3:
                    vol *= WHALE_WALL_MULTIPLIER * 2
                if regime == "stop_hunt" and i < 3:
                    vol *= 0.3
                bids.append((price, vol))
            asks = []
            for i in range(n_levels):
                price = round(mid + (i + 1) * g, max(2, len(str(g).split(".")[-1])))
                vol = max(base_vol * (1 + 0.3 * rng.standard_normal()), 10.0)
                if direction_bias < 0 and i < 10:
                    vol *= (1 + abs(direction_bias) * 0.5)
                if regime == "whale_ask" and i == 3:
                    vol *= WHALE_WALL_MULTIPLIER * 2
                if regime == "spoof" and i == 5:
                    vol *= 6.0
                asks.append((price, vol))
            result[g] = OrderBookSnapshot(bids, asks, g)
        return result


class OrderBookRegistry:
    """Manages one OrderBookAnalyzer per instrument."""

    def __init__(self, symbols: list):
        self._analyzers: Dict[str, OrderBookAnalyzer] = {
            sym: OrderBookAnalyzer(sym) for sym in symbols
        }

    def update(self, symbol: str,
               snapshots: Dict[float, OrderBookSnapshot]) -> OrderBookState:
        if symbol not in self._analyzers:
            self._analyzers[symbol] = OrderBookAnalyzer(symbol)
        return self._analyzers[symbol].update(snapshots)

    def get_state(self, symbol: str) -> Optional[OrderBookState]:
        a = self._analyzers.get(symbol)
        return a.last() if a else None

    def build_synthetic_snapshot(self, symbol: str, mid: float,
                                   direction_bias: float = 0.0,
                                   regime: str = "normal"
                                   ) -> Dict[float, "OrderBookSnapshot"]:
        """Delegate to analyzer's snapshot builder."""
        a = self._analyzers.get(symbol)
        if a is None:
            self._analyzers[symbol] = OrderBookAnalyzer(symbol)
            a = self._analyzers[symbol]
        return a.build_synthetic_snapshot(symbol, mid, direction_bias, regime)

    def get_kappa(self, symbol: str) -> float:
        """Drop-in replacement for volume-based κ estimate."""
        state = self.get_state(symbol)
        return state.kappa_ob if state else 0.05

    def get_direction(self, symbol: str) -> float:
        """Weighted direction from OB multi-granularity."""
        state = self.get_state(symbol)
        return state.weighted_direction if state else 0.0

    def get_phase_vote(self, symbol: str) -> Tuple[int, float]:
        state = self.get_state(symbol)
        if state:
            return state.ob_phase_vote, state.ob_confidence
        return 1, 0.5

    def log_all(self):
        for sym, a in self._analyzers.items():
            log.info(a.summary())

    def build_synthetic_snapshot(self, symbol: str, mid: float,
                                   direction_bias: float = 0.0,
                                   regime: str = "normal"
                                   ) -> Dict[float, OrderBookSnapshot]:
        """
        Build synthetic order book snapshots for testing/dry-run.
        direction_bias: +1 = more bids, -1 = more asks
        regime: normal | stop_hunt | whale_bid | whale_ask | spoof
        """
        rng = np.random.default_rng(int(time.time() * 1000) % 2**31)
        result = {}

        for g in GRANULARITIES:
            n_levels = 30
            base_vol = 1000.0 / g   # finer granularity = more levels, lower per-level vol

            # Bids: price levels below mid
            bids = []
            for i in range(n_levels):
                price = round((mid - (i + 1) * g), max(2, len(str(g).split(".")[-1])))
                vol   = base_vol * (1 + 0.3 * rng.standard_normal())
                vol   = max(vol, 10.0)

                # Direction bias
                if direction_bias > 0 and i < 10:
                    vol *= (1 + direction_bias * 0.5)

                # Regime modifications
                if regime == "whale_bid" and i == 3:
                    vol *= WHALE_WALL_MULTIPLIER * 2
                if regime == "stop_hunt" and i < 3:
                    vol *= 0.3   # thin book near price (about to flush)

                bids.append((price, vol))

            # Asks: price levels above mid
            asks = []
            for i in range(n_levels):
                price = round((mid + (i + 1) * g), max(2, len(str(g).split(".")[-1])))
                vol   = base_vol * (1 + 0.3 * rng.standard_normal())
                vol   = max(vol, 10.0)

                if direction_bias < 0 and i < 10:
                    vol *= (1 + abs(direction_bias) * 0.5)

                if regime == "whale_ask" and i == 3:
                    vol *= WHALE_WALL_MULTIPLIER * 2
                if regime == "spoof" and i == 5:
                    vol *= 6.0   # large ask wall that will "disappear"

                asks.append((price, vol))

            result[g] = OrderBookSnapshot(bids, asks, g)

        return result
