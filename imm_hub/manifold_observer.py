"""
IMM AGI Hub — Manifold Observer
Fetches market data and computes per-instrument ManifoldState each cycle.
Uses yfinance for macro anchors + EA tick feeds for tradeable pairs.
Swap out _fetch_ohlcv() for any live data feed.
"""
from __future__ import annotations
import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Deque, Optional
import numpy as np

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

from models import Config, ManifoldState, Direction, Phase
from receiver_array import ReceiverArray

log = logging.getLogger("manifold_observer")

# Macro anchor → yfinance ticker map
MACRO_YF_MAP = {
    "DXY": "DX-Y.NYB",
    "DX-Y.NYB": "DX-Y.NYB",
    "SPX": "^GSPC",
    "^GSPC": "^GSPC",
    "TLT": "TLT",
    "^TNX": "^TNX",
}

# pip value approximations (USD per pip per 0.01 lot)
PIP_VALUES = {
    "EURUSD": 0.10, "GBPUSD": 0.10, "AUDUSD": 0.10, "NZDUSD": 0.10,
    "USDJPY": 0.009, "USDCAD": 0.075, "USDCHF": 0.11,
    "GBPJPY": 0.009, "EURJPY": 0.009, "AUDJPY": 0.009,
    "XAUUSD": 0.10, "XAGUSD": 0.10,
    "DEFAULT": 0.10,
}


class ManifoldObserver:
    """
    Maintains a rolling window of OHLCV data per instrument and computes
    the full ManifoldState each update cycle.
    """

    def __init__(self, config: Config, receiver_array: ReceiverArray):
        self.config  = config
        self.ra      = receiver_array
        self.lookback = config.data_lookback

        # Rolling buffers: symbol → deque of close prices
        all_syms = list(set(config.tradeable + config.macro_anchors))
        self._closes: Dict[str, Deque[float]] = {
            s: deque(maxlen=self.lookback) for s in all_syms
        }
        self._volumes: Dict[str, Deque[float]] = {
            s: deque(maxlen=self.lookback) for s in all_syms
        }
        self._highs:  Dict[str, Deque[float]] = {s: deque(maxlen=self.lookback) for s in all_syms}
        self._lows:   Dict[str, Deque[float]] = {s: deque(maxlen=self.lookback) for s in all_syms}

        # Per-instrument rho_env tracker
        self._rho_env: Dict[str, float] = {s: 0.0 for s in all_syms}
        self._N_tot:   Dict[str, float] = {s: 1.0 for s in all_syms}
        self._kappa:   Dict[str, float] = {s: 0.01 for s in all_syms}

        # Last computed states
        self._states: Dict[str, ManifoldState] = {}

        # EA tick feed override (populated by hub_server when EAs push ticks)
        self.ea_tick_feed: Dict[str, float] = {}

    # ── Public ───────────────────────────────────────────────────────

    async def update_all(self) -> Dict[str, ManifoldState]:
        """Fetch latest data and compute states for all instruments."""
        tasks = []
        all_syms = list(set(self.config.tradeable + self.config.macro_anchors))
        for sym in all_syms:
            tasks.append(self._update_symbol(sym))
        await asyncio.gather(*tasks, return_exceptions=True)
        return dict(self._states)

    def inject_ea_tick(self, symbol: str, bid: float, ask: float,
                       volume: float = 0.0):
        """Called by hub_server when an EA pushes a tick update."""
        mid = (bid + ask) / 2.0
        self.ea_tick_feed[symbol] = mid
        # Push into rolling buffer immediately
        closes = self._closes.get(symbol)
        if closes is not None:
            closes.append(mid)

    def get_state(self, symbol: str) -> Optional[ManifoldState]:
        return self._states.get(symbol)

    # ── Per-symbol update ─────────────────────────────────────────────

    async def _update_symbol(self, symbol: str):
        try:
            await self._fetch_and_buffer(symbol)
            state = self._compute_state(symbol)
            self._states[symbol] = state
        except Exception as e:
            log.warning(f"update_symbol({symbol}) failed: {e}")

    async def _fetch_and_buffer(self, symbol: str):
        """Fetch recent OHLCV and update rolling buffers."""
        # For macro anchors, use yfinance
        yf_ticker = MACRO_YF_MAP.get(symbol)
        if yf_ticker and YF_AVAILABLE:
            await asyncio.get_event_loop().run_in_executor(
                None, self._yf_fetch, symbol, yf_ticker
            )
        # EA-fed symbols: buffers already updated by inject_ea_tick
        # If no live data yet, buffer stays as-is

    def _yf_fetch(self, symbol: str, ticker: str):
        try:
            df = yf.download(ticker, period="5d", interval="5m",
                             progress=False, auto_adjust=True)
            if df.empty:
                return
            closes  = df["Close"].dropna().values[-self.lookback:]
            volumes = df["Volume"].dropna().values[-self.lookback:]
            highs   = df["High"].dropna().values[-self.lookback:]
            lows    = df["Low"].dropna().values[-self.lookback:]

            for c in closes:  self._closes[symbol].append(float(c))
            for v in volumes: self._volumes[symbol].append(float(v))
            for h in highs:   self._highs[symbol].append(float(h))
            for l in lows:    self._lows[symbol].append(float(l))
        except Exception as e:
            log.debug(f"yf_fetch({ticker}): {e}")

    # ── State computation ─────────────────────────────────────────────

    def _compute_state(self, symbol: str) -> ManifoldState:
        closes  = np.array(list(self._closes[symbol]))
        volumes = np.array(list(self._volumes[symbol]))
        highs   = np.array(list(self._highs[symbol]))
        lows    = np.array(list(self._lows[symbol]))

        if len(closes) < 10:
            return self._empty_state(symbol)

        # Log returns
        returns = np.diff(np.log(closes + 1e-12))

        # ── Spectral gap ──────────────────────────────────────────────
        delta = self._estimate_spectral_gap(returns)
        grad_delta = self._estimate_grad_delta(returns)

        # ── Entropy ───────────────────────────────────────────────────
        H_market = self._estimate_entropy(returns)

        # ── Time pressure: normalised volatility spike ────────────────
        recent_vol  = float(np.std(returns[-10:])) if len(returns) >= 10 else 0.05
        baseline_vol = float(np.std(returns)) if len(returns) > 10 else 0.05
        P_time = float(np.clip(recent_vol / (baseline_vol + 1e-9), 0.0, 1.0))

        # ── ATR ───────────────────────────────────────────────────────
        atr = self._compute_atr(highs, lows, closes)

        # ── Coherence cycle tracking ──────────────────────────────────
        psi_sq, rho_env, N_tot, kappa, gamma_rate = \
            self._update_coherence_cycle(symbol, returns, delta, volumes)

        revival_threshold = (kappa * rho_env) > gamma_rate

        # ── Receiver step ─────────────────────────────────────────────
        R_new   = self.ra.step(symbol, H_market, P_time)
        R_char  = self.ra.get_R_char(symbol)
        V       = self.ra.V(R_new, H_market, P_time)
        S       = -V

        # ── Phase detection ───────────────────────────────────────────
        phase, confidence = self._detect_phase(
            psi_sq, rho_env, N_tot, delta, revival_threshold)

        # ── Gradient direction ────────────────────────────────────────
        direction = self._infer_direction(returns, delta, grad_delta)

        current_price = float(closes[-1]) if len(closes) > 0 else 0.0

        return ManifoldState(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            phase=phase,
            phase_confidence=confidence,
            delta=delta,
            grad_delta=grad_delta,
            psi_sq=psi_sq,
            rho_env=rho_env,
            N_tot=N_tot,
            kappa=kappa,
            gamma_rate=gamma_rate,
            revival_threshold=revival_threshold,
            H_market=H_market,
            P_time=P_time,
            R=R_new,
            R_char=R_char,
            V=V,
            S=S,
            current_price=current_price,
            atr=atr,
            returns=returns[-50:] if len(returns) >= 50 else returns,
            gradient_direction=direction,
        )

    # ── Maths ─────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_spectral_gap(returns: np.ndarray, lags: int = 20) -> float:
        """
        Estimate Δ from autocorrelation decay rate.
        log|ACF(k)| ≈ -Δ·k  →  slope gives Δ.
        Fast decay = large Δ (stable). Slow decay = small Δ (bifurcation near).
        """
        n = len(returns)
        if n < lags + 2:
            return 0.5
        # Guard against flat / zero-variance data
        if np.std(returns) < 1e-12:
            return 0.5
        valid_lags, valid_log_acf = [], []
        for lag in range(1, min(lags, n // 2)):
            x = returns[:-lag]
            y = returns[lag:]
            # Skip if either window has no variance
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue
            with np.errstate(invalid='ignore', divide='ignore'):
                a = np.corrcoef(x, y)[0, 1]
            if np.isfinite(a) and abs(a) > 0.01:
                valid_lags.append(lag)
                valid_log_acf.append(np.log(abs(a)))
        if len(valid_lags) < 3:
            return 0.5
        slope, _ = np.polyfit(valid_lags, valid_log_acf, 1)
        return float(np.clip(abs(slope), 0.001, 2.0))

    @staticmethod
    def _estimate_grad_delta(returns: np.ndarray,
                              window: int = 10) -> float:
        """
        ∇Δ estimated as rate of change of spectral gap across two windows.
        Positive = spectral gap increasing (stabilising).
        Negative = spectral gap decreasing (bifurcation building).
        """
        if len(returns) < 2 * window + 4:
            return 0.0
        from manifold_observer import ManifoldObserver as _M  # avoid circular
        d_recent = _M._estimate_spectral_gap(returns[-window:],     lags=8)
        d_prior  = _M._estimate_spectral_gap(returns[-2*window:-window], lags=8)
        return float(d_recent - d_prior)

    @staticmethod
    def _estimate_entropy(returns: np.ndarray, bins: int = 20) -> float:
        """Shannon entropy of return distribution, normalised to [0,1]."""
        if len(returns) < 5:
            return 0.5
        if np.std(returns) < 1e-12:
            return 0.0   # flat returns = zero entropy
        with np.errstate(invalid='ignore', divide='ignore'):
            hist, _ = np.histogram(returns, bins=bins, density=True)
            hist = hist[hist > 0]
            if len(hist) == 0:
                return 0.5
            raw = -float(np.sum(hist * np.log(hist + 1e-12)))
            log_bins = np.log(max(bins, 2))
            return float(np.clip(raw / log_bins, 0.0, 1.0))

    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int = 14) -> float:
        if len(highs) < 2:
            return 0.001
        tr = np.maximum(highs[1:] - lows[1:],
             np.maximum(abs(highs[1:] - closes[:-1]),
                        abs(lows[1:]  - closes[:-1])))
        n = min(period, len(tr))
        return float(np.mean(tr[-n:]))

    def _update_coherence_cycle(
        self, symbol: str,
        returns: np.ndarray,
        delta: float,
        volumes: np.ndarray,
    ):
        """
        Track ψ², ρ_env, N_tot dynamics.

        ψ² proxy: trend coherence = (directional ratio of recent returns)²
        ρ_env: accumulated load from Phase II transfer; decays with κ
        N_tot: conserved total (corrected by tolerance)
        κ: re-coupling strength from bid-ask proxy (volume acceleration)
        γ: ongoing projection pressure from entropy × vol
        """
        if len(returns) < 5:
            return 1.0, 0.0, 1.0, 0.01, 0.02

        # ψ² = fraction of recent bars moving in dominant direction (squared)
        window = min(20, len(returns))
        r_win  = returns[-window:]
        n_up   = int(np.sum(r_win > 0))
        n_dn   = window - n_up
        dom_frac = max(n_up, n_dn) / max(window, 1)
        psi_sq_raw = dom_frac ** 2   # ∈ [0.25, 1.0]

        # Normalise psi_sq relative to running N_tot baseline
        N_prev = self._N_tot[symbol]
        psi_sq = float(np.clip(psi_sq_raw * N_prev, 0.0, N_prev))

        # γ = projection pressure = entropy × recent vol
        recent_vol  = float(np.std(returns[-5:])) if len(returns) >= 5 else 0.02
        H           = self._estimate_entropy(returns)
        gamma_rate  = float(np.clip(H * recent_vol * 50, 0.001, 0.5))

        # κ = re-coupling strength: inverse of spectral gap (slow decay = imminent coupling)
        # High volume acceleration also raises κ
        vol_accel = 0.0
        if len(volumes) >= 6:
            v_r = float(np.mean(volumes[-3:])) / (float(np.mean(volumes[-6:-3])) + 1e-9)
            vol_accel = float(np.clip(v_r - 1.0, 0.0, 2.0)) * 0.01
        kappa = float(np.clip(0.005 + vol_accel + (0.1 / (delta + 0.1)), 0.001, 0.5))
        self._kappa[symbol] = kappa

        # ρ_env dynamics: drho/dt = Γ·ψ_transfer - κ·ρ_env
        rho_prev     = self._rho_env[symbol]
        transfer_in  = gamma_rate * (N_prev - psi_sq) * 0.1
        rho_new      = float(np.clip(
            rho_prev + transfer_in - kappa * rho_prev, 0.0, N_prev
        ))
        self._rho_env[symbol] = rho_new

        # N_tot conservation check
        N_tot = psi_sq + rho_new
        if abs(N_tot - N_prev) > self.config.spectral.N_tot_conservation_tolerance:
            # Soft correction toward conservation
            correction = 0.05 * (N_prev - N_tot)
            N_tot      = N_tot + correction
        self._N_tot[symbol] = float(np.clip(N_tot, 0.01, 2.0))

        return psi_sq, rho_new, self._N_tot[symbol], kappa, gamma_rate

    @staticmethod
    def _detect_phase(psi_sq: float, rho_env: float, N_tot: float,
                      delta: float, revival_threshold: bool):
        """
        Phase I:   psi_sq high, rho_env low, delta large
        Phase II:  psi_sq declining, rho_env building, delta shrinking
        Phase III: revival_threshold True, rho_env significant
        """
        if N_tot < 1e-6:
            return Phase.ONE, 0.5

        psi_frac = psi_sq / N_tot
        rho_frac = rho_env / N_tot

        if revival_threshold and rho_frac > 0.30:
            confidence = float(np.clip(rho_frac * 2.0, 0.5, 1.0))
            return Phase.THREE, confidence

        if psi_frac < 0.65 and delta < 0.20 and rho_frac > 0.10:
            confidence = float(np.clip((0.65 - psi_frac) * 3.0, 0.3, 0.9))
            return Phase.TWO, confidence

        confidence = float(np.clip(psi_frac, 0.4, 0.95))
        return Phase.ONE, confidence

    @staticmethod
    def _infer_direction(returns: np.ndarray,
                         delta: float,
                         grad_delta: float) -> Direction:
        """
        Infer revival direction from:
        1. Net momentum of recent returns (dominant directional bias)
        2. Sign of ∇Δ (strengthening or weakening trend)
        3. Weighted combination
        """
        if len(returns) < 5:
            return Direction.FLAT

        # Short momentum
        short_mom = float(np.sum(returns[-5:]))
        # Medium momentum
        mid_mom   = float(np.sum(returns[-15:])) if len(returns) >= 15 else short_mom

        combined = 0.6 * short_mom + 0.4 * mid_mom

        # ∇Δ adjustment: negative grad_delta means bifurcation building,
        # amplify the existing momentum direction
        if abs(grad_delta) > 0.02:
            combined *= (1.0 + abs(grad_delta) * 2.0)

        threshold = 1e-5
        if combined > threshold:
            return Direction.BUY
        if combined < -threshold:
            return Direction.SELL
        return Direction.FLAT

    def _empty_state(self, symbol: str) -> ManifoldState:
        R = self.ra.get_R(symbol)
        return ManifoldState(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            R=R,
            R_char=self.ra.get_R_char(symbol),
        )
