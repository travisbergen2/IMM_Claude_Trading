"""
IMM AGI Hub — Dry-Run Simulator
Validates the full pipeline with synthetic market data.
No live connections required.

Run:
    python simulator.py [--bars 300] [--instruments 3] [--seed 42]

Generates synthetic multi-instrument data with embedded Phase II/III
cycles, runs the complete hub pipeline, and prints a signal report.
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

logging.basicConfig(
    level=logging.WARNING,  # suppress noise; we print our own report
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from models import (
    Config, ManifoldState, TradeSignal, Trade, Direction, Phase
)
from receiver_array    import ReceiverArray
from manifold_observer import ManifoldObserver
from phase_detector    import PhaseIIIDetector, SpectralRiderDetector
from exit_engine       import ExitEngine
from risk_gate         import RiskGate
from macro_field       import MacroField
from data_feed         import DataFeedManager


# ── Synthetic market generator ────────────────────────────────────────────────

def generate_cycle_data(
    n_bars: int = 300,
    start_price: float = 1.1000,
    seed: int = 42
) -> np.ndarray:
    """
    Generate close prices with embedded Phase cycles:
    - Phase I (bars 0-60):    coherent uptrend
    - Phase II (bars 60-100): compression, energy transfers to ρ_env
    - Phase III (bar ~100):   revival burst
    - Phase I (100-180):      new trend
    - Phase II (180-220):     compression
    - Phase III (bar ~220):   second revival
    - etc.
    """
    rng    = np.random.default_rng(seed)
    prices = np.zeros(n_bars)
    p      = start_price

    cycle_len = 100
    for i in range(n_bars):
        phase_i = i % cycle_len

        if phase_i < 50:
            # Phase I: coherent drift
            vol  = 0.0004
            drift = 0.00008
        elif phase_i < 80:
            # Phase II: compression — small range, lower vol
            vol   = 0.0002
            drift = 0.0
        else:
            # Phase III: revival burst
            vol   = 0.0012
            drift = 0.00025 * (1 if (seed % 2 == 0) else -1)

        ret  = drift + vol * rng.standard_normal()
        p    = p * np.exp(ret)
        prices[i] = p

    return prices


# ── Pipeline runner ────────────────────────────────────────────────────────────

class SimulationRunner:

    def __init__(self, config: Config, n_bars: int = 300, seed: int = 42):
        self.config  = config
        self.n_bars  = n_bars
        self.seed    = seed

        self.receiver_array = ReceiverArray(config)
        self.risk_gate      = RiskGate(config)
        self.macro_field    = MacroField()

        self.p3_detectors: Dict[str, PhaseIIIDetector] = {
            sym: PhaseIIIDetector(sym, config)
            for sym in config.tradeable
        }

        self.signals:     List[dict] = []
        self.sim_trades:  List[dict] = []
        self.open_trades: Dict[str, tuple] = {}  # trade_id → (entry_bar, entry_price, direction)

    def run(self) -> dict:
        """
        Simulate n_bars of market data through the full pipeline.
        Returns a summary dict.
        """
        print(f"\n{'='*60}")
        print(f"  IMM AGI SIMULATOR  |  {self.n_bars} bars  |  seed={self.seed}")
        print(f"  Instruments: {len(self.config.tradeable)}")
        print(f"{'='*60}\n")

        # Generate price series per instrument
        price_data: Dict[str, np.ndarray] = {}
        for i, sym in enumerate(self.config.tradeable):
            start = 1.0 + i * 0.05
            price_data[sym] = generate_cycle_data(self.n_bars, start, self.seed + i)

        # Rolling window for each instrument
        WINDOW = 60
        windows: Dict[str, deque] = {
            sym: deque(maxlen=WINDOW) for sym in self.config.tradeable
        }

        bar_interval_min = 5
        t0 = datetime.utcnow() - timedelta(minutes=self.n_bars * bar_interval_min)

        for bar_idx in range(self.n_bars):
            bar_time = t0 + timedelta(minutes=bar_idx * bar_interval_min)

            states: Dict[str, ManifoldState] = {}

            for sym in self.config.tradeable:
                windows[sym].append(price_data[sym][bar_idx])
                closes = np.array(list(windows[sym]))

                if len(closes) < 15:
                    continue

                returns = np.diff(np.log(closes + 1e-12))
                highs  = closes * 1.0003
                lows   = closes * 0.9997
                atr    = ManifoldObserver._compute_atr(highs, lows, closes)

                # ── Phase injection from cycle position ───────────────
                # Directly model the three-phase cycle so the detector
                # can exercise its full state machine on synthetic data.
                # Each 100-bar cycle: P1 (0-49), P2 (50-79), P3 (80-99)
                cycle_pos = bar_idx % 100

                if cycle_pos < 50:
                    # Phase I: coherent, low entropy, large spectral gap
                    delta      = 0.35 + 0.10 * (cycle_pos / 50.0)
                    H          = 0.15
                    psi_sq     = 0.82
                    rho_env    = 0.08
                    kappa      = 0.015
                    gamma_rate = 0.05
                    P_time     = 0.20
                elif cycle_pos < 80:
                    # Phase II: compression, spectral gap shrinking, rho_env building
                    frac   = (cycle_pos - 50) / 30.0
                    delta  = 0.45 - 0.40 * frac   # 0.45 → 0.05
                    H      = 0.30 + 0.35 * frac    # entropy rising
                    psi_sq = 0.82 - 0.40 * frac    # coherence draining
                    rho_env= 0.08 + 0.45 * frac    # load accumulating
                    kappa  = 0.015 + 0.12 * frac
                    gamma_rate = 0.05 + 0.03 * frac
                    P_time = 0.40 + 0.30 * frac
                else:
                    # Phase III: revival threshold crossed
                    frac   = (cycle_pos - 80) / 20.0
                    delta  = 0.05
                    H      = 0.65
                    psi_sq = 0.42 + 0.30 * frac
                    rho_env= 0.53 - 0.20 * frac
                    kappa  = 0.35
                    gamma_rate = 0.08
                    P_time = 0.70

                N_tot = psi_sq + rho_env
                revival_threshold = (kappa * rho_env) > gamma_rate

                phase, conf = ManifoldObserver._detect_phase(
                    psi_sq, rho_env, N_tot, delta, revival_threshold
                )

                direction = ManifoldObserver._infer_direction(returns, delta, 0.0)

                R_new = self.receiver_array.step(sym, H, P_time)
                V     = self.receiver_array.V(R_new, H, P_time)
                S     = -V

                state = ManifoldState(
                    symbol=sym,
                    timestamp=bar_time,
                    phase=phase,
                    phase_confidence=conf,
                    delta=delta,
                    grad_delta=0.0,
                    psi_sq=psi_sq,
                    rho_env=rho_env,
                    N_tot=N_tot,
                    kappa=kappa,
                    gamma_rate=gamma_rate,
                    revival_threshold=revival_threshold,
                    H_market=H,
                    P_time=P_time,
                    R=R_new,
                    R_char=self.receiver_array.get_R_char(sym),
                    V=V,
                    S=S,
                    current_price=float(closes[-1]),
                    atr=atr,
                    returns=returns[-50:] if len(returns) >= 50 else returns,
                    gradient_direction=direction,
                )
                states[sym] = state

            # Update macro (use first 3 instruments as synthetic anchors)
            self.macro_field.update(states)

            # Run phase detectors
            theta = self.risk_gate.theta_trade()

            for sym, state in states.items():
                if len(self.open_trades) >= self.config.risk.max_concurrent_trades:
                    break

                signal = self.p3_detectors[sym].update(state, theta)
                if signal is None:
                    continue

                macro_ok, macro_score, _ = self.macro_field.check(signal, states)
                if not macro_ok:
                    continue

                trade_id   = str(uuid.uuid4())[:8]
                entry_price = state.current_price
                direction  = signal.direction

                self.signals.append({
                    "bar":            bar_idx,
                    "time":           bar_time.strftime("%H:%M"),
                    "symbol":         sym,
                    "direction":      direction.value,
                    "revival":        round(signal.revival_strength, 3),
                    "stability_S":    round(state.S, 2),
                    "delta":          round(state.delta, 4),
                    "H":              round(state.H_market, 3),
                    "p2_bars":        signal.phase2_duration_bars,
                    "macro_score":    round(macro_score, 3),
                    "entry_price":    round(entry_price, 5),
                    "trade_id":       trade_id,
                })

                self.open_trades[trade_id] = (bar_idx, entry_price, direction, sym, state.atr)
                self.p3_detectors[sym].reset()

            # Simulate simple exits: close after 10 bars or at stop
            to_close = []
            for tid, (entry_bar, ep, dire, sym, atr) in list(self.open_trades.items()):
                if bar_idx - entry_bar >= 10:
                    close_price = price_data[sym][bar_idx]
                    sign  = 1 if dire == Direction.BUY else -1
                    pnl_p = sign * (close_price - ep) * 10_000
                    pnl_u = pnl_p * 0.10
                    self.sim_trades.append({
                        "trade_id":  tid,
                        "symbol":    sym,
                        "direction": dire.value,
                        "bars_held": bar_idx - entry_bar,
                        "pnl_pips":  round(pnl_p, 1),
                        "pnl_usd":   round(pnl_u, 2),
                    })
                    if pnl_u < 0:
                        self.risk_gate.daily_dd_used += abs(pnl_u)
                    to_close.append(tid)
            for tid in to_close:
                self.open_trades.pop(tid, None)

        return self._report()

    def _report(self) -> dict:
        signals = self.signals
        trades  = self.sim_trades

        print(f"  {'─'*56}")
        print(f"  SIGNALS DETECTED: {len(signals)}")
        print(f"  {'─'*56}")
        for s in signals:
            arrow = "↑" if s["direction"] == "BUY" else "↓"
            print(
                f"  bar={s['bar']:03d}  {s['symbol']:<8}  {arrow}  "
                f"revival={s['revival']:.2f}  S={s['stability_S']:+.1f}  "
                f"Δ={s['delta']:.4f}  macro={s['macro_score']:.2f}  "
                f"p2={s['p2_bars']}b"
            )

        print(f"\n  SIMULATED TRADES (10-bar exit): {len(trades)}")
        print(f"  {'─'*56}")
        winners = [t for t in trades if t["pnl_usd"] > 0]
        losers  = [t for t in trades if t["pnl_usd"] <= 0]
        total   = sum(t["pnl_usd"] for t in trades)

        for t in trades:
            emoji = "✅" if t["pnl_usd"] > 0 else "❌"
            print(
                f"  {emoji}  {t['symbol']:<8}  {t['direction']:<4}  "
                f"{t['bars_held']:2d}b  "
                f"{t['pnl_pips']:+6.1f}pip  ${t['pnl_usd']:+.2f}"
            )

        if trades:
            wr = len(winners) / len(trades)
            print(f"\n  Win rate: {wr:.1%}  |  Total P&L: ${total:.2f}")
            if winners and losers:
                avg_w = np.mean([t["pnl_usd"] for t in winners])
                avg_l = np.mean([t["pnl_usd"] for t in losers])
                pf    = abs(sum(t["pnl_usd"] for t in winners)) / max(abs(sum(t["pnl_usd"] for t in losers)), 0.01)
                print(f"  Avg W: ${avg_w:.2f}  Avg L: ${avg_l:.2f}  PF: {pf:.2f}")

        print(f"\n  Phase III signals: {len([s for s in signals])} across {self.n_bars} bars")
        print(f"  Signal rate: {len(signals)/self.n_bars*100:.1f}% of bars")
        print(f"{'='*60}\n")

        return {
            "n_bars": self.n_bars,
            "signals": len(signals),
            "trades": len(trades),
            "win_rate": len(winners)/max(len(trades), 1),
            "total_pnl": total,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMM AGI Simulator")
    parser.add_argument("--bars",        default=300, type=int)
    parser.add_argument("--instruments", default=4,   type=int)
    parser.add_argument("--seed",        default=42,  type=int)
    args = parser.parse_args()

    cfg = Config()
    cfg.tradeable = cfg.tradeable[:args.instruments]

    runner = SimulationRunner(cfg, n_bars=args.bars, seed=args.seed)
    result = runner.run()
