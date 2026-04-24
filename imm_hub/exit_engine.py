"""
IMM AGI Hub — Exit Engine
Phase-cycle-aware dynamic exit management.
Tracks revival energy remaining (ψ²/N_tot) and tightens trail accordingly.
Detects nested Phase III cycles for extended runners.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional
import numpy as np

from models import ManifoldState, Trade, ExitAction, ExitReason, Direction

log = logging.getLogger("exit_engine")

DT = 1.0   # time step normalisation


class ExitEngine:
    """
    One ExitEngine instance per open trade.

    State tracked:
    - initial_psi_sq, initial_N_tot: baseline at entry
    - peak_psi_sq: maximum coherence seen (high-water mark)
    - last_psi_sq: previous cycle value (for velocity)
    - bars_in_trade
    - nested_cycle_watching: True if we're waiting for a nested Phase III

    Exit decision hierarchy (highest priority first):
    1. Hard stop -1R → CLOSE_ALL
    2. Stability degraded → CLOSE_ALL
    3. Revival threshold lost AND no nested signal building → CLOSE_PARTIAL 70%
    4. Coherence velocity < 0 AND revival_remaining < 0.55:
       a. rho_env building → MOVE_TO_BE_PLUS (wait for nested)
       b. otherwise → CLOSE_PARTIAL 50%
    5. Coherence active → TRAIL_STOP (tightening as energy depletes)
    6. Default → HOLD
    """

    # Thresholds
    HARD_STOP_R:           float = -1.0
    STABILITY_CLOSE:       float = -25.0
    REVIVAL_FADING_FRAC:   float = 0.55
    NESTED_RHO_MIN:        float = 0.20
    NESTED_WAIT_BARS:      int   = 8

    def __init__(self, trade: Trade, entry_state: ManifoldState):
        self.trade          = trade
        self.initial_psi_sq = entry_state.psi_sq
        self.initial_N_tot  = entry_state.N_tot
        self.peak_psi_sq    = entry_state.psi_sq
        self.last_psi_sq    = entry_state.psi_sq
        self.bars_in_trade  = 0

        # Nested cycle detection
        self.nested_cycle_watching:  bool = False
        self.nested_wait_bars_count: int  = 0
        self.be_plus_set:            bool = False

        log.info(
            f"[{trade.symbol}] ExitEngine initialised  "
            f"ψ₀²={self.initial_psi_sq:.4f}  N₀={self.initial_N_tot:.4f}"
        )

    # ── Main update ───────────────────────────────────────────────────

    def update(self, state: ManifoldState, current_pnl_r: float) -> ExitAction:
        """
        Called each hub cycle while trade is open.
        Returns an ExitAction; ExitReason.HOLD means do nothing.
        """
        self.bars_in_trade += 1
        self.peak_psi_sq    = max(self.peak_psi_sq, state.psi_sq)

        N_safe = max(state.N_tot, 1e-6)
        revival_remaining   = state.psi_sq / N_safe
        coherence_velocity  = (state.psi_sq - self.last_psi_sq) / DT
        self.last_psi_sq    = state.psi_sq

        # ── 1. Hard stop ──────────────────────────────────────────────
        if current_pnl_r <= self.HARD_STOP_R:
            log.warning(f"[{self.trade.symbol}] HARD STOP -1R hit")
            return ExitAction.close_all("hard_stop_1R")

        # ── 2. Stability degraded ─────────────────────────────────────
        if state.S < self.STABILITY_CLOSE:
            log.info(f"[{self.trade.symbol}] CLOSE: stability degraded S={state.S:.2f}")
            return ExitAction.close_all("stability_degraded")

        # ── 3. Revival threshold lost ─────────────────────────────────
        if not state.revival_threshold:
            if not self.nested_cycle_watching:
                log.info(f"[{self.trade.symbol}] Revival threshold lost → close 70%")
                return ExitAction.close_partial(0.70)

        # ── 4. Nested cycle management ────────────────────────────────
        if coherence_velocity < 0 and revival_remaining < self.REVIVAL_FADING_FRAC:
            rho_frac = state.rho_env / N_safe

            if rho_frac > self.NESTED_RHO_MIN:
                # rho_env building → potential nested Phase III
                if not self.nested_cycle_watching:
                    self.nested_cycle_watching  = True
                    self.nested_wait_bars_count = 0
                    log.info(f"[{self.trade.symbol}] Nested cycle watch — BE+")
                    if not self.be_plus_set:
                        self.be_plus_set = True
                        return ExitAction.move_to_be_plus()
                else:
                    self.nested_wait_bars_count += 1
                    # If nested revival fires (threshold back True) → let it run
                    if state.revival_threshold and rho_frac > 0.30:
                        self.nested_cycle_watching  = False
                        self.nested_wait_bars_count = 0
                        log.info(f"[{self.trade.symbol}] Nested Phase III confirmed — hold")
                        return ExitAction.hold()
                    # Timeout
                    if self.nested_wait_bars_count > self.NESTED_WAIT_BARS:
                        self.nested_cycle_watching = False
                        log.info(f"[{self.trade.symbol}] Nested timeout → close 50%")
                        return ExitAction.close_partial(0.50)
            else:
                # Fading with no nested buildup
                if revival_remaining < 0.30:
                    log.info(f"[{self.trade.symbol}] Revival exhausted → close 50%")
                    return ExitAction.close_partial(0.50)

        # ── 5. Phase III active → trail ───────────────────────────────
        if (coherence_velocity > 0
                and revival_remaining > self.REVIVAL_FADING_FRAC
                and state.revival_threshold):
            new_stop = self._compute_trail(state, revival_remaining)
            return ExitAction.trail(new_stop)

        # ── 6. Default: hold ──────────────────────────────────────────
        return ExitAction.hold()

    # ── Trail computation ─────────────────────────────────────────────

    def _compute_trail(self, state: ManifoldState,
                        revival_remaining: float) -> float:
        """
        ATR multiplier scales with revival energy remaining.
        Full energy → wide trail (let it run).
        Depleting energy → tight trail (lock profits).

        multiplier = 0.5 + 1.5 * revival_remaining
          revival=1.0 → 2.0×ATR
          revival=0.6 → 1.4×ATR
          revival=0.3 → 0.95×ATR
        """
        atr_multiplier = 0.5 + 1.5 * float(np.clip(revival_remaining, 0.0, 1.0))
        direction = self.trade.direction

        if direction == Direction.BUY:
            return state.current_price - atr_multiplier * state.atr
        else:
            return state.current_price + atr_multiplier * state.atr

    # ── Diagnostics ───────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "trade_id":         self.trade.trade_id,
            "symbol":           self.trade.symbol,
            "bars_in_trade":    self.bars_in_trade,
            "peak_psi_sq":      round(self.peak_psi_sq, 5),
            "last_psi_sq":      round(self.last_psi_sq, 5),
            "initial_N_tot":    round(self.initial_N_tot, 5),
            "nested_watching":  self.nested_cycle_watching,
            "be_set":           self.be_plus_set,
        }
