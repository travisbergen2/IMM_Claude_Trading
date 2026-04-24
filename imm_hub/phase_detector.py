"""
IMM AGI Hub — Phase III Detector
Core signal engine. Tracks coherence cycle per instrument.
Emits TradeSignal when Phase III revival threshold is confirmed.
"""
from __future__ import annotations
import logging
from collections import deque
from datetime import datetime
from typing import Optional, Deque
import numpy as np

from models import Config, ManifoldState, TradeSignal, Phase, Direction

log = logging.getLogger("phase_detector")

DT = 1.0   # nominal time step between updates (in "bars")


class PhaseIIIDetector:
    """
    Tracks the Three-Phase Coherence Cycle for a single instrument.

    State machine:
        WATCHING → PHASE_II_ACCUMULATING → PHASE_III_TRIGGER

    Fires TradeSignal only when all of:
      • Phase == THREE
      • revival_threshold True
      • Phase II ran for at least phase2_min_bars
      • Receiver stability S > θ_trade
      • Direction is not FLAT
    """

    def __init__(self, symbol: str, config: Config):
        self.symbol       = symbol
        self.cfg_spectral = config.spectral

        self.phase_history: Deque[Phase]  = deque(maxlen=200)
        self.delta_history: Deque[float]  = deque(maxlen=200)
        self.revival_history: Deque[bool] = deque(maxlen=50)

        # Phase II tracking
        self.in_phase2: bool = False
        self.phase2_start: Optional[datetime] = None
        self.phase2_bars: int = 0
        self.N_tot_at_p2_entry: float = 1.0
        self.rho_env_integral: float = 0.0

        # Cooldown after firing (bars before next signal allowed)
        self.bars_since_signal: int = 999
        self.COOLDOWN_BARS: int = 10

        # Signal deduplication: track last direction fired
        self.last_signal_direction: Optional[Direction] = None
        self.last_signal_time: Optional[datetime] = None

    # ── Public ───────────────────────────────────────────────────────

    def update(self, state: ManifoldState,
               theta_trade: float) -> Optional[TradeSignal]:
        """
        Called each hub cycle. Returns TradeSignal if Phase III confirmed,
        else None.
        """
        self.bars_since_signal += 1
        self.phase_history.append(state.phase)
        self.delta_history.append(state.delta)
        self.revival_history.append(state.revival_threshold)

        self._update_phase2_tracker(state)

        if not self._fire_conditions_met(state, theta_trade):
            return None

        signal = self._build_signal(state)
        self._on_signal_fired(signal)
        return signal

    def reset(self):
        """Reset cycle tracker (e.g. after a trade is opened)."""
        self.in_phase2    = False
        self.phase2_start = None
        self.phase2_bars  = 0
        self.rho_env_integral = 0.0
        self.bars_since_signal = 0

    # ── Phase II accumulation ─────────────────────────────────────────

    def _update_phase2_tracker(self, state: ManifoldState):
        prev_phase = (self.phase_history[-2]
                      if len(self.phase_history) >= 2
                      else Phase.ONE)
        curr_phase = state.phase

        # Transition I → II: start accumulating
        if prev_phase == Phase.ONE and curr_phase == Phase.TWO:
            self.in_phase2          = True
            self.phase2_start       = state.timestamp
            self.phase2_bars        = 1
            self.N_tot_at_p2_entry  = state.N_tot
            self.rho_env_integral   = state.rho_env * DT
            log.debug(f"[{self.symbol}] Phase II entry — Δ={state.delta:.4f}")

        elif curr_phase == Phase.TWO and self.in_phase2:
            self.phase2_bars      += 1
            self.rho_env_integral += state.rho_env * DT

        elif curr_phase == Phase.ONE and self.in_phase2:
            # Exited Phase II without Phase III — reset
            self.in_phase2  = False
            self.phase2_bars = 0
            self.rho_env_integral = 0.0

    # ── Fire conditions ───────────────────────────────────────────────

    def _fire_conditions_met(self, state: ManifoldState,
                              theta_trade: float) -> bool:
        if state.phase != Phase.THREE:
            return False
        if not state.revival_threshold:
            return False
        if self.phase2_bars < self.cfg_spectral.phase2_min_bars:
            return False
        if state.S < theta_trade:
            log.debug(f"[{self.symbol}] BLOCKED S={state.S:.2f} < θ={theta_trade:.2f}")
            return False
        if state.gradient_direction == Direction.FLAT:
            return False
        if self.bars_since_signal < self.COOLDOWN_BARS:
            return False

        # Require κ·ρ_env to exceed Γ by the configured margin
        revival_ratio = (state.kappa * state.rho_env) / max(state.gamma_rate, 1e-9)
        if revival_ratio < self.cfg_spectral.phase3_revival_threshold:
            return False

        return True

    # ── Signal construction ───────────────────────────────────────────

    def _build_signal(self, state: ManifoldState) -> TradeSignal:
        revival_strength = (state.kappa * state.rho_env) / max(state.gamma_rate, 1e-9)
        delay_estimate   = max(1, int(1.0 / max(state.kappa, 0.001)))

        phase2_bars = self.phase2_bars if self.in_phase2 else 0

        log.info(
            f"[{self.symbol}] *** PHASE III SIGNAL ***  "
            f"dir={state.gradient_direction.value}  "
            f"revival={revival_strength:.3f}  "
            f"S={state.S:.2f}  "
            f"p2_bars={phase2_bars}  "
            f"ρ_acc={self.rho_env_integral:.4f}"
        )

        return TradeSignal(
            symbol=self.symbol,
            direction=state.gradient_direction,
            confidence=state.phase_confidence,
            revival_strength=revival_strength,
            delay_bars=delay_estimate,
            receiver_stability=state.S,
            phase2_duration_bars=phase2_bars,
            rho_env_accumulated=self.rho_env_integral,
        )

    def _on_signal_fired(self, signal: TradeSignal):
        self.bars_since_signal     = 0
        self.last_signal_direction = signal.direction
        self.last_signal_time      = signal.timestamp
        # Don't reset in_phase2 yet — exit engine still needs the context


class SpectralRiderDetector:
    """
    Phase II probe detector for the SPECTRAL RIDER EA.
    Lower bar than PhaseIIIDetector — fires early into Phase II
    with a reduced probe position when conditions warrant.
    """

    def __init__(self, symbol: str, config: Config):
        self.symbol      = symbol
        self.cfg         = config.spectral
        self.phase_hist: Deque[Phase] = deque(maxlen=50)
        self.p2_bars: int = 0
        self.PROBE_MIN_P2_BARS: int = 3
        self.PROBE_COOLDOWN:    int = 15
        self.bars_since_probe:  int = 999
        self.PROBE_REVIVAL_THRESHOLD: float = 0.8  # lower than Phase III

    def update(self, state: ManifoldState,
               theta_probe: float) -> Optional[TradeSignal]:
        """
        Returns a probe TradeSignal if Phase II conditions are deep enough
        to warrant a pre-position.
        """
        self.bars_since_probe += 1
        self.phase_hist.append(state.phase)

        if state.phase == Phase.TWO:
            self.p2_bars += 1
        elif state.phase == Phase.ONE:
            self.p2_bars = 0

        if not self._probe_conditions_met(state, theta_probe):
            return None

        signal = TradeSignal(
            symbol=self.symbol,
            direction=state.gradient_direction,
            confidence=state.phase_confidence * 0.7,  # lower confidence for probe
            revival_strength=(state.kappa * state.rho_env) / max(state.gamma_rate, 1e-9),
            delay_bars=max(3, int(1.0 / max(state.kappa, 0.001))),
            receiver_stability=state.S,
            phase2_duration_bars=self.p2_bars,
            rho_env_accumulated=state.rho_env,
        )

        self.bars_since_probe = 0
        log.info(f"[{self.symbol}] SPECTRAL RIDER PROBE  dir={signal.direction.value}")
        return signal

    def _probe_conditions_met(self, state: ManifoldState,
                               theta_probe: float) -> bool:
        if state.phase != Phase.TWO:
            return False
        if self.p2_bars < self.PROBE_MIN_P2_BARS:
            return False
        if state.delta > 0.25:  # gap not small enough yet
            return False
        if state.gradient_direction == Direction.FLAT:
            return False
        if state.S < theta_probe:
            return False
        revival_ratio = (state.kappa * state.rho_env) / max(state.gamma_rate, 1e-9)
        if revival_ratio < self.PROBE_REVIVAL_THRESHOLD:
            return False
        if self.bars_since_probe < self.PROBE_COOLDOWN:
            return False
        return True
