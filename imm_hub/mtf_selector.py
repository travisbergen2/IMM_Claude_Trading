"""
IMM AGI Hub — Multi-Timeframe Selector
Encodes Travis's manual timeframe scanning process.

The process:
  1. Decide macro direction from 1h and 4h
  2. Find the timeframe where Phase II is deepest / most complete
  3. Select the timeframe with the most legible / profitable cycle
  4. Size based on how many timeframes agree

MTF Score formula:
  score(tf) = (ρ_env/N_tot) × (1/Δ) × SI_peak_bonus × phase2_maturity × ob_alignment

The timeframe with highest score is where you deploy.
When multiple TFs align in Phase III simultaneously = maximum size.

Timeframe hierarchy:
  1m  → τ_r ~ 3-8 min    κ = 0.35   cycle = 20 bars
  5m  → τ_r ~ 15-40 min  κ = 0.18   cycle = 25 bars
  15m → τ_r ~ 45-90 min  κ = 0.10   cycle = 30 bars
  1h  → τ_r ~ 2-6 hrs    κ = 0.05   cycle = 35 bars
  4h  → τ_r ~ 8-24 hrs   κ = 0.02   cycle = 40 bars

The 1m/5m are retail harvesting layers.
The 1h/4h are institutional direction layers.
Trade WITH 1h/4h direction, AFTER 1m/5m harvesting completes.
"""
from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Tuple
import numpy as np

log = logging.getLogger("mtf_selector")

# Timeframe definitions
TIMEFRAMES = {
    "1m":   {"κ_natural": 0.35, "cycle_bars": 20,  "weight": 0.10, "label": "1-min"},
    "5m":   {"κ_natural": 0.18, "cycle_bars": 25,  "weight": 0.20, "label": "5-min"},
    "15m":  {"κ_natural": 0.10, "cycle_bars": 30,  "weight": 0.25, "label": "15-min"},
    "1h":   {"κ_natural": 0.05, "cycle_bars": 35,  "weight": 0.30, "label": "1-hour"},
    "4h":   {"κ_natural": 0.02, "cycle_bars": 40,  "weight": 0.15, "label": "4-hour"},
}

# Minimum timeframe agreement for full-size entry
FULL_SIZE_MIN_AGREEMENT = 3   # at least 3 TFs in Phase III
HALF_SIZE_MIN_AGREEMENT = 2
PROBE_MIN_AGREEMENT     = 1


@dataclass
class TFState:
    """IMM state for one instrument at one timeframe."""
    tf:             str
    phase:          int       # 1, 2, 3
    confidence:     float
    psi_sq:         float
    rho_env:        float
    N_tot:          float
    delta:          float     # spectral gap
    kappa:          float
    gamma_rate:     float
    revival_threshold: bool
    H_market:       float
    direction:      float     # -1 to +1
    phase2_bars:    int       # how long Phase II has run
    mtf_score:      float     # composite predictability score


@dataclass
class MTFAnalysis:
    """Multi-timeframe analysis result for one instrument."""
    symbol:             str

    # Selected timeframe for trading
    best_tf:            str       # timeframe with highest MTF score
    best_tf_score:      float
    best_tf_direction:  float     # -1 to +1

    # Agreement count
    phase3_count:       int       # how many TFs are in Phase III
    phase2_count:       int
    direction_agreement: float    # fraction of TFs agreeing on direction

    # Direction consensus
    macro_direction:    float     # from 1h + 4h average
    micro_direction:    float     # from 1m + 5m average
    directions_aligned: bool      # macro and micro pointing same way

    # Size recommendation
    size_multiplier:    float     # 0.25 / 0.50 / 0.75 / 1.0
    size_reason:        str

    # Individual TF states
    tf_states:          Dict[str, TFState] = field(default_factory=dict)

    # Order book alignment
    ob_aligned:         bool  = False
    ob_direction:       float = 0.0

    # Harvest signal: 1m/5m currently harvesting retail
    harvest_in_progress: bool = False
    harvest_direction:   float = 0.0   # direction retail is being pushed

    # Overall signal
    fire_signal:        bool  = False
    fire_tf:            str   = ""
    fire_direction:     float = 0.0


class MTFSelector:
    """
    Multi-timeframe cycle stack analyzer.
    Scores each timeframe and selects the most profitable entry point.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._tf_states: Dict[str, TFState] = {}
        self._direction_history: Deque[Tuple[str, float]] = deque(maxlen=50)

    # ── Public ───────────────────────────────────────────────────────

    def update(self, tf_data: Dict[str, dict],
               ob_direction: float = 0.0,
               ob_phase: int = 1) -> MTFAnalysis:
        """
        tf_data: dict of {tf_name: {phase, psi_sq, rho_env, N_tot, delta,
                                    kappa, gamma_rate, H, direction, phase2_bars}}
        ob_direction: from OrderBookAnalyzer.get_direction()
        ob_phase: from OrderBookAnalyzer.get_phase_vote()
        """
        tf_states = {}

        for tf, data in tf_data.items():
            if tf not in TIMEFRAMES:
                continue
            tfp = TIMEFRAMES[tf]

            score = self._mtf_score(
                rho_env=data.get("rho_env", 0.5),
                N_tot=data.get("N_tot", 1.0),
                delta=data.get("delta", 0.3),
                phase2_bars=data.get("phase2_bars", 0),
                cycle_bars=tfp["cycle_bars"],
                phase=data.get("phase", 1),
                revival_threshold=data.get("revival_threshold", False),
                si_peak=data.get("si_peak", False),
            )

            state = TFState(
                tf=tf,
                phase=data.get("phase", 1),
                confidence=data.get("confidence", 0.5),
                psi_sq=data.get("psi_sq", 0.8),
                rho_env=data.get("rho_env", 0.1),
                N_tot=data.get("N_tot", 1.0),
                delta=data.get("delta", 0.3),
                kappa=data.get("kappa", 0.05),
                gamma_rate=data.get("gamma_rate", 0.05),
                revival_threshold=data.get("revival_threshold", False),
                H_market=data.get("H", 0.5),
                direction=data.get("direction", 0.0),
                phase2_bars=data.get("phase2_bars", 0),
                mtf_score=score,
            )
            tf_states[tf] = state

        self._tf_states = tf_states
        analysis = self._build_analysis(tf_states, ob_direction, ob_phase)
        return analysis

    # ── MTF Score ─────────────────────────────────────────────────────

    @staticmethod
    def _mtf_score(rho_env: float, N_tot: float, delta: float,
                   phase2_bars: int, cycle_bars: int,
                   phase: int, revival_threshold: bool,
                   si_peak: bool) -> float:
        """
        Composite predictability score for a single timeframe.

        Higher score = more legible cycle, more profitable entry.

        Components:
        1. ρ_env/N_tot: how much energy has transferred to background
           (higher = more compression = bigger revival potential)
        2. 1/Δ: proximity to bifurcation (smaller gap = more imminent)
        3. phase2_maturity: how complete the compression phase is
           (full maturity = all the harvesting done, ready for revival)
        4. revival_threshold: is κ·ρ_env > Γ already?
        5. si_peak: Cipolla stupid actors recently peaked (harvesting done)
        """
        if N_tot < 1e-6:
            return 0.0

        # 1. Compression depth
        compression = rho_env / N_tot   # 0 to 1

        # 2. Bifurcation proximity
        bifurcation = 1.0 / (delta + 0.05)   # larger when Δ small
        bifurcation = float(np.clip(bifurcation / 20.0, 0.0, 1.0))  # normalise

        # 3. Phase II maturity
        if cycle_bars > 0:
            maturity = float(np.clip(phase2_bars / (cycle_bars * 0.35), 0.0, 1.0))
        else:
            maturity = 0.5

        # 4. Revival threshold bonus
        revival_bonus = 0.3 if revival_threshold else 0.0

        # 5. Cipolla peak bonus
        si_bonus = 0.2 if si_peak else 0.0

        # 6. Phase penalty: only high scores in Phase II/III
        if phase == 1:
            phase_mult = 0.2
        elif phase == 2:
            phase_mult = 0.8
        else:
            phase_mult = 1.0

        raw_score = (
            compression * 0.35 +
            bifurcation * 0.25 +
            maturity    * 0.20 +
            revival_bonus * 0.30 +
            si_bonus      * 0.20
        ) * phase_mult

        return float(np.clip(raw_score, 0.0, 1.0))

    # ── Analysis builder ──────────────────────────────────────────────

    def _build_analysis(self, tf_states: Dict[str, TFState],
                         ob_direction: float, ob_phase: int) -> MTFAnalysis:

        if not tf_states:
            return self._empty_analysis()

        # Count phases
        p3_tfs = [tf for tf, s in tf_states.items() if s.phase == 3]
        p2_tfs = [tf for tf, s in tf_states.items() if s.phase == 2]

        # Best timeframe by score
        best_tf   = max(tf_states, key=lambda tf: tf_states[tf].mtf_score)
        best_state = tf_states[best_tf]

        # Macro direction: 1h + 4h weighted
        macro_dir = self._weighted_direction(tf_states, ["1h", "4h"])
        micro_dir = self._weighted_direction(tf_states, ["1m", "5m"])

        # Direction agreement across all TFs
        all_dirs    = [s.direction for s in tf_states.values() if abs(s.direction) > 0.05]
        if all_dirs:
            dominant_sign  = 1 if sum(1 for d in all_dirs if d > 0) > len(all_dirs) / 2 else -1
            dir_agreement  = sum(1 for d in all_dirs if np.sign(d) == dominant_sign) / len(all_dirs)
            consensus_dir  = dominant_sign * dir_agreement
        else:
            dir_agreement  = 0.0
            consensus_dir  = 0.0

        # Macro/micro alignment
        # The key insight: trade when 1h/4h (macro) and 1m/5m (micro) agree
        # AFTER micro shows the retail harvest is complete
        dirs_aligned = (abs(macro_dir) > 0.15 and
                        np.sign(macro_dir) == np.sign(micro_dir) and
                        abs(micro_dir) > 0.10)

        # Harvest detection
        # When 1m/5m are BOTH in Phase II = retail being compressed/liquidated.
        # Phase III on 1m/5m = harvest is COMPLETE = fire zone, not blocked.
        # Only block when BOTH micro TFs are in Phase II simultaneously.
        harvest_tfs = [tf for tf in ["1m", "5m"] if tf in tf_states and
                       tf_states[tf].phase == 2]
        harvest_in_progress = len(harvest_tfs) >= 2   # both must be compressing
        harvest_dir = micro_dir if harvest_in_progress else 0.0

        # Order book alignment
        ob_aligned = (ob_phase == 3 and
                      abs(ob_direction) > 0.15 and
                      np.sign(ob_direction) == np.sign(macro_dir))

        # Size multiplier
        size_mult, size_reason = self._compute_size(
            p3_count=len(p3_tfs),
            dirs_aligned=dirs_aligned,
            ob_aligned=ob_aligned,
            best_score=best_state.mtf_score,
            harvest_in_progress=harvest_in_progress,
        )

        # Fire signal
        fire = (
            len(p3_tfs) >= PROBE_MIN_AGREEMENT and
            abs(macro_dir) > 0.15 and
            size_mult >= 0.25 and
            not harvest_in_progress  # don't fire during the harvest itself
        )

        fire_dir = macro_dir if fire else 0.0
        fire_tf  = best_tf  if fire else ""

        return MTFAnalysis(
            symbol=self.symbol,
            best_tf=best_tf,
            best_tf_score=round(best_state.mtf_score, 4),
            best_tf_direction=round(best_state.direction, 4),
            phase3_count=len(p3_tfs),
            phase2_count=len(p2_tfs),
            direction_agreement=round(dir_agreement, 3),
            macro_direction=round(macro_dir, 4),
            micro_direction=round(micro_dir, 4),
            directions_aligned=dirs_aligned,
            size_multiplier=size_mult,
            size_reason=size_reason,
            tf_states=tf_states,
            ob_aligned=ob_aligned,
            ob_direction=round(ob_direction, 4),
            harvest_in_progress=harvest_in_progress,
            harvest_direction=round(harvest_dir, 4),
            fire_signal=fire,
            fire_tf=fire_tf,
            fire_direction=round(fire_dir, 4),
        )

    # ── Size logic ────────────────────────────────────────────────────

    @staticmethod
    def _compute_size(p3_count: int, dirs_aligned: bool,
                      ob_aligned: bool, best_score: float,
                      harvest_in_progress: bool) -> Tuple[float, str]:
        """
        Position size multiplier based on agreement and quality.

        Full size (1.0):  3+ TFs in Phase III + directions aligned + OB confirmed
        75%:              2+ TFs + directions aligned
        50%:              1-2 TFs Phase III + some agreement
        25% (probe):      Spectral Rider probe only
        0%:               Harvest in progress (wait)
        """
        if harvest_in_progress:
            return 0.0, "WAIT: harvest in progress on 1m/5m"

        if p3_count >= FULL_SIZE_MIN_AGREEMENT and dirs_aligned and ob_aligned:
            return 1.0, f"FULL: {p3_count} TFs P3 + macro/micro aligned + OB confirmed"

        if p3_count >= FULL_SIZE_MIN_AGREEMENT and dirs_aligned:
            return 0.75, f"75%: {p3_count} TFs P3 + aligned (OB not confirmed)"

        if p3_count >= HALF_SIZE_MIN_AGREEMENT and dirs_aligned:
            return 0.50, f"50%: {p3_count} TFs P3 + partial alignment"

        if p3_count >= PROBE_MIN_AGREEMENT and best_score > 0.40:
            return 0.25, f"PROBE: {p3_count} TF P3, score={best_score:.2f}"

        return 0.0, f"NO SIGNAL: p3={p3_count} score={best_score:.2f}"

    # ── Direction helpers ─────────────────────────────────────────────

    @staticmethod
    def _weighted_direction(tf_states: Dict[str, TFState],
                             tfs: List[str]) -> float:
        """Weighted average direction for a subset of timeframes."""
        total_w = 0.0
        total_d = 0.0
        for tf in tfs:
            if tf in tf_states:
                w = TIMEFRAMES[tf]["weight"]
                total_d += tf_states[tf].direction * w
                total_w += w
        return total_d / total_w if total_w > 0 else 0.0

    def _empty_analysis(self) -> MTFAnalysis:
        return MTFAnalysis(
            symbol=self.symbol,
            best_tf="", best_tf_score=0.0, best_tf_direction=0.0,
            phase3_count=0, phase2_count=0,
            direction_agreement=0.0,
            macro_direction=0.0, micro_direction=0.0,
            directions_aligned=False,
            size_multiplier=0.0, size_reason="no data",
        )

    def summary(self) -> str:
        if not self._tf_states:
            return f"[{self.symbol}] No MTF data"
        phases = {tf: s.phase for tf, s in self._tf_states.items()}
        scores = {tf: round(s.mtf_score, 2) for tf, s in self._tf_states.items()}
        return (
            f"[{self.symbol}] phases={phases}  "
            f"scores={scores}"
        )



    def _build_tf_data(self, bar_idx: int,
                                 seed: int = 0) -> Dict[str, dict]:
        """
        Build synthetic multi-timeframe state for dry-run testing.
        Encodes the layered harvesting structure.
        """
        rng   = np.random.default_rng(seed + bar_idx)
        result = {}

        for tf, tfp in TIMEFRAMES.items():
            cycle = tfp["cycle_bars"]

            # Each TF has its own cycle offset so they don't all fire together
            offset = hash(tf) % cycle
            pos    = (bar_idx + offset) % cycle

            if pos < cycle * 0.55:    # Phase I
                phase, psi, rho, delta = 1, 0.80, 0.10, 0.30
            elif pos < cycle * 0.78:  # Phase II
                frac = (pos - cycle*0.55) / (cycle * 0.23)
                phase, psi, rho, delta = 2, 0.80-0.40*frac, 0.10+0.45*frac, 0.40-0.38*frac
            else:                     # Phase III
                frac = (pos - cycle*0.78) / (cycle * 0.22)
                phase, psi, rho, delta = 3, 0.40+0.30*frac, 0.55-0.20*frac, 0.02

            N_tot = psi + rho
            gamma = 0.06 + rng.random() * 0.02
            kappa = tfp["κ_natural"] * (1.5 if phase == 3 else 1.0)
            revival = (kappa * rho) > gamma

            # Direction: macro TFs set direction, micro follows
            if tf in ["1h", "4h"]:
                direction = 0.40 if seed % 2 == 0 else -0.40
            else:
                direction = 0.30 if seed % 2 == 0 else -0.30
                if phase == 2:
                    direction *= -0.3   # retail being squeezed against macro

            result[tf] = {
                "phase": phase, "psi_sq": psi, "rho_env": rho,
                "N_tot": N_tot, "delta": delta, "kappa": kappa,
                "gamma_rate": gamma, "revival_threshold": revival,
                "H": 0.3 + (phase - 1) * 0.2,
                "direction": direction + rng.standard_normal() * 0.05,
                "phase2_bars": int((pos - cycle * 0.55) / 1) if phase == 2 else 0,
                "confidence": 0.6 + rng.random() * 0.3,
                "si_peak": (phase == 3 and frac < 0.3) if phase == 3 else False,
            }

        return result

class MTFRegistry:
    """Manages one MTFSelector per instrument."""

    def __init__(self, symbols: list):
        self._selectors: Dict[str, MTFSelector] = {
            sym: MTFSelector(sym) for sym in symbols
        }

    def update(self, symbol: str, tf_data: Dict[str, dict],
               ob_direction: float = 0.0,
               ob_phase: int = 1) -> MTFAnalysis:
        if symbol not in self._selectors:
            self._selectors[symbol] = MTFSelector(symbol)
        return self._selectors[symbol].update(tf_data, ob_direction, ob_phase)

    def build_synthetic_tf_data(self, symbol: str, bar_idx: int,
                                 seed: int = 0) -> Dict[str, dict]:
        """Delegate to selector's synthetic builder."""
        if symbol not in self._selectors:
            self._selectors[symbol] = MTFSelector(symbol)
        return self._selectors[symbol]._build_tf_data(bar_idx, seed)
