"""
IMM AGI Hub — Projection-Induced Duality Layer
Formalising Travis Bergen's PID paper (2026) in the trading context.

Core result (Theorem 1):
    Non-invertible projection Π : M → E produces structural partition in E
    between components corresponding to unresolved degrees of freedom in M.

Trading translation:
    Different market actors implement distinct projection operators Π(i)_exp.
    Retail, advanced-retail, institutional, and whale actors each compress
    M differently. Their apparent "disagreement" about price direction is not
    noise — it is structured information about which fiber Π⁻¹(y) we are in.

    The fiber Π⁻¹(y) is large when many M-states map to the same observable y.
    This is EXACTLY Phase II: high ρ_env, multiple M-states consistent with
    the observed compression. The forward image F(y) = Π(φ_ε(Π⁻¹(y))) collapses
    toward a single basin when the revival threshold κ·ρ_env > Γ is met.
    That collapse IS Phase III. PID tells us WHY it is forced.

PID Signal Validator:
    Uses multi-observer projection disagreement as a SIGNAL QUALITY metric.
    When different actor classes (OB granularities, TFs, Cipolla populations)
    disagree maximally → we are inside a large fiber → high uncertainty.
    When they converge → the fiber is collapsing → high confidence Phase III.
"""
from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

log = logging.getLogger("pid_layer")


@dataclass
class PIDState:
    """
    Projection-Induced Duality state for one instrument.
    Measures the degree of observer convergence / divergence.
    """
    symbol:              str

    # Fiber size estimate: how many M-states are consistent with current obs
    # Large fiber → Phase II compression → uncertainty
    # Small fiber → Phase III collapse → high confidence
    fiber_size_estimate: float   # 0=collapsed, 1=maximally uncertain

    # Observer convergence: fraction of projection operators agreeing on direction
    # Π(ob_01), Π(ob_1), Π(5m), Π(1h), Π(4h), Π(cipolla_bi)
    observer_convergence: float  # 0=all disagree, 1=all agree

    # Projection entropy: H of the direction distribution across observers
    # Low H → consensus → fiber collapsing → trade
    # High H → disagreement → fiber large → wait
    projection_entropy:  float   # 0=consensus, 1=maximum disagreement

    # PID confidence: how certain we are that Phase III collapse is imminent
    pid_confidence:      float   # 0-1

    # Dominant projection direction when convergence > threshold
    consensus_direction: float   # -1 to +1

    # Fiber collapse signal: True when fiber is collapsing toward Phase III
    fiber_collapsing:    bool    = False

    # ΔS = information lost in projection (entropy increase from M to E)
    # Large ΔS = large fiber = much was discarded = Phase II
    # Small ΔS = small fiber = observers agree = Phase III imminent
    delta_S:             float   = 0.0


class PIDValidator:
    """
    Validates trade signals using the PID framework.

    Multiple projection operators (order book granularities, timeframes,
    Cipolla actor populations) are treated as different observer projections
    of the same underlying manifold M.

    Theorem 1 application:
    When these projections DIVERGE → large fiber → many M-states consistent
    with observation → Phase II uncertainty → wait.

    When these projections CONVERGE → small fiber → one dominant M-state
    consistent → Phase III collapse → fire.
    """

    CONVERGENCE_THRESHOLD = 0.55   # minimum agreement fraction to fire
    ENTROPY_BLOCK_THRESHOLD = 0.72  # block if projection entropy too high
    FIBER_COLLAPSE_THRESHOLD = 0.40 # fiber_size below this = collapsing

    def compute(self,
                # Order book direction signals (different Π resolutions)
                ob_dir_1_0: float,    # Π at 1.0 granularity
                ob_dir_0_1: float,    # Π at 0.1 granularity
                ob_dir_0_01: float,   # Π at 0.01 granularity (finest)
                # Multi-timeframe directions
                dir_1m:  float,
                dir_5m:  float,
                dir_1h:  float,
                dir_4h:  float,
                # Cipolla actor projections
                cipolla_bi: float,    # bandit index direction (+/-1 * BI)
                cipolla_ii: float,    # intelligent index direction
                # Symbol
                symbol: str = "UNKNOWN"
                ) -> PIDState:
        """
        Compute PID state from all available projection operators.

        Each input is a direction signal in [-1, +1].
        We measure their agreement as the fiber size proxy.
        """
        directions = np.array([
            ob_dir_1_0, ob_dir_0_1, ob_dir_0_01,
            dir_1m, dir_5m, dir_1h, dir_4h,
            cipolla_bi, cipolla_ii,
        ])

        # Remove near-zero (uninformative) projections
        active = directions[np.abs(directions) > 0.08]

        if len(active) < 2:
            return PIDState(
                symbol=symbol, fiber_size_estimate=0.8,
                observer_convergence=0.0, projection_entropy=0.8,
                pid_confidence=0.2, consensus_direction=0.0,
                fiber_collapsing=False, delta_S=0.8,
            )

        # ── Observer convergence ──────────────────────────────────────
        # What fraction agree on dominant sign?
        n_pos = np.sum(active > 0)
        n_neg = np.sum(active < 0)
        n_tot = len(active)
        dominant_n = max(n_pos, n_neg)
        convergence = dominant_n / n_tot

        dominant_sign = 1.0 if n_pos >= n_neg else -1.0
        consensus_dir = dominant_sign * convergence

        # ── Projection entropy ─────────────────────────────────────────
        # Treat direction distribution as a 2-class probability
        # H = -p·log(p) - (1-p)·log(1-p)
        p = dominant_n / n_tot
        if 0 < p < 1:
            proj_entropy = float(-p * np.log2(p) - (1-p) * np.log2(1-p))
        else:
            proj_entropy = 0.0  # perfect consensus

        # ── Fiber size estimate ────────────────────────────────────────
        # Large fiber = many M-states consistent = high entropy = Phase II
        # Small fiber = consensus = fiber collapsing = Phase III
        # fiber_size ∝ projection_entropy
        fiber_size = float(np.clip(proj_entropy, 0.0, 1.0))

        # ── ΔS (information loss in projection) ───────────────────────
        # ΔS = variance across observer projections
        # High variance = each Π(i) captures different aspect of M = large fiber
        delta_S = float(np.std(active)) if len(active) > 1 else 0.5

        # ── PID confidence ─────────────────────────────────────────────
        # High confidence when: convergence high AND entropy low AND fiber small
        pid_conf = float(np.clip(
            convergence * (1.0 - proj_entropy) * (1.0 - fiber_size * 0.5),
            0.0, 1.0
        ))

        fiber_collapsing = (fiber_size < self.FIBER_COLLAPSE_THRESHOLD and
                            convergence > self.CONVERGENCE_THRESHOLD)

        return PIDState(
            symbol=symbol,
            fiber_size_estimate=round(fiber_size, 4),
            observer_convergence=round(convergence, 4),
            projection_entropy=round(proj_entropy, 4),
            pid_confidence=round(pid_conf, 4),
            consensus_direction=round(consensus_dir, 4),
            fiber_collapsing=fiber_collapsing,
            delta_S=round(delta_S, 4),
        )

    def validate_signal(self, pid: PIDState, signal_direction: float) -> tuple:
        """
        Gate a trade signal through PID framework.

        Returns (allowed: bool, confidence_multiplier: float, reason: str)

        Theorem 1: if fiber is large (proj_entropy high), M-states are
        underdetermined → signal is degenerate → wait.
        If fiber is collapsing → Phase III forced → fire with confidence.
        """
        # Block: projection entropy too high (fiber too large, Phase II)
        if pid.projection_entropy > self.ENTROPY_BLOCK_THRESHOLD:
            return False, 0.0, f"PID BLOCK: entropy={pid.projection_entropy:.2f} > {self.ENTROPY_BLOCK_THRESHOLD}"

        # Block: observer convergence insufficient
        if pid.observer_convergence < self.CONVERGENCE_THRESHOLD:
            return False, 0.0, f"PID BLOCK: convergence={pid.observer_convergence:.2f} < {self.CONVERGENCE_THRESHOLD}"

        # Block: consensus direction disagrees with signal
        if (abs(pid.consensus_direction) > 0.20 and
                np.sign(pid.consensus_direction) != np.sign(signal_direction)):
            return False, 0.0, (f"PID BLOCK: consensus={pid.consensus_direction:+.2f} "
                                 f"vs signal={signal_direction:+.2f}")

        # Size boost when fiber is actively collapsing (Theorem 1 payoff)
        if pid.fiber_collapsing:
            mult = float(np.clip(1.0 + pid.pid_confidence * 0.5, 1.0, 1.5))
            return True, mult, f"PID BOOST: fiber collapsing, conf={pid.pid_confidence:.2f}"

        # Normal pass
        mult = float(np.clip(pid.pid_confidence + 0.5, 0.5, 1.2))
        return True, mult, f"PID PASS: conf={pid.pid_confidence:.2f}"

    def log_state(self, pid: PIDState):
        collapse = "🌀COLLAPSING" if pid.fiber_collapsing else ""
        log.info(
            f"[{pid.symbol}] PID: fiber={pid.fiber_size_estimate:.2f}  "
            f"conv={pid.observer_convergence:.2f}  "
            f"H={pid.projection_entropy:.2f}  "
            f"conf={pid.pid_confidence:.2f}  "
            f"ΔS={pid.delta_S:.3f}  {collapse}"
        )
