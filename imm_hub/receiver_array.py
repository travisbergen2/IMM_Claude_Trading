"""
IMM AGI Hub — Receiver Array
Maintains one adaptive receiver R per instrument.
Implements gradient flow dynamics with coupling matrix and homeostasis.
"""
from __future__ import annotations
import numpy as np
from typing import Dict
from models import Config, ManifoldState


class ReceiverArray:
    """
    Per-instrument receiver R = (TI, SG, FT, UE, AR) ∈ [0,100]^5

    Dynamics:
        dR/dt = -AR · ∇_R V(R,E)       ← environment pressure
              + W · R_e · 10            ← inter-primitive coupling
              + κ_h · (R_char - R)/100  ← homeostatic regulation

    Stability potential:
        V(R,E) = α·SG²·(1-FT̃)·H + β·TI²·P + γ·FT̃²/(UẼ+ε)
    """

    ALPHA   = 0.008
    BETA    = 0.005
    GAMMA   = 8.0
    EPS     = 0.01
    KAPPA_H = 0.40   # homeostatic strength

    # Coupling matrix W (5×5)
    # Rows/cols: TI, SG, FT, UE, AR
    W = np.array([
        [ 0.000,  0.052, -0.070,  0.017,  0.017],
        [ 0.052,  0.000, -0.087,  0.017,  0.000],
        [-0.035, -0.070,  0.000,  0.052,  0.017],
        [ 0.017,  0.017, -0.052,  0.000,  0.070],
        [ 0.017,  0.000,  0.017,  0.070,  0.000],
    ])

    def __init__(self, config: Config):
        self.config = config
        # Per-instrument state: R, R_char
        self._receivers: Dict[str, np.ndarray] = {}
        self._char_points: Dict[str, np.ndarray] = {}

        for sym in config.tradeable + config.macro_anchors:
            r0 = self._default_R(sym, config)
            self._receivers[sym]   = r0.copy()
            self._char_points[sym] = r0.copy()

    # ── Public interface ──────────────────────────────────────────────

    def get_R(self, symbol: str) -> np.ndarray:
        return self._receivers.get(symbol, np.array([55.0, 60.0, 58.0, 65.0, 55.0]))

    def get_R_char(self, symbol: str) -> np.ndarray:
        return self._char_points.get(symbol, self.get_R(symbol))

    def step(self, symbol: str, H: float, P: float,
             lr: float = 0.05) -> np.ndarray:
        """Advance receiver one gradient-flow step. Returns updated R."""
        R      = self._receivers[symbol]
        R_char = self._char_points[symbol]

        AR = R[4] / 100.0
        gV = self._grad_V(R, H, P)

        dR = (
            -AR * gV
            + lr * (self.W @ (R * 10.0 / 100.0))
            + lr * self.KAPPA_H * (R_char - R) / 100.0
        )

        R_new = np.clip(R + dR, 0.0, 100.0)
        self._receivers[symbol] = R_new
        return R_new

    def V(self, R: np.ndarray, H: float, P: float) -> float:
        """Stability potential V(R, E)."""
        TI, SG, FT, UE, _ = R
        FTn = FT / 100.0
        UEn = UE / 100.0
        return (
            self.ALPHA * SG**2 * (1.0 - FTn) * H
            + self.BETA  * TI**2 * P
            + self.GAMMA * FTn**2 / (UEn + self.EPS)
        )

    def stability(self, R: np.ndarray, H: float, P: float) -> float:
        """S(R,E) = -V(R,E). Higher is more stable."""
        return -self.V(R, H, P)

    def regime_label(self, R: np.ndarray) -> str:
        """Classify current receiver into named stability regime."""
        TI, SG, FT, UE, _ = R
        if TI > 70 and SG > 70:
            return "OSCILLATORY"
        if SG > 70 and FT < 40:
            return "OVERLOAD"
        if FT > 70 and UE < 40:
            return "RIGID"
        if TI > 80 and SG < 40:
            return "SLUGGISH"
        return "STABLE"

    # ── Private helpers ───────────────────────────────────────────────

    def _grad_V(self, R: np.ndarray, H: float, P: float,
                h: float = 1e-2) -> np.ndarray:
        """Central-difference gradient of V w.r.t. R."""
        grad = np.zeros(5)
        for i in range(5):
            Rf = R.copy(); Rb = R.copy()
            Rf[i] += h;   Rb[i] -= h
            grad[i] = (self.V(Rf, H, P) - self.V(Rb, H, P)) / (2.0 * h)
        return grad

    @staticmethod
    def _default_R(symbol: str, config: Config) -> np.ndarray:
        sym = symbol.upper()
        if "XAU" in sym or "GOLD" in sym:
            return np.array(config.receiver.gold, dtype=float)
        if "SPX" in sym or "SPY" in sym or "NAS" in sym or "GSPC" in sym:
            return np.array(config.receiver.indices, dtype=float)
        return np.array(config.receiver.forex_majors, dtype=float)
