"""
IMM AGI Hub — Data Models
All dataclasses and enums used across modules.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class Phase(int, Enum):
    ONE   = 1   # Coherent propagation
    TWO   = 2   # Projection / Transfer (compression)
    THREE = 3   # Passive Revival (fire zone)


class Direction(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"


class ExitReason(str, Enum):
    HOLD              = "HOLD"
    TRAIL_STOP        = "TRAIL_STOP"
    CLOSE_PARTIAL     = "CLOSE_PARTIAL"
    CLOSE_ALL         = "CLOSE_ALL"
    MOVE_TO_BE_PLUS   = "MOVE_TO_BE_PLUS"


# ─────────────────────────────────────────────
# Manifold State
# ─────────────────────────────────────────────

@dataclass
class ManifoldState:
    symbol: str
    timestamp: datetime

    # Phase classification
    phase: Phase = Phase.ONE
    phase_confidence: float = 0.5

    # Spectral gap
    delta: float = 0.5
    grad_delta: float = 0.0

    # Three-phase coherence cycle
    psi_sq: float = 1.0       # ||ψ||²  visible coherence energy
    rho_env: float = 0.0      # ρ_env   background accumulated load
    N_tot: float = 1.0        # conservation: psi_sq + rho_env
    kappa: float = 0.01       # re-coupling strength
    gamma_rate: float = 0.02  # ongoing projection pressure
    revival_threshold: bool = False  # κ·ρ_env > γ

    # Entropy field
    H_market: float = 0.5    # Shannon entropy of return dist [0,1]
    P_time: float = 0.5      # Time pressure metric [0,1]

    # Receiver state (adapted per instrument)
    R: np.ndarray = field(default_factory=lambda: np.array([55.0, 60.0, 58.0, 65.0, 55.0]))
    R_char: np.ndarray = field(default_factory=lambda: np.array([55.0, 60.0, 58.0, 65.0, 55.0]))
    V: float = 0.0            # Stability potential
    S: float = 0.0            # Stability score = -V

    # Price data
    current_price: float = 0.0
    atr: float = 0.001
    returns: np.ndarray = field(default_factory=lambda: np.zeros(50))

    # Direction from spectral gradient
    gradient_direction: Direction = Direction.FLAT


# ─────────────────────────────────────────────
# Trade Signal
# ─────────────────────────────────────────────

@dataclass
class TradeSignal:
    symbol: str
    direction: Direction
    confidence: float
    revival_strength: float       # κ·ρ_env / γ
    delay_bars: int
    receiver_stability: float     # S(R,E) at signal time
    phase2_duration_bars: int
    rho_env_accumulated: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────────
# Open Trade
# ─────────────────────────────────────────────

@dataclass
class Trade:
    trade_id: str
    symbol: str
    direction: Direction
    lots: float
    entry_price: float
    entry_time: datetime
    ea_id: str
    current_stop: float = 0.0
    current_price: float = 0.0
    pnl_r: float = 0.0           # P&L in R multiples
    pnl_usd: float = 0.0
    initial_risk_usd: float = 0.0
    open: bool = True


# ─────────────────────────────────────────────
# Exit Action
# ─────────────────────────────────────────────

@dataclass
class ExitAction:
    reason: ExitReason
    new_stop: Optional[float] = None
    close_fraction: Optional[float] = None
    close_reason: Optional[str] = None

    @staticmethod
    def hold() -> ExitAction:
        return ExitAction(reason=ExitReason.HOLD)

    @staticmethod
    def trail(new_stop: float) -> ExitAction:
        return ExitAction(reason=ExitReason.TRAIL_STOP, new_stop=new_stop)

    @staticmethod
    def close_partial(fraction: float) -> ExitAction:
        return ExitAction(reason=ExitReason.CLOSE_PARTIAL, close_fraction=fraction)

    @staticmethod
    def close_all(reason: str) -> ExitAction:
        return ExitAction(reason=ExitReason.CLOSE_ALL, close_reason=reason)

    @staticmethod
    def move_to_be_plus() -> ExitAction:
        return ExitAction(reason=ExitReason.MOVE_TO_BE_PLUS)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class ReceiverConfig:
    forex_majors: list = field(default_factory=lambda: [55.0, 60.0, 58.0, 65.0, 55.0])
    gold: list         = field(default_factory=lambda: [65.0, 70.0, 55.0, 60.0, 50.0])
    indices: list      = field(default_factory=lambda: [50.0, 65.0, 55.0, 62.0, 58.0])


@dataclass
class SpectralConfig:
    gap_estimation_lags: int = 25
    phase2_min_bars: int = 5
    phase3_revival_threshold: float = 1.15
    N_tot_conservation_tolerance: float = 0.08


@dataclass
class RiskConfig:
    max_risk_per_trade_pct: float = 0.005
    max_concurrent_trades: int = 4
    max_correlated_exposure: int = 2
    daily_stop_at_dd_pct: float = 0.85


@dataclass
class Config:
    account_size: float = 10_000.0
    daily_dd_pct: float = 0.05
    max_dd_pct: float   = 0.10
    broker: str         = "aquafunded"

    macro_anchors: list  = field(default_factory=lambda: ["DX-Y.NYB", "^TNX", "^GSPC"])
    tradeable: list      = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
        "XAUUSD", "USDCAD", "GBPJPY", "EURJPY"
    ])

    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    risk: RiskConfig         = field(default_factory=RiskConfig)

    ngrok_port: int = 8000
    loop_interval: float = 2.0   # seconds between main loop ticks
    data_lookback: int = 100      # bars of history for calculations

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        import yaml
        with open(path, encoding='utf-8', errors='replace') as f:
            d = yaml.safe_load(f)
        cfg = cls()
        acc = d.get("account", {})
        cfg.account_size = float(acc.get("size", 10000))
        cfg.daily_dd_pct = float(acc.get("daily_dd_pct", 0.05))
        cfg.max_dd_pct   = float(acc.get("max_dd_pct", 0.10))
        cfg.broker       = acc.get("broker", "aquafunded")

        inst = d.get("instruments", {})
        cfg.macro_anchors = inst.get("macro_anchors", cfg.macro_anchors)
        cfg.tradeable     = inst.get("tradeable", cfg.tradeable)

        rec = d.get("receiver", {})
        cfg.receiver.forex_majors = rec.get("forex_majors", cfg.receiver.forex_majors)
        cfg.receiver.gold         = rec.get("gold", cfg.receiver.gold)
        cfg.receiver.indices      = rec.get("indices", cfg.receiver.indices)

        sp = d.get("spectral", {})
        cfg.spectral.gap_estimation_lags      = int(sp.get("gap_estimation_lags", 25))
        cfg.spectral.phase2_min_bars          = int(sp.get("phase2_min_bars", 5))
        cfg.spectral.phase3_revival_threshold = float(sp.get("phase3_revival_threshold", 1.15))

        rk = d.get("risk", {})
        cfg.risk.max_risk_per_trade_pct  = float(rk.get("max_risk_per_trade_pct", 0.005))
        cfg.risk.max_concurrent_trades   = int(rk.get("max_concurrent_trades", 4))
        cfg.risk.max_correlated_exposure = int(rk.get("max_correlated_exposure", 2))
        cfg.risk.daily_stop_at_dd_pct    = float(rk.get("daily_stop_at_dd_pct", 0.85))

        ng = d.get("ngrok", {})
        cfg.ngrok_port = int(ng.get("port", 8000))
        return cfg
