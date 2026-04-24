"""
IMM AGI Hub — Risk Gate
AquaFunded 2-Step Pro $10k constraints embedded as IMM stability thresholds.
Scales cleanly to any account size via config.
"""
from __future__ import annotations
import logging
from datetime import date, datetime, timezone
from typing import Dict
import numpy as np

from models import Config, Trade, Direction

log = logging.getLogger("risk_gate")

PIP_VALUES: Dict[str, float] = {
    "EURUSD": 10.0, "GBPUSD": 10.0, "AUDUSD": 10.0, "NZDUSD": 10.0,
    "USDJPY":  9.1, "USDCAD":  7.5, "USDCHF": 11.0,
    "GBPJPY":  9.1, "EURJPY":  9.1, "AUDJPY":  9.1, "CADJPY": 9.1,
    "XAUUSD": 10.0, "XAGUSD": 50.0,
    "DEFAULT": 10.0,
}

CONTRACT_SIZES: Dict[str, float] = {
    "XAUUSD": 100.0,
    "DEFAULT": 100_000.0,
}


class RiskGate:
    """
    Structural drawdown management grounded in IMM stability framework.

    Key principle: trades are blocked not by dollar limits directly, but by
    raising the required stability threshold θ_trade as DD pressure grows.
    This creates a soft shutoff well before hard limits are breached.
    """

    THETA_BASE = -15.0   # minimum stability score when no DD used
    KAPPA_DD   =  3.0    # how fast θ rises with DD pressure
    HARD_BLOCK = 0.85    # DD fraction at which trading is fully blocked

    def __init__(self, config: Config):
        self.account_size    = config.account_size
        self.daily_dd_limit  = config.daily_dd_pct
        self.max_dd_limit    = config.max_dd_pct
        self.risk_cfg        = config.risk

        self.daily_dd_hard   = config.account_size * config.daily_dd_pct
        self.max_dd_hard     = config.account_size * config.max_dd_pct

        # Drawdown tracking
        self.daily_dd_used:  float = 0.0
        self.max_dd_used:    float = 0.0
        self.session_locked: bool  = False
        self.session_date:   date  = datetime.now(timezone.utc).date()

        # Open trade tracking for correlated exposure
        self._open_trades: Dict[str, Trade] = {}

    # ── Daily reset ───────────────────────────────────────────────────

    def check_new_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.session_date:
            log.info(f"New trading day {today} — resetting daily DD tracker.")
            self.daily_dd_used  = 0.0
            self.session_locked = False
            self.session_date   = today

    # ── Threshold computation ─────────────────────────────────────────

    def theta_trade(self) -> float:
        """
        Minimum receiver stability score required to open a trade.

        θ_trade = θ_base * (1 + κ_dd * pressure)

        pressure → 0 : θ_trade = θ_base  (easy to trade)
        pressure → 1 : θ_trade → +∞      (blocked)
        """
        if self.session_locked:
            return float("inf")

        daily_pressure = self.daily_dd_used / max(self.daily_dd_hard, 1.0)
        max_pressure   = self.max_dd_used   / max(self.max_dd_hard,   1.0)
        combined       = max(daily_pressure, max_pressure)

        if combined >= self.HARD_BLOCK:
            log.warning(
                f"DD pressure {combined:.1%} ≥ {self.HARD_BLOCK:.1%} — "
                f"trading BLOCKED"
            )
            return float("inf")

        theta = self.THETA_BASE * (1.0 + self.KAPPA_DD * combined)
        return theta

    def theta_probe(self) -> float:
        """Lower threshold for Spectral Rider probe entries (50% of full bar)."""
        t = self.theta_trade()
        if t == float("inf"):
            return float("inf")
        return t * 0.5

    # ── Position sizing ───────────────────────────────────────────────

    def position_size(self, symbol: str, stability: float,
                      revival_strength: float, atr: float) -> float:
        """
        IMM-grounded position sizing.

        risk_amount = min(0.5% account, 20% remaining daily budget)
                      × stability_factor
                      × signal_factor

        lots = risk_amount / (atr × pip_value_per_lot)
        """
        remaining = max(self.daily_dd_hard - self.daily_dd_used, 0.0)
        max_risk  = min(
            self.account_size * self.risk_cfg.max_risk_per_trade_pct,
            remaining * 0.20
        )
        if max_risk <= 0:
            return 0.0

        stability_factor = float(np.clip((stability + 20.0) / 30.0, 0.1, 1.0))
        signal_factor    = float(np.clip(revival_strength / 2.0,     0.3, 1.0))

        risk_amount = max_risk * stability_factor * signal_factor

        pip_val  = PIP_VALUES.get(symbol.upper(), PIP_VALUES["DEFAULT"])
        contract = CONTRACT_SIZES.get(symbol.upper(), CONTRACT_SIZES["DEFAULT"])

        # risk = lots × atr_in_pips × pip_value_per_lot
        atr_pips = atr * 10_000.0 if "JPY" not in symbol.upper() else atr * 100.0
        if "XAU" in symbol.upper() or "XAG" in symbol.upper():
            atr_pips = atr * 10.0

        sl_pips = max(atr_pips * 1.5, 5.0)
        lots = risk_amount / max(sl_pips * pip_val * 0.01, 0.0001)

        # Enforce min/max lot bounds
        lots = float(np.clip(lots, 0.01, 10.0))

        log.debug(
            f"[{symbol}] sizing: risk=${risk_amount:.2f}  "
            f"sl={sl_pips:.1f}pip  lots={lots:.3f}"
        )
        return round(lots, 2)

    def sl_pips(self, symbol: str, atr: float) -> int:
        """Initial stop-loss in pips = 1.5×ATR."""
        atr_pips = atr * 10_000.0
        if "JPY" in symbol.upper():
            atr_pips = atr * 100.0
        if "XAU" in symbol.upper() or "XAG" in symbol.upper():
            atr_pips = atr * 10.0
        return max(int(atr_pips * 1.5), 8)

    # ── Trade registration / DD tracking ─────────────────────────────

    def register_trade(self, trade: Trade):
        self._open_trades[trade.trade_id] = trade

    def update_trade_pnl(self, trade_id: str, pnl_usd: float):
        if trade_id in self._open_trades:
            self._open_trades[trade_id].pnl_usd = pnl_usd
        self._recompute_dd()

    def close_trade(self, trade_id: str, realised_pnl: float):
        if trade_id in self._open_trades:
            del self._open_trades[trade_id]
        # Negative realised PnL increases used DD
        if realised_pnl < 0:
            self.daily_dd_used += abs(realised_pnl)
            self.max_dd_used   += abs(realised_pnl)
            log.info(
                f"Trade closed P&L=${realised_pnl:.2f}  "
                f"daily_DD=${self.daily_dd_used:.2f}/{self.daily_dd_hard:.2f}  "
                f"max_DD=${self.max_dd_used:.2f}/{self.max_dd_hard:.2f}"
            )
        self._check_limits()

    def _recompute_dd(self):
        """Include floating losses in DD calculation."""
        floating = sum(
            min(t.pnl_usd, 0) for t in self._open_trades.values()
        )
        # We track max_dd_used including floating losses
        effective_max = self.max_dd_used + abs(min(floating, 0))
        if effective_max > self.max_dd_hard * 0.92:
            log.warning(
                f"Floating+realised DD ${effective_max:.2f} approaching "
                f"max hard limit ${self.max_dd_hard:.2f}"
            )

    def _check_limits(self):
        if self.daily_dd_used >= self.daily_dd_hard * self.HARD_BLOCK:
            log.critical(
                f"Daily DD ${self.daily_dd_used:.2f} at {self.HARD_BLOCK:.0%} "
                f"of limit — locking session."
            )
            self.session_locked = True

    # ── Correlated exposure check ─────────────────────────────────────

    def correlated_exposure_ok(self, symbol: str, direction: Direction) -> bool:
        """
        Prevent over-exposure in correlated instruments.
        e.g. BUY EURUSD + BUY GBPUSD = 2 USD-short positions.
        """
        group = self._correlation_group(symbol)
        same_dir_count = sum(
            1 for t in self._open_trades.values()
            if self._correlation_group(t.symbol) == group
            and t.direction == direction
            and t.open
        )
        limit = self.risk_cfg.max_correlated_exposure
        if same_dir_count >= limit:
            log.debug(
                f"[{symbol}] Correlated exposure limit {limit} reached "
                f"in group '{group}'"
            )
            return False
        return True

    @staticmethod
    def _correlation_group(symbol: str) -> str:
        sym = symbol.upper()
        if any(x in sym for x in ["EUR", "GBP", "AUD", "NZD"]):
            return "USD_SHORT"
        if any(x in sym for x in ["JPY", "CHF"]):
            return "SAFE_HAVEN"
        if "XAU" in sym or "XAG" in sym:
            return "METALS"
        if "USD" in sym:
            return "USD_LONG"
        return "OTHER"

    # ── Status ────────────────────────────────────────────────────────

    def status_dict(self) -> dict:
        return {
            "account_size":   self.account_size,
            "daily_dd_used":  round(self.daily_dd_used, 2),
            "daily_dd_hard":  self.daily_dd_hard,
            "max_dd_used":    round(self.max_dd_used, 2),
            "max_dd_hard":    self.max_dd_hard,
            "session_locked": self.session_locked,
            "open_trades":    len(self._open_trades),
            "theta_trade":    round(self.theta_trade(), 2),
        }
