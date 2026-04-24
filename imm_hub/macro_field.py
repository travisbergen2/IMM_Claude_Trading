"""
IMM AGI Hub — Macro Field Layer
Cross-instrument coherence checks.
DXY / TLT / SPX are entropy anchors that validate or block individual signals.
"""
from __future__ import annotations
import logging
from typing import Dict, Optional, Tuple
import numpy as np

from models import ManifoldState, TradeSignal, Direction, Phase

log = logging.getLogger("macro_field")


# ── Instrument → correlation group mapping ────────────────────────────────

def _usd_bias(symbol: str, direction: Direction) -> float:
    """
    +1  if trade implies USD weakness
    -1  if trade implies USD strength
     0  if neutral
    """
    sym = symbol.upper()
    if direction == Direction.BUY:
        if any(x in sym[:3] for x in ["EUR", "GBP", "AUD", "NZD", "CAD", "CHF"]):
            return +1.0   # buying EUR/GBP vs USD = USD weak
        if sym[:3] == "USD":
            return -1.0   # buying USD/xxx = USD strong
        if "XAU" in sym or "XAG" in sym:
            return +0.5   # metals = mild USD weakness
    else:  # SELL
        if any(x in sym[:3] for x in ["EUR", "GBP", "AUD", "NZD", "CAD", "CHF"]):
            return -1.0
        if sym[:3] == "USD":
            return +1.0
        if "XAU" in sym or "XAG" in sym:
            return -0.5
    return 0.0


class MacroField:
    """
    Validates trade signals against macro anchor state.

    Scoring system: each anchor check adds to or subtracts from a consistency
    score in [-1, +1]. Signal is allowed when score > PASS_THRESHOLD.
    """

    PASS_THRESHOLD: float = 0.20   # low bar — macro is veto, not filter
    VETO_THRESHOLD: float = -0.40  # hard veto

    def __init__(self):
        self._macro_states: Dict[str, ManifoldState] = {}

    def update(self, states: Dict[str, ManifoldState]):
        """Called each hub cycle with the latest macro anchor states."""
        anchor_keys = {
            "DXY": ["DX-Y.NYB", "DXY"],
            "SPX": ["^GSPC", "SPX", "SPY"],
            "TLT": ["TLT", "^TNX"],
        }
        for label, keys in anchor_keys.items():
            for k in keys:
                if k in states:
                    self._macro_states[label] = states[k]
                    break

    def check(self, signal: TradeSignal,
              all_states: Dict[str, ManifoldState]) -> Tuple[bool, float, str]:
        """
        Returns (allowed, score, reason_string).
        """
        usd_bias = _usd_bias(signal.symbol, signal.direction)
        score    = 0.0
        notes    = []

        # ── DXY check ─────────────────────────────────────────────────
        dxy = self._macro_states.get("DXY")
        if dxy is not None:
            # If trade implies USD weakness (+1) and DXY is in Phase II/III
            # down-trend → consistent (+0.4)
            dxy_dir = dxy.gradient_direction
            if usd_bias > 0 and dxy_dir == Direction.SELL:
                score += 0.40
                notes.append("DXY_sell_confirms")
            elif usd_bias > 0 and dxy.phase == Phase.THREE and dxy_dir == Direction.SELL:
                score += 0.50
                notes.append("DXY_P3_sell_strong_confirm")
            elif usd_bias > 0 and dxy_dir == Direction.BUY and dxy.phase == Phase.THREE:
                score -= 0.50   # DXY buying hard = USD strong = veto
                notes.append("DXY_P3_buy_VETO")
            elif usd_bias < 0 and dxy_dir == Direction.BUY:
                score += 0.40
                notes.append("DXY_buy_confirms_USD_long")
            elif usd_bias < 0 and dxy_dir == Direction.SELL and dxy.phase == Phase.THREE:
                score -= 0.50
                notes.append("DXY_P3_sell_USD_long_VETO")
        else:
            notes.append("DXY_unavailable")

        # ── SPX check ─────────────────────────────────────────────────
        spx = self._macro_states.get("SPX")
        if spx is not None:
            # Risk-on (SPX Phase I/III up) supports AUD, NZD, EUR long;
            # opposes JPY long (safe haven)
            sym = signal.symbol.upper()
            risk_on_pair = any(x in sym[:3] for x in ["AUD", "NZD", "EUR", "GBP"])
            safe_haven   = "JPY" in sym[:3] or "CHF" in sym[:3] or "XAU" in sym

            spx_up = (spx.gradient_direction == Direction.BUY
                      and spx.phase in [Phase.ONE, Phase.THREE])

            if risk_on_pair and signal.direction == Direction.BUY and spx_up:
                score += 0.25
                notes.append("SPX_risk_on_confirm")
            elif safe_haven and signal.direction == Direction.BUY and spx_up:
                score -= 0.20
                notes.append("SPX_risk_on_vs_safe_haven")
            elif spx.phase == Phase.TWO and safe_haven and signal.direction == Direction.BUY:
                score += 0.15
                notes.append("SPX_P2_safe_haven_ok")
        else:
            notes.append("SPX_unavailable")

        # ── TLT / rates check ─────────────────────────────────────────
        tlt = self._macro_states.get("TLT")
        if tlt is not None:
            sym = signal.symbol.upper()
            # Rising rates (TLT sell) = USD positive, EM negative
            rates_rising = tlt.gradient_direction == Direction.SELL

            if "XAU" in sym and signal.direction == Direction.BUY and rates_rising:
                score -= 0.20   # Gold struggles vs rising rates
                notes.append("rates_rising_vs_gold")
            elif usd_bias > 0 and rates_rising:
                score -= 0.15   # USD-weak pairs headwind
                notes.append("rates_rising_vs_USD_weak")
            elif usd_bias < 0 and rates_rising:
                score += 0.10   # USD-strong helped by rising rates
                notes.append("rates_rising_USD_strong")
        else:
            notes.append("TLT_unavailable")

        # ── Decision ──────────────────────────────────────────────────
        reason = " | ".join(notes)

        if score < self.VETO_THRESHOLD:
            log.info(f"[{signal.symbol}] MACRO VETO score={score:.2f} — {reason}")
            return False, score, reason

        if score < self.PASS_THRESHOLD:
            log.debug(
                f"[{signal.symbol}] macro weak score={score:.2f} — "
                f"allowing with caution"
            )
            # Still allow but caller can reduce size

        return True, score, reason

    def size_adjustment(self, macro_score: float) -> float:
        """
        Returns a multiplier (0.3 – 1.0) to apply to position size
        based on macro consistency score.
        """
        if macro_score >= 0.5:
            return 1.0
        if macro_score >= 0.20:
            return 0.75
        if macro_score >= 0.0:
            return 0.50
        return 0.30
