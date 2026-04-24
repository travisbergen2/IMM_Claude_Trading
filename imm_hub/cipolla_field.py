"""
IMM AGI Hub — Cipolla Field Analysis
Integrates Carlo Cipolla's Basic Laws of Human Stupidity into market analysis.

Cipolla (1976) identified four actor types on a 2D grid:
  X-axis: benefit to self   Y-axis: benefit to others

  Helpless  (-self, +others)  loses for others' gain
  Intelligent (+self, +others) gains while benefiting market
  Bandit    (+self, -others)  gains at market's expense
  Stupid    (-self, -others)  harms self AND market — the most dangerous

The critical insight for trading:
  "Stupid people cause losses to others with no gain to themselves."
  They are UNPREDICTABLE — unlike bandits who are predictably self-interested.

IMM Translation:
  High Stupid Index  → Phase II compression driver
    Stupid actors inject incoherent noise into the informational manifold.
    Volume moves without information content. Price oscillates randomly.
    ρ_env builds as coherent energy transfers to the background channel.

  Bandit population rising  → Phase III precursor
    Bandits position AGAINST the stupid's noise.
    Smart money accumulates opposite to the retail panic.
    κ (re-coupling strength) increases as bandit flow builds pressure.

  Stupid Index peak then declining  → Phase III trigger signal
    When stupid actors exhaust their capital, the bandit/intelligent
    flow dominates. The revival is forced. We fire.

References:
  Cipolla, C.M. (1976). The Basic Laws of Human Stupidity.
  Available in: "Allegro ma non Troppo" (1988).

Market proxies (no external data required):
  SI (Stupid Index):    Volume-return incoherence + short-lag ACF sign flip
  BI (Bandit Index):    Directional flow on elevated volume
  II (Intelligent):     Coherent price advance with matching volume
  HI (Helpless Index):  Price gives back without volume support
"""
from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional
import numpy as np

log = logging.getLogger("cipolla_field")


@dataclass
class CipollaState:
    """Cipolla quadrant scores for one instrument at one moment."""
    symbol: str

    # Raw scores [0, 1]
    stupid_index:      float  = 0.0   # SI — incoherent noise actors
    bandit_index:      float  = 0.0   # BI — exploitative smart flow
    intelligent_index: float  = 0.0   # II — constructive informed flow
    helpless_index:    float  = 0.0   # HI — passive losers

    # Derived fields
    stupidity_pressure: float = 0.0   # SI momentum (dSI/dt)
    bandit_rising:      bool  = False  # BI accelerating
    stupid_peak:        bool  = False  # SI recently peaked (transition signal)
    phase_suggestion:   int   = 1      # Cipolla-implied phase (1/2/3)

    # Cipolla constant: fraction of population that is always stupid
    # Empirically estimated per instrument
    sigma: float = 0.25   # Cipolla's estimate for general populations

    def cipolla_entropy_adjustment(self) -> float:
        """
        Adjust entropy estimate upward when stupid population is active.
        H_adjusted = H_raw + ΔH_cipolla
        ΔH_cipolla ∈ [0, 0.25]
        """
        return float(np.clip(self.stupid_index * 0.30, 0.0, 0.25))

    def cipolla_kappa_adjustment(self) -> float:
        """
        Adjust re-coupling strength κ upward when bandits are rising
        AND stupid index is peaking (transition setup).
        κ_adjusted = κ_raw × (1 + Δκ_cipolla)
        """
        if self.stupid_peak and self.bandit_rising:
            return float(np.clip(self.bandit_index * 0.50, 0.0, 0.40))
        return 0.0

    def summary(self) -> str:
        dominant = max(
            [("STUPID", self.stupid_index),
             ("BANDIT", self.bandit_index),
             ("INTELLIGENT", self.intelligent_index),
             ("HELPLESS", self.helpless_index)],
            key=lambda x: x[1]
        )[0]
        peak_str = " [PEAK→P3]" if self.stupid_peak else ""
        return (
            f"SI={self.stupid_index:.2f} BI={self.bandit_index:.2f} "
            f"II={self.intelligent_index:.2f} HI={self.helpless_index:.2f} "
            f"→{dominant}{peak_str}"
        )


class CipollaAnalyser:
    """
    Per-instrument Cipolla field tracker.

    Computes population quadrant scores from OHLCV data without
    requiring external sentiment data.

    Core methods:
      update(returns, volumes) → CipollaState
    """

    # Cipolla's universal constant:
    # "In any human population, always and inevitably, a fraction σ
    # will be stupid regardless of intelligence, education, or status."
    # Market application: roughly 25% of volume-days are stupid regardless
    # of overall market regime.
    SIGMA: float = 0.25

    def __init__(self, symbol: str, history_len: int = 50):
        self.symbol       = symbol
        self.history_len  = history_len

        # Rolling history of per-bar scores
        self._si_history: Deque[float] = deque(maxlen=history_len)
        self._bi_history: Deque[float] = deque(maxlen=history_len)
        self._ii_history: Deque[float] = deque(maxlen=history_len)
        self._hi_history: Deque[float] = deque(maxlen=history_len)

        self._last_state: Optional[CipollaState] = None

    # ── Public ───────────────────────────────────────────────────────

    def update(self, returns: np.ndarray,
               volumes: np.ndarray) -> CipollaState:
        """
        Compute Cipolla state from recent returns and volumes.

        Returns CipollaState with all four indices and derived signals.
        """
        if len(returns) < 5 or len(volumes) < 5:
            return CipollaState(symbol=self.symbol)

        n       = min(len(returns), len(volumes) - 1)
        r       = returns[-n:]
        v       = volumes[-n:]     # volumes aligned to return bars

        # ── Core computations ─────────────────────────────────────────

        si = self._stupid_index(r, v)
        bi = self._bandit_index(r, v)
        ii = self._intelligent_index(r, v)
        hi = self._helpless_index(r, v)

        # ── Normalise so they sum to ~1 ───────────────────────────────
        total = si + bi + ii + hi + 1e-9
        si_n  = si / total
        bi_n  = bi / total
        ii_n  = ii / total
        hi_n  = hi / total

        # ── Store history ─────────────────────────────────────────────
        self._si_history.append(si_n)
        self._bi_history.append(bi_n)
        self._ii_history.append(ii_n)
        self._hi_history.append(hi_n)

        # ── Derived signals ───────────────────────────────────────────
        stupidity_pressure, stupid_peak = self._si_momentum()
        bandit_rising                   = self._bi_accelerating()
        phase_suggestion                = self._infer_phase(si_n, bi_n, stupid_peak)

        state = CipollaState(
            symbol=self.symbol,
            stupid_index=round(si_n, 4),
            bandit_index=round(bi_n, 4),
            intelligent_index=round(ii_n, 4),
            helpless_index=round(hi_n, 4),
            stupidity_pressure=round(stupidity_pressure, 4),
            bandit_rising=bandit_rising,
            stupid_peak=stupid_peak,
            phase_suggestion=phase_suggestion,
            sigma=self.SIGMA,
        )

        self._last_state = state
        return state

    def last(self) -> Optional[CipollaState]:
        return self._last_state

    # ── Cipolla quadrant estimators ───────────────────────────────────

    @staticmethod
    def _stupid_index(returns: np.ndarray,
                      volumes: np.ndarray) -> float:
        """
        Stupid actors: act against their own interest AND harm the market.

        Market proxy:
        1. Volume-return incoherence: large volume with tiny price move
           (crowd fighting itself, cancelling out — pure noise injection)
        2. Negative short-lag autocorrelation of volume-weighted returns
           (stupid actors consistently buy tops and sell bottoms — reversals)
        3. Volume spikes on ZERO or negative information content days

        SI rises when:
        - High volume, low absolute return (price going nowhere despite activity)
        - Returns oscillate randomly (chasing)
        - Volume above average on days price closes unchanged
        """
        if len(returns) < 5:
            return 0.25

        abs_ret     = np.abs(returns)
        vol_norm    = volumes / (np.mean(volumes) + 1e-9)

        # Incoherence ratio: high volume / low return = stupid activity
        incoherence = np.mean(vol_norm * (1.0 - np.tanh(abs_ret * 1000)))
        incoherence = float(np.clip(incoherence, 0.0, 2.0)) / 2.0

        # Short-lag oscillation: stupid actors whipsaw
        if len(returns) >= 4:
            sign_changes = np.sum(np.diff(np.sign(returns)) != 0)
            oscillation  = sign_changes / max(len(returns) - 1, 1)
            # Pure random walk = ~0.5 sign changes per bar
            # Stupid > random: oscillation > 0.6
            stupid_oscillation = float(np.clip((oscillation - 0.4) / 0.4, 0.0, 1.0))
        else:
            stupid_oscillation = 0.25

        return float(np.clip(
            0.6 * incoherence + 0.4 * stupid_oscillation, 0.0, 1.0
        ))

    @staticmethod
    def _bandit_index(returns: np.ndarray,
                      volumes: np.ndarray) -> float:
        """
        Bandit actors: gain at the market's expense.

        Market proxy:
        1. Sustained directional move (positive autocorrelation) on HIGH volume
           — smart money positioning, momentum without exhaustion
        2. Volume accelerating into a move (institutional accumulation)
        3. Price advance / decline with volume confirming (not contradicting)

        BI rises when:
        - Recent returns all in same direction AND volume is above average
        - Volume trend aligns with price trend (accumulation/distribution)
        """
        if len(returns) < 3:
            return 0.25

        # Directional coherence: fraction of returns in dominant direction
        n_up  = np.sum(returns > 0)
        n_dn  = len(returns) - n_up
        dom   = max(n_up, n_dn) / max(len(returns), 1)
        # Subtract Cipolla baseline (random = 0.5, stupid adds ~0.1)
        directionality = float(np.clip((dom - 0.55) / 0.40, 0.0, 1.0))

        # Volume alignment: is volume higher on dominant-direction bars?
        if len(returns) >= 4 and len(volumes) >= 4:
            dominant_sign = 1 if n_up > n_dn else -1
            dom_vols  = volumes[:-1][returns[-len(volumes)+1:] * dominant_sign > 0]
            anti_vols = volumes[:-1][returns[-len(volumes)+1:] * dominant_sign <= 0]
            if len(dom_vols) > 0 and len(anti_vols) > 0:
                vol_ratio = np.mean(dom_vols) / (np.mean(anti_vols) + 1e-9)
                vol_alignment = float(np.clip((vol_ratio - 1.0) / 1.5, 0.0, 1.0))
            else:
                vol_alignment = 0.3
        else:
            vol_alignment = 0.3

        return float(np.clip(
            0.55 * directionality + 0.45 * vol_alignment, 0.0, 1.0
        ))

    @staticmethod
    def _intelligent_index(returns: np.ndarray,
                           volumes: np.ndarray) -> float:
        """
        Intelligent actors: gain while benefiting market price discovery.

        Market proxy:
        1. Coherent trend with LOW volatility (efficient, not disruptive)
        2. Consistent volume (not spiky — informed flow is steady)
        3. Positive but not extreme autocorrelation (information being priced in)

        II rises when:
        - Sharp Sharpe-like ratio: high mean return / low std dev
        - Volume is steady (not erratic)
        - Autocorrelation is mildly positive (momentum but not overshoot)
        """
        if len(returns) < 5:
            return 0.25

        mu  = np.mean(returns)
        sig = np.std(returns) + 1e-9
        sharpe_proxy = float(np.clip(abs(mu) / sig * 3.0, 0.0, 1.0))

        # Volume consistency: low coefficient of variation
        cv_vol = float(np.std(volumes) / (np.mean(volumes) + 1e-9))
        vol_steadiness = float(np.clip(1.0 - cv_vol / 2.0, 0.0, 1.0))

        # Mild positive autocorrelation
        if len(returns) >= 4:
            x, y = returns[:-1], returns[1:]
            if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                with np.errstate(invalid='ignore', divide='ignore'):
                    acf1 = float(np.corrcoef(x, y)[0, 1])
                if not np.isfinite(acf1):
                    acf1 = 0.0
            else:
                acf1 = 0.0
            # Target: acf1 in [0.05, 0.35] — too high is overshoot (stupid/bandit)
            acf_score = float(np.clip(
                1.0 - abs(acf1 - 0.20) / 0.30, 0.0, 1.0
            ))
        else:
            acf_score = 0.5

        return float(np.clip(
            0.4 * sharpe_proxy + 0.3 * vol_steadiness + 0.3 * acf_score, 0.0, 1.0
        ))

    @staticmethod
    def _helpless_index(returns: np.ndarray,
                        volumes: np.ndarray) -> float:
        """
        Helpless actors: lose for others' gain.

        Market proxy:
        1. Price gives back gains on rising volume (bag-holders distributing)
        2. Long tail reversals: price extends then violently mean-reverts
        3. Late-cycle volume surge followed immediately by reversal

        HI rises when:
        - Negative returns on high volume (underwater longs forced out)
        - Volume highest on the reversal bar (capitulation)
        """
        if len(returns) < 5:
            return 0.25

        n   = min(len(returns), len(volumes) - 1)
        r   = returns[-n:]
        v   = volumes[-n:]

        # Reversal-volume correlation: helpless = losing money WITH participation
        # High volume on negative return days = forced liquidation
        vol_norm = v / (np.mean(v) + 1e-9)
        # Align volumes to returns
        v_aligned = vol_norm[:len(r)]
        helpless_pressure = np.mean(
            np.maximum(-r, 0) * v_aligned
        ) * 1000.0  # scale to [0,1] range

        return float(np.clip(helpless_pressure, 0.0, 1.0))

    # ── Derived signals ───────────────────────────────────────────────

    def _si_momentum(self) -> tuple:
        """
        Returns (SI velocity, is_SI_peak).
        SI peak = SI was rising for several bars and has now started declining.
        This is the key Phase III precursor.
        """
        if len(self._si_history) < 5:
            return 0.0, False

        si_arr  = np.array(list(self._si_history))
        recent  = si_arr[-3:]
        prior   = si_arr[-6:-3] if len(si_arr) >= 6 else si_arr[:3]

        velocity  = float(np.mean(recent) - np.mean(prior))

        # Peak detection: SI was above sigma, now declining
        si_peak = False
        if len(si_arr) >= 8:
            max_idx = int(np.argmax(si_arr[-8:]))
            if (max_idx < 5 and           # peak was 3+ bars ago
                    si_arr[-8:][max_idx] > self.SIGMA * 1.3 and  # above sigma threshold
                    si_arr[-1] < si_arr[-8:][max_idx] * 0.85):   # now declining >15%
                si_peak = True

        return velocity, si_peak

    def _bi_accelerating(self) -> bool:
        """True if bandit index has been rising for at least 3 bars."""
        if len(self._bi_history) < 4:
            return False
        bi_arr = np.array(list(self._bi_history))
        slope, _ = np.polyfit(range(min(5, len(bi_arr))),
                               bi_arr[-min(5, len(bi_arr)):], 1)
        return bool(slope > 0.002)

    @staticmethod
    def _infer_phase(si: float, bi: float, stupid_peak: bool) -> int:
        """
        Cipolla-derived phase suggestion.

        Phase I: Intelligent + Bandit dominate (coherent informed flow)
        Phase II: Stupid dominates (incoherent noise, compression)
        Phase III: Stupid peaked & declining, Bandit rising (transition)
        """
        if stupid_peak and bi > 0.25:
            return 3
        if si > 0.35:
            return 2
        return 1


class CipollaRegistry:
    """
    Manages one CipollaAnalyser per instrument.
    Called by ManifoldObserver each cycle.
    """

    def __init__(self, symbols: list):
        self._analysers = {
            sym: CipollaAnalyser(sym) for sym in symbols
        }

    def update(self, symbol: str,
               returns: np.ndarray,
               volumes: np.ndarray) -> CipollaState:
        analyser = self._analysers.get(symbol)
        if analyser is None:
            self._analysers[symbol] = CipollaAnalyser(symbol)
            analyser = self._analysers[symbol]
        return analyser.update(returns, volumes)

    def get_state(self, symbol: str) -> Optional[CipollaState]:
        a = self._analysers.get(symbol)
        return a.last() if a else None

    def all_states(self) -> dict:
        return {
            sym: a.last()
            for sym, a in self._analysers.items()
            if a.last() is not None
        }

    def log_summary(self):
        for sym, state in self.all_states().items():
            log.info(f"  [{sym}] Cipolla: {state.summary()}")
