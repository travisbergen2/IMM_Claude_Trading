"""
IMM AGI Hub — Trade Journal
Logs every trade with full IMM state context.
Writes JSONL file + in-memory session stats.
Provides session summary with Phase III accuracy metrics.
"""
from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import List, Optional, Dict
import numpy as np

log = logging.getLogger("trade_journal")

JOURNAL_FILE = "imm_trades.jsonl"


@dataclass
class JournalEntry:
    # Identity
    trade_id:      str
    symbol:        str
    direction:     str
    timestamp_open: str
    timestamp_close: Optional[str] = None

    # Entry context
    lots:               float = 0.0
    entry_price:        float = 0.0
    initial_sl_pips:    int   = 0
    initial_risk_usd:   float = 0.0

    # Exit context
    close_price:        float = 0.0
    close_reason:       str   = ""
    realised_pnl_usd:   float = 0.0
    pnl_r:              float = 0.0
    bars_in_trade:      int   = 0

    # IMM state at entry
    phase_at_entry:          int   = 0
    phase_confidence:        float = 0.0
    revival_strength:        float = 0.0
    receiver_stability_S:    float = 0.0
    spectral_gap_delta:      float = 0.0
    entropy_H:               float = 0.0
    time_pressure_P:         float = 0.0
    psi_sq_at_entry:         float = 0.0
    rho_env_at_entry:        float = 0.0
    N_tot_at_entry:          float = 0.0
    kappa_at_entry:          float = 0.0
    gamma_rate_at_entry:     float = 0.0
    phase2_duration_bars:    int   = 0
    rho_env_accumulated:     float = 0.0

    # IMM state at close
    phase_at_close:          int   = 0
    psi_sq_at_close:         float = 0.0
    revival_remaining:       float = 0.0

    # Receiver state (R vector at entry)
    R_TI: float = 0.0
    R_SG: float = 0.0
    R_FT: float = 0.0
    R_UE: float = 0.0
    R_AR: float = 0.0

    # Macro state
    macro_score:    float = 0.0
    macro_notes:    str   = ""
    ea_id:          str   = ""

    # Challenge tracking
    daily_dd_used_at_entry:  float = 0.0
    max_dd_used_at_entry:    float = 0.0


class TradeJournal:
    """
    Records every trade with full IMM context.
    Writes to JSONL file for persistence.
    Provides real-time session metrics.
    """

    def __init__(self, journal_path: str = JOURNAL_FILE):
        self.journal_path = journal_path
        self._open_entries:  Dict[str, JournalEntry] = {}
        self._closed_entries: List[JournalEntry]     = []
        self._session_start = datetime.now(timezone.utc)
        self._load_today()

    # ── Public API ────────────────────────────────────────────────────

    def open_trade(self, entry: JournalEntry):
        """Called when a trade is opened."""
        self._open_entries[entry.trade_id] = entry
        log.info(
            f"JOURNAL OPEN  {entry.trade_id}  "
            f"{entry.symbol} {entry.direction}  "
            f"lots={entry.lots}  S={entry.receiver_stability_S:.1f}  "
            f"revival={entry.revival_strength:.2f}"
        )

    def close_trade(self, trade_id: str,
                    close_price: float,
                    close_reason: str,
                    realised_pnl: float,
                    bars_in_trade: int,
                    phase_at_close: int = 0,
                    psi_sq_at_close: float = 0.0,
                    revival_remaining: float = 0.0):
        """Called when a trade is closed."""
        entry = self._open_entries.pop(trade_id, None)
        if entry is None:
            return

        entry.timestamp_close    = datetime.now(timezone.utc).isoformat()
        entry.close_price        = close_price
        entry.close_reason       = close_reason
        entry.realised_pnl_usd   = realised_pnl
        entry.bars_in_trade      = bars_in_trade
        entry.phase_at_close     = phase_at_close
        entry.psi_sq_at_close    = psi_sq_at_close
        entry.revival_remaining  = revival_remaining

        if entry.initial_risk_usd > 0:
            entry.pnl_r = realised_pnl / entry.initial_risk_usd

        self._closed_entries.append(entry)
        self._write(entry)

        emoji = "✅" if realised_pnl > 0 else "❌"
        log.info(
            f"JOURNAL CLOSE {trade_id}  "
            f"{entry.symbol} {entry.direction}  "
            f"P&L=${realised_pnl:.2f} ({entry.pnl_r:+.2f}R)  "
            f"reason={close_reason}  {emoji}"
        )

    # ── Session metrics ───────────────────────────────────────────────

    def session_stats(self) -> dict:
        closed = self._closed_entries
        if not closed:
            return {
                "trades": 0, "open": len(self._open_entries),
                "message": "No closed trades this session"
            }

        n       = len(closed)
        winners = [t for t in closed if t.realised_pnl_usd > 0]
        losers  = [t for t in closed if t.realised_pnl_usd <= 0]

        win_rate  = len(winners) / n
        total_pnl = sum(t.realised_pnl_usd for t in closed)
        total_r   = sum(t.pnl_r for t in closed)
        avg_win   = (np.mean([t.realised_pnl_usd for t in winners])
                     if winners else 0.0)
        avg_loss  = (np.mean([t.realised_pnl_usd for t in losers])
                     if losers else 0.0)
        expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss)

        # Phase III accuracy: what fraction were actually Phase III at entry?
        p3_entries = [t for t in closed if t.phase_at_entry == 3]
        p3_win_rate = (
            sum(1 for t in p3_entries if t.realised_pnl_usd > 0) / len(p3_entries)
            if p3_entries else 0.0
        )

        # Avg revival strength for winners vs losers
        avg_revival_win  = np.mean([t.revival_strength for t in winners]) if winners else 0.0
        avg_revival_loss = np.mean([t.revival_strength for t in losers])  if losers  else 0.0

        # Best close reason
        close_reason_counts: Dict[str, int] = {}
        for t in closed:
            close_reason_counts[t.close_reason] = \
                close_reason_counts.get(t.close_reason, 0) + 1

        return {
            "session_start":    self._session_start.isoformat(),
            "trades":           n,
            "open":             len(self._open_entries),
            "win_rate":         round(win_rate, 3),
            "total_pnl_usd":    round(total_pnl, 2),
            "total_r":          round(total_r, 3),
            "avg_win_usd":      round(avg_win, 2),
            "avg_loss_usd":     round(avg_loss, 2),
            "expectancy_usd":   round(expectancy, 2),
            "profit_factor":    round(
                abs(sum(t.realised_pnl_usd for t in winners)) /
                max(abs(sum(t.realised_pnl_usd for t in losers)), 0.01),
                3
            ),
            "phase3_entries":   len(p3_entries),
            "phase3_win_rate":  round(p3_win_rate, 3),
            "avg_revival_winners":  round(avg_revival_win, 3),
            "avg_revival_losers":   round(avg_revival_loss, 3),
            "close_reason_counts":  close_reason_counts,
            "avg_bars_in_trade": round(np.mean([t.bars_in_trade for t in closed]), 1),
        }

    def print_session_summary(self):
        s = self.session_stats()
        print("\n" + "=" * 60)
        print("  IMM AGI HUB — SESSION SUMMARY")
        print("=" * 60)
        print(f"  Trades:        {s.get('trades', 0)} closed, {s.get('open', 0)} open")
        print(f"  Win rate:      {s.get('win_rate', 0):.1%}")
        print(f"  Total P&L:     ${s.get('total_pnl_usd', 0):.2f}  ({s.get('total_r', 0):+.2f}R)")
        print(f"  Profit factor: {s.get('profit_factor', 0):.2f}")
        print(f"  Expectancy:    ${s.get('expectancy_usd', 0):.2f}")
        print(f"  Phase III wr:  {s.get('phase3_win_rate', 0):.1%}  ({s.get('phase3_entries', 0)} trades)")
        print(f"  Avg revival (W/L): {s.get('avg_revival_winners', 0):.2f} / {s.get('avg_revival_losers', 0):.2f}")
        print(f"  Avg bars held: {s.get('avg_bars_in_trade', 0):.1f}")
        print("=" * 60 + "\n")

    # ── Persistence ───────────────────────────────────────────────────

    def _write(self, entry: JournalEntry):
        try:
            with open(self.journal_path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except Exception as e:
            log.error(f"Journal write failed: {e}")

    def _load_today(self):
        """Load today's closed trades from file on startup."""
        if not os.path.exists(self.journal_path):
            return
        today = datetime.now(timezone.utc).date().isoformat()
        try:
            with open(self.journal_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if d.get("timestamp_open", "").startswith(today):
                            entry = JournalEntry(**{
                                k: v for k, v in d.items()
                                if k in JournalEntry.__dataclass_fields__
                            })
                            if entry.timestamp_close:
                                self._closed_entries.append(entry)
                            else:
                                self._open_entries[entry.trade_id] = entry
                    except Exception:
                        pass
            log.info(
                f"Journal loaded: {len(self._closed_entries)} closed, "
                f"{len(self._open_entries)} open from today"
            )
        except Exception as e:
            log.warning(f"Journal load failed: {e}")
