from __future__ import annotations  # must be first real code line
import warnings, numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
"""
IMM AGI Trading Hub — V2
"""
import asyncio
import argparse
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple

from models import (
    Config, ManifoldState, TradeSignal, Trade, ExitAction,
    ExitReason, Direction, Phase
)
from receiver_array    import ReceiverArray
from manifold_observer import ManifoldObserver
from phase_detector    import PhaseIIIDetector, SpectralRiderDetector
from exit_engine       import ExitEngine
from risk_gate         import RiskGate
from macro_field       import MacroField
from cipolla_field     import CipollaRegistry
from orderbook_analyzer import OrderBookRegistry, OrderBookSnapshot, OrderBookState
from mtf_selector      import MTFRegistry, MTFAnalysis
from trade_journal     import TradeJournal, JournalEntry
from hub_server_http   import HubServerHTTP, start_ngrok

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("imm_v2")

import numpy as np


class IMMHubV2:
    """
    Full IMM AGI hub with order book, multi-timeframe, and Cipolla integration.

    Signal pipeline (each cycle):
      1. ManifoldObserver  → ManifoldState per instrument (OHLCV-based phase)
      2. OrderBookRegistry → OrderBookState per instrument (OB-based direction + κ)
      3. MTFRegistry       → MTFAnalysis per instrument (TF scoring + size)
      4. CipollaRegistry   → CipollaState per instrument (population filter)
      5. MacroField        → cross-instrument validation
      6. PhaseIIIDetector  → TradeSignal (gated by all above)
      7. RiskGate          → final size, θ_trade
      8. TradeJournal      → log entry with full context
      9. EA command        → via HTTP polling hub

    Entry requires ALL of:
      ✓ Phase III from ManifoldObserver
      ✓ OB direction aligned with MTF macro direction
      ✓ No active stop-hunt on OB (harvest not in progress)
      ✓ Cipolla SI < 0.40 (stupid actors not dominant)
      ✓ S(R,E) > θ_trade
      ✓ MTF size_multiplier > 0 (at least probe level)
    """

    def __init__(self, config: Config, dry_run: bool = False):
        self.config  = config
        self.dry_run = dry_run

        # ── Core modules ───────────────────────────────────────────────
        self.receiver_array  = ReceiverArray(config)
        self.observer        = ManifoldObserver(config, self.receiver_array)
        self.risk_gate       = RiskGate(config)
        self.macro_field     = MacroField()
        self.journal         = TradeJournal()

        # ── New V2 modules ─────────────────────────────────────────────
        all_syms = list(set(config.tradeable + config.macro_anchors))
        self.ob_registry  = OrderBookRegistry(all_syms)
        self.mtf_registry = MTFRegistry(config.tradeable)
        self.cipolla      = CipollaRegistry(all_syms)

        # ── Per-instrument detectors ───────────────────────────────────
        self.p3_detectors: Dict[str, PhaseIIIDetector] = {
            sym: PhaseIIIDetector(sym, config) for sym in config.tradeable
        }
        self.sr_detectors: Dict[str, SpectralRiderDetector] = {
            sym: SpectralRiderDetector(sym, config) for sym in config.tradeable
        }

        # ── State caches ───────────────────────────────────────────────
        self.latest_states:   Dict[str, ManifoldState]  = {}
        self.latest_ob:       Dict[str, OrderBookState] = {}
        self.latest_mtf:      Dict[str, MTFAnalysis]    = {}
        self.open_trades:     Dict[str, Trade]          = {}
        self.exit_engines:    Dict[str, ExitEngine]     = {}
        self.macro_states:    Dict[str, ManifoldState]  = {}

        # ── HTTP server (TradeLocker/Windows compatible) ───────────────
        self.server    = HubServerHTTP(hub_callback=self._on_ea_message)
        self.ngrok_url = ""

    # ── Startup ───────────────────────────────────────────────────────

    async def start(self):
        log.info("=" * 64)
        log.info("  IMM AGI Hub V2 — Order Book + MTF + Cipolla")
        log.info(f"  Account: ${self.config.account_size:,.0f}  "
                 f"DD: {self.config.daily_dd_pct:.0%}/{self.config.max_dd_pct:.0%}  "
                 f"Dry-run: {self.dry_run}")
        log.info(f"  Instruments: {self.config.tradeable}")
        log.info("=" * 64)

        self.ngrok_url = start_ngrok(
            self.config.ngrok_port,
            auth_token=getattr(self.config, "ngrok_auth_token", None)
        )

        await asyncio.gather(
            self.server.start(port=self.config.ngrok_port),
            self._market_loop(),
            self._risk_monitor_loop(),
            self._status_loop(),
        )

    # ── Main market loop ───────────────────────────────────────────────

    async def _market_loop(self):
        log.info("Market loop V2 started.")
        while True:
            try:
                await self._cycle()
            except Exception as e:
                log.error(f"Market loop error: {e}", exc_info=True)
            await asyncio.sleep(self.config.loop_interval)

    async def _cycle(self):
        # ── 1. Manifold states (OHLCV) ─────────────────────────────────
        states = await self.observer.update_all()
        self.latest_states = states

        # ── 2. Macro field ─────────────────────────────────────────────
        for sym in self.config.macro_anchors:
            if sym in states:
                self.macro_states[sym] = states[sym]
        self.macro_field.update(states)
        self.risk_gate.check_new_day()

        # ── 3. OB + MTF + Cipolla updates ──────────────────────────────
        for sym in self.config.tradeable:
            state = states.get(sym)
            if state is None:
                continue

            # OB state (populated live by EA tick messages with depth)
            ob_state = self.latest_ob.get(sym)

            # MTF: build from manifold state (live TF data fed by EA)
            tf_data     = self._build_tf_data_from_state(sym, state, ob_state)
            ob_dir      = ob_state.weighted_direction if ob_state else 0.0
            ob_phase    = ob_state.ob_phase_vote      if ob_state else 1
            mtf_analysis = self.mtf_registry.update(sym, tf_data, ob_dir, ob_phase)
            self.latest_mtf[sym] = mtf_analysis

            # Cipolla
            if len(state.returns) >= 5:
                volumes = np.ones(len(state.returns)) * 1000.0
                self.cipolla.update(sym, state.returns, volumes)

        # ── 4. Signal detection ────────────────────────────────────────
        for sym in self.config.tradeable:
            if len(self.open_trades) >= self.config.risk.max_concurrent_trades:
                break

            state    = states.get(sym)
            ob_state = self.latest_ob.get(sym)
            mtf      = self.latest_mtf.get(sym)
            cip      = self.cipolla.get_state(sym)

            if state is None:
                continue

            theta  = self.risk_gate.theta_trade()
            signal = self.p3_detectors[sym].update(state, theta)

            if signal is None:
                # Try Spectral Rider probe
                probe = self.sr_detectors[sym].update(state, self.risk_gate.theta_probe())
                if probe:
                    await self._process_signal(probe, state, ob_state, mtf, cip,
                                               ea_id="spectral_rider",
                                               base_size_factor=0.25)
                continue

            await self._process_signal(signal, state, ob_state, mtf, cip,
                                       ea_id="remora",
                                       base_size_factor=1.0)

        # ── 5. Exit management ─────────────────────────────────────────
        for trade_id in list(self.exit_engines.keys()):
            trade = self.open_trades.get(trade_id)
            if not trade:
                continue
            state = states.get(trade.symbol)
            if not state:
                continue
            action = self.exit_engines[trade_id].update(state, trade.pnl_r)
            if action.reason != ExitReason.HOLD:
                await self._dispatch_exit(trade_id, action, state)

        # ── 6. Push status to server ───────────────────────────────────
        self._push_status(states)

    # ── Signal processing ──────────────────────────────────────────────

    async def _process_signal(self,
                               signal:          TradeSignal,
                               state:           ManifoldState,
                               ob_state:        Optional[OrderBookState],
                               mtf:             Optional[MTFAnalysis],
                               cip_state,
                               ea_id:           str,
                               base_size_factor: float = 1.0):
        """
        Full gating pipeline before dispatching to EA.

        Gates (in order):
        1. Risk gate: S(R,E) > θ_trade
        2. Correlated exposure
        3. Macro consistency
        4. OB alignment: if available, direction must agree
        5. Stop-hunt block: don't enter during active harvest
        6. Cipolla gate: stupid index not dominant
        7. MTF size multiplier
        """

        # ── Gate 1: Risk ───────────────────────────────────────────────
        theta = self.risk_gate.theta_trade()
        if state.S < theta:
            log.debug(f"[{signal.symbol}] BLOCK risk: S={state.S:.1f}<θ={theta:.1f}")
            return

        # ── Gate 2: Correlated exposure ────────────────────────────────
        if not self.risk_gate.correlated_exposure_ok(signal.symbol, signal.direction):
            return

        # ── Gate 3: Macro ──────────────────────────────────────────────
        macro_ok, macro_score, macro_reason = self.macro_field.check(
            signal, self.latest_states
        )
        if not macro_ok:
            log.debug(f"[{signal.symbol}] BLOCK macro: {macro_reason}")
            return

        # ── Gate 4: Order book alignment ───────────────────────────────
        ob_size_adj   = 1.0
        ob_block_msg  = ""
        if ob_state is not None:
            ob_dir   = ob_state.weighted_direction
            sig_sign = 1 if signal.direction == Direction.BUY else -1

            # OB direction must agree OR be neutral (abs < 0.10)
            if abs(ob_dir) > 0.10 and np.sign(ob_dir) != sig_sign:
                log.debug(f"[{signal.symbol}] BLOCK OB direction: "
                          f"signal={signal.direction.value} ob_dir={ob_dir:+.3f}")
                return

            # Reduce size if OB is weak/neutral
            if abs(ob_dir) < 0.15:
                ob_size_adj = 0.70

            # Boost size if OB Phase III + direction aligned
            if ob_state.ob_phase_vote == 3 and np.sign(ob_dir) == sig_sign:
                ob_size_adj = min(ob_size_adj * 1.25, 1.25)

        # ── Gate 5: Stop-hunt / harvest block ──────────────────────────
        if ob_state and ob_state.stop_hunt_setup:
            log.debug(f"[{signal.symbol}] BLOCK stop-hunt active — waiting for flush")
            return

        if mtf and mtf.harvest_in_progress:
            log.debug(f"[{signal.symbol}] BLOCK harvest in progress on 1m/5m")
            return

        # ── Gate 6: Cipolla gate ────────────────────────────────────────
        if cip_state is not None:
            if cip_state.stupid_index > 0.40 and not cip_state.stupid_peak:
                log.debug(f"[{signal.symbol}] BLOCK Cipolla SI={cip_state.stupid_index:.2f} "
                          f"(stupid dominant, no peak yet)")
                return
            # Cipolla entropy adjustment to H_market
            state.H_market = float(np.clip(
                state.H_market + cip_state.cipolla_entropy_adjustment(), 0, 1
            ))

        # ── Gate 7: MTF size multiplier ────────────────────────────────
        mtf_size_mult = mtf.size_multiplier if mtf else 0.50
        if mtf_size_mult <= 0:
            return

        # ── Final size calculation ─────────────────────────────────────
        macro_adj  = self.macro_field.size_adjustment(macro_score)
        base_lots  = self.risk_gate.position_size(
            signal.symbol, state.S, signal.revival_strength, state.atr
        )
        final_lots = round(
            base_lots * base_size_factor * mtf_size_mult * ob_size_adj * macro_adj,
            2
        )
        if final_lots < 0.01:
            return

        sl_pips  = self.risk_gate.sl_pips(signal.symbol, state.atr)
        trade_id = str(uuid.uuid4())[:8]

        # ── Build EA command ───────────────────────────────────────────
        best_tf   = mtf.best_tf if mtf else "1h"
        cmd = {
            "type":               "ENTER",
            "trade_id":           trade_id,
            "symbol":             signal.symbol,
            "direction":          signal.direction.value,
            "lots":               final_lots,
            "entry_type":         "MARKET",
            "initial_sl_pips":    sl_pips,
            "initial_tp_pips":    0,
            "exit_mode":          "DYNAMIC",
            # IMM context
            "revival_strength":   round(signal.revival_strength, 3),
            "receiver_stability": round(state.S, 2),
            "phase2_bars":        signal.phase2_duration_bars,
            # MTF context
            "best_tf":            best_tf,
            "mtf_p3_count":       mtf.phase3_count if mtf else 0,
            "mtf_size_mult":      round(mtf_size_mult, 2),
            # OB context
            "ob_kappa":           round(ob_state.kappa_ob, 3) if ob_state else 0,
            "ob_direction":       round(ob_state.weighted_direction, 3) if ob_state else 0,
            "whale_detected":     (ob_state.whale_bid_detected or
                                   ob_state.whale_ask_detected) if ob_state else False,
            # Exit levels from OB walls
            "exit_target_up":     ob_state.exit_target_up if ob_state else 0.0,
            "exit_target_dn":     ob_state.exit_target_dn if ob_state else 0.0,
            # Risk
            "macro_score":        round(macro_score, 3),
        }

        # Log
        whale_tag = "🐋" if cmd["whale_detected"] else ""
        log.info(
            f"⚡ FIRE  {signal.symbol} {signal.direction.value}  "
            f"lots={final_lots}  tf={best_tf}  "
            f"revival={signal.revival_strength:.2f}  S={state.S:.1f}  "
            f"mtf={mtf_size_mult:.0%}  ob_κ={cmd['ob_kappa']}  "
            f"macro={macro_score:.2f}  ea={ea_id}{whale_tag}"
            + (" [DRY]" if self.dry_run else "")
        )

        # Dispatch
        if not self.dry_run:
            sent = await self.server.send_to_ea(ea_id, cmd)
            if not sent:
                log.warning(f"EA {ea_id} not connected")
                return

        # Register trade
        initial_risk = final_lots * sl_pips * 0.10
        trade = Trade(
            trade_id=trade_id, symbol=signal.symbol,
            direction=signal.direction, lots=final_lots,
            entry_price=state.current_price, entry_time=datetime.now(timezone.utc),
            ea_id=ea_id, initial_risk_usd=initial_risk,
        )
        self.open_trades[trade_id]  = trade
        self.exit_engines[trade_id] = ExitEngine(trade, state)
        self.risk_gate.register_trade(trade)
        self.p3_detectors[signal.symbol].reset()

        # Journal
        R = state.R
        cip_si = cip_state.stupid_index if cip_state else 0.0
        self.journal.open_trade(JournalEntry(
            trade_id=trade_id, symbol=signal.symbol,
            direction=signal.direction.value,
            timestamp_open=datetime.now(timezone.utc).isoformat(),
            lots=final_lots, entry_price=state.current_price,
            initial_sl_pips=sl_pips, initial_risk_usd=initial_risk,
            phase_at_entry=state.phase.value,
            phase_confidence=state.phase_confidence,
            revival_strength=signal.revival_strength,
            receiver_stability_S=state.S,
            spectral_gap_delta=state.delta,
            entropy_H=state.H_market, time_pressure_P=state.P_time,
            psi_sq_at_entry=state.psi_sq, rho_env_at_entry=state.rho_env,
            N_tot_at_entry=state.N_tot, kappa_at_entry=state.kappa,
            gamma_rate_at_entry=state.gamma_rate,
            phase2_duration_bars=signal.phase2_duration_bars,
            rho_env_accumulated=signal.rho_env_accumulated,
            R_TI=R[0], R_SG=R[1], R_FT=R[2], R_UE=R[3], R_AR=R[4],
            macro_score=macro_score, macro_notes=macro_reason,
            cipolla_si=cip_si, ea_id=ea_id,
            daily_dd_used_at_entry=self.risk_gate.daily_dd_used,
            max_dd_used_at_entry=self.risk_gate.max_dd_used,
        ))

    # ── MTF data builder ───────────────────────────────────────────────

    def _build_tf_data_from_state(self, sym: str, state: ManifoldState,
                                   ob_state: Optional[OrderBookState]) -> dict:
        """
        Build MTF-compatible data dict from the primary manifold state.
        In live trading the EA sends multi-TF data; this is the fallback
        using the primary (5m) observation with scaled parameters.
        """
        cip = self.cipolla.get_state(sym)
        si_peak = cip.stupid_peak if cip else False

        # Scale the observed parameters across TF hierarchy
        # Using the known κ_natural values per TF
        base = {
            "phase":    state.phase.value,
            "psi_sq":   state.psi_sq,
            "rho_env":  state.rho_env,
            "N_tot":    state.N_tot,
            "delta":    state.delta,
            "kappa":    state.kappa,
            "gamma_rate": state.gamma_rate,
            "revival_threshold": state.revival_threshold,
            "H":        state.H_market,
            "direction": (1.0 if state.gradient_direction == Direction.BUY else
                          -1.0 if state.gradient_direction == Direction.SELL else 0.0),
            "phase2_bars": 0,
            "confidence": state.phase_confidence,
            "si_peak":   si_peak,
        }

        # Use OB direction to enrich finer TF estimates if available
        if ob_state:
            base["direction"] = float(np.clip(
                0.6 * base["direction"] + 0.4 * ob_state.weighted_direction,
                -1.0, 1.0
            ))

        # Return same data for all TFs (will be differentiated when
        # EA sends per-TF data via the multi-TF tick message format)
        return {"1m": base, "5m": base, "15m": base, "1h": base, "4h": base}

    # ── Exit dispatch ──────────────────────────────────────────────────

    async def _dispatch_exit(self, trade_id: str, action: ExitAction,
                              state: ManifoldState):
        trade = self.open_trades.get(trade_id)
        if not trade:
            return

        cmd: dict = {"trade_id": trade_id, "symbol": trade.symbol}

        if action.reason == ExitReason.CLOSE_ALL:
            cmd["type"]   = "CLOSE_ALL"
            cmd["reason"] = action.close_reason
            log.info(f"[{trade.symbol}:{trade_id}] CLOSE ALL — {action.close_reason}")
            await self._finalise_trade(trade_id, state.current_price, action.close_reason)

        elif action.reason == ExitReason.CLOSE_PARTIAL:
            cmd["type"]           = "CLOSE_PARTIAL"
            cmd["close_fraction"] = action.close_fraction
            log.info(f"[{trade.symbol}:{trade_id}] PARTIAL {action.close_fraction:.0%}")
            trade.lots = round(trade.lots * (1.0 - action.close_fraction), 2)
            if trade.lots < 0.01:
                await self._finalise_trade(trade_id, state.current_price, "partial_full")

        elif action.reason == ExitReason.TRAIL_STOP:
            cmd["type"]      = "TRAIL_STOP"
            cmd["new_stop"]  = action.new_stop
            trade.current_stop = action.new_stop

        elif action.reason == ExitReason.MOVE_TO_BE_PLUS:
            cmd["type"]     = "MOVE_TO_BE_PLUS"
            cmd["new_stop"] = trade.entry_price + (
                0.0005 if trade.direction == Direction.BUY else -0.0005
            )

        if not self.dry_run:
            await self.server.send_to_ea(trade.ea_id, cmd)

    async def _finalise_trade(self, trade_id: str, close_price: float,
                               reason: str = "closed"):
        trade = self.open_trades.pop(trade_id, None)
        ee    = self.exit_engines.pop(trade_id, None)
        if not trade:
            return

        direction_sign = 1.0 if trade.direction == Direction.BUY else -1.0
        pnl_pips = direction_sign * (close_price - trade.entry_price) * 10_000.0
        pnl_usd  = pnl_pips * 0.10 * trade.lots * 10.0

        self.risk_gate.close_trade(trade_id, pnl_usd)

        bars = ee.bars_in_trade if ee else 0
        state = self.latest_states.get(trade.symbol)
        psi_close = state.psi_sq if state else 0.0
        rv_rem    = (psi_close / state.N_tot if state and state.N_tot > 0 else 0.0)

        self.journal.close_trade(
            trade_id=trade_id, close_price=close_price,
            close_reason=reason, realised_pnl=pnl_usd,
            bars_in_trade=bars,
            phase_at_close=state.phase.value if state else 0,
            psi_sq_at_close=psi_close,
            revival_remaining=rv_rem,
        )
        log.info(f"[{trade.symbol}:{trade_id}] CLOSED "
                 f"P&L={pnl_pips:.1f}pip  ${pnl_usd:.2f}  reason={reason}")

    # ── EA message handler ─────────────────────────────────────────────

    async def _on_ea_message(self, ea_id: str, data: dict):
        msg_type = data.get("type", "")

        if msg_type == "tick_update":
            symbol = data.get("symbol", "")
            bid    = float(data.get("bid", 0))
            ask    = float(data.get("ask", 0))

            if symbol and bid and ask:
                self.observer.inject_ea_tick(symbol, bid, ask)

                # ── Parse L2 depth if present ──────────────────────────
                # EA sends depth as: "bids": [[price,vol],...], "asks": [[price,vol],...]
                raw_bids = data.get("bids", [])
                raw_asks = data.get("asks", [])
                if raw_bids and raw_asks:
                    await self._process_depth(symbol, bid, ask, raw_bids, raw_asks)

                # ── Multi-TF data if present ───────────────────────────
                tf_data = data.get("tf_data")
                if tf_data and symbol in self.latest_mtf:
                    ob_s = self.latest_ob.get(symbol)
                    self.latest_mtf[symbol] = self.mtf_registry.update(
                        symbol, tf_data,
                        ob_direction=ob_s.weighted_direction if ob_s else 0.0,
                        ob_phase=ob_s.ob_phase_vote if ob_s else 1,
                    )

            # Update open trade P&L
            trade_id = data.get("trade_id", "")
            if trade_id and trade_id in self.open_trades:
                open_pnl = float(data.get("open_pnl", 0))
                self.open_trades[trade_id].pnl_usd = open_pnl
                ir = self.open_trades[trade_id].initial_risk_usd
                if ir > 0:
                    self.open_trades[trade_id].pnl_r = open_pnl / ir
                self.risk_gate.update_trade_pnl(trade_id, open_pnl)

        elif msg_type == "trade_opened":
            log.info(f"EA {ea_id}: confirmed OPEN {data.get('trade_id','?')}")

        elif msg_type == "trade_closed":
            tid = data.get("trade_id", "")
            cp  = float(data.get("close_price", 0))
            if tid:
                await self._finalise_trade(tid, cp, "ea_confirmed_close")

        elif msg_type == "order_error":
            log.error(f"EA {ea_id}: order error {data}")

    async def _process_depth(self, symbol: str, bid: float, ask: float,
                              raw_bids: list, raw_asks: list):
        """Parse L2 depth from EA and feed OrderBookRegistry."""
        try:
            # EA sends [[price, volume], ...] for each side
            # Build snapshots at each granularity by bucketing
            mid = (bid + ask) / 2.0
            bids_parsed = [(float(p), float(v)) for p, v in raw_bids if p and v]
            asks_parsed = [(float(p), float(v)) for p, v in raw_asks if p and v]

            if not bids_parsed or not asks_parsed:
                return

            # Create snapshots at three granularities by rebucketing
            from orderbook_analyzer import OrderBookSnapshot, GRANULARITIES
            snapshots = {}
            for g in GRANULARITIES:
                # Bucket volumes into granularity-sized price buckets
                bid_buckets: dict = {}
                for p, v in bids_parsed:
                    bucket = round(int(p / g) * g, 8)
                    bid_buckets[bucket] = bid_buckets.get(bucket, 0) + v

                ask_buckets: dict = {}
                for p, v in asks_parsed:
                    bucket = round(int(p / g) * g + g, 8)
                    ask_buckets[bucket] = ask_buckets.get(bucket, 0) + v

                b_sorted = sorted(bid_buckets.items(), reverse=True)[:30]
                a_sorted = sorted(ask_buckets.items())[:30]

                snapshots[g] = OrderBookSnapshot(b_sorted, a_sorted, g)

            ob_state = self.ob_registry.update(symbol, snapshots)
            self.latest_ob[symbol] = ob_state

            # Log notable detections
            if ob_state.whale_bid_detected or ob_state.whale_ask_detected:
                side = "bid" if ob_state.whale_bid_detected else "ask"
                log.info(f"[{symbol}] 🐋 WHALE WALL on {side}")
            if ob_state.spoof_bid_detected or ob_state.spoof_ask_detected:
                side = "bid" if ob_state.spoof_bid_detected else "ask"
                log.warning(f"[{symbol}] ⚠ SPOOF on {side}")
            if ob_state.stop_hunt_setup:
                log.info(f"[{symbol}] 🎯 STOP HUNT setup detected")

        except Exception as e:
            log.debug(f"depth parse error {symbol}: {e}")

    # ── Risk monitor ───────────────────────────────────────────────────

    async def _risk_monitor_loop(self):
        log.info("Risk monitor started.")
        while True:
            try:
                status = self.risk_gate.status_dict()
                if status["daily_dd_used"] >= self.risk_gate.daily_dd_hard * 0.95:
                    await self._emergency_close_all("DAILY_DD_95pct")
                    self.risk_gate.session_locked = True
                if status["max_dd_used"] >= self.risk_gate.max_dd_hard * 0.92:
                    await self._emergency_close_all("MAX_DD_92pct")
                    log.critical(f"⛔ MAX DD APPROACHING: "
                                 f"${status['max_dd_used']:.0f}/${self.risk_gate.max_dd_hard:.0f}")
            except Exception as e:
                log.error(f"Risk monitor: {e}")
            await asyncio.sleep(5)

    async def _emergency_close_all(self, reason: str):
        log.warning(f"EMERGENCY CLOSE ALL — {reason}")
        cmd = {"type": "CLOSE_ALL", "reason": reason}
        if not self.dry_run:
            await self.server.broadcast(cmd)
        for tid in list(self.open_trades.keys()):
            state = self.latest_states.get(self.open_trades[tid].symbol)
            price = state.current_price if state else 0.0
            await self._finalise_trade(tid, price, reason)

    # ── Status loop ────────────────────────────────────────────────────

    async def _status_loop(self):
        while True:
            await asyncio.sleep(60)
            try:
                risk  = self.risk_gate.status_dict()
                eas   = self.server.connected_eas()
                stats = self.journal.session_stats()

                phases = {
                    sym: self.latest_states[sym].phase.value
                    for sym in self.config.tradeable
                    if sym in self.latest_states
                }
                ob_dirs = {
                    sym: round(self.latest_ob[sym].weighted_direction, 3)
                    for sym in self.latest_ob
                }
                mtf_info = {
                    sym: {
                        "p3": self.latest_mtf[sym].phase3_count,
                        "size": self.latest_mtf[sym].size_multiplier,
                        "harvest": self.latest_mtf[sym].harvest_in_progress,
                    }
                    for sym in self.latest_mtf
                }

                log.info(
                    f"STATUS | open={len(self.open_trades)}  "
                    f"daily_DD=${risk['daily_dd_used']:.0f}/{risk['daily_dd_hard']:.0f}  "
                    f"max_DD=${risk['max_dd_used']:.0f}/{risk['max_dd_hard']:.0f}  "
                    f"EAs={eas}  trades={stats.get('trades',0)}  "
                    f"WR={stats.get('win_rate',0):.0%}"
                )

                self.server.set_hub_state({
                    "open_trades":   len(self.open_trades),
                    "risk":          risk,
                    "phases":        phases,
                    "ob_directions": ob_dirs,
                    "mtf":           mtf_info,
                    "session_stats": stats,
                    "ngrok_url":     self.ngrok_url,
                    "dry_run":       self.dry_run,
                    "recent_signals": [],
                })
            except Exception as e:
                log.error(f"Status loop: {e}")

    def _push_status(self, states: dict):
        """Lightweight status push each cycle (for dashboard)."""
        phases = {sym: s.phase.value for sym, s in states.items()
                  if sym in self.config.tradeable}
        self.server.set_hub_state({
            "open_trades": len(self.open_trades),
            "risk": self.risk_gate.status_dict(),
            "phases": phases,
            "dry_run": self.dry_run,
            "ngrok_url": self.ngrok_url,
        })


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IMM AGI Hub V2")
    parser.add_argument("--config",  default="imm_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        config = Config.from_yaml(args.config)
    except FileNotFoundError:
        log.warning(f"Config {args.config} not found — using defaults")
        config = Config()

    hub = IMMHubV2(config, dry_run=args.dry_run)
    asyncio.run(hub.start())


if __name__ == "__main__":
    main()
