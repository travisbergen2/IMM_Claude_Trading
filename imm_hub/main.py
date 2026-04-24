"""
IMM AGI Trading Hub — Main
Orchestrates: ManifoldObserver → PhaseDetectors → RiskGate → MacroField
             → EA commands via HubServer.

Run:
    python main.py [--config imm_config.yaml] [--dry-run]
"""
from __future__ import annotations
import asyncio
import argparse
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional

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
from hub_server        import HubServer, start_ngrok

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("imm_hub")


class IMMHub:
    """
    Central AGI orchestrator.

    Cycle (every config.loop_interval seconds):
    1. Observer updates manifold states for all instruments
    2. Receivers step under gradient flow
    3. Macro field updated
    4. Phase detectors emit signals
    5. Signals validated through risk gate + macro field
    6. Commands dispatched to EAs
    7. Active trade exits monitored
    8. DD limits checked
    """

    def __init__(self, config: Config, dry_run: bool = False):
        self.config   = config
        self.dry_run  = dry_run

        # Core modules
        self.receiver_array = ReceiverArray(config)
        self.observer       = ManifoldObserver(config, self.receiver_array)
        self.risk_gate      = RiskGate(config)
        self.macro_field    = MacroField()

        # Per-instrument detectors
        all_tradeable = config.tradeable
        self.p3_detectors: Dict[str, PhaseIIIDetector] = {
            sym: PhaseIIIDetector(sym, config) for sym in all_tradeable
        }
        self.sr_detectors: Dict[str, SpectralRiderDetector] = {
            sym: SpectralRiderDetector(sym, config) for sym in all_tradeable
        }

        # Active trades and their exit engines
        self.open_trades:  Dict[str, Trade]       = {}
        self.exit_engines: Dict[str, ExitEngine]  = {}

        # Latest states (cached for status endpoint)
        self.latest_states: Dict[str, ManifoldState] = {}

        # Hub server (WebSocket + REST)
        self.server = HubServer(hub_callback=self._on_ea_message)
        self.ngrok_url = ""

    # ── Startup ───────────────────────────────────────────────────────

    async def start(self):
        log.info("=" * 60)
        log.info("  IMM AGI Trading Hub — starting")
        log.info(f"  Account: ${self.config.account_size:,.0f}  "
                 f"Daily DD: {self.config.daily_dd_pct:.0%}  "
                 f"Max DD: {self.config.max_dd_pct:.0%}")
        log.info(f"  Instruments: {len(self.config.tradeable)}  "
                 f"Dry-run: {self.dry_run}")
        log.info("=" * 60)

        # Start ngrok
        self.ngrok_url = start_ngrok(self.config.ngrok_port)

        # Run all coroutines concurrently
        await asyncio.gather(
            self.server.start(port=self.config.ngrok_port),
            self._market_loop(),
            self._risk_monitor_loop(),
            self._status_loop(),
        )

    # ── Main market loop ──────────────────────────────────────────────

    async def _market_loop(self):
        log.info("Market loop started.")
        while True:
            try:
                await self._cycle()
            except Exception as e:
                log.error(f"Market loop error: {e}", exc_info=True)
            await asyncio.sleep(self.config.loop_interval)

    async def _cycle(self):
        """One full observation-detection-execution cycle."""
        # ── 1. Fetch all states ───────────────────────────────────────
        states = await self.observer.update_all()
        self.latest_states = states

        # ── 2. Update macro field ─────────────────────────────────────
        self.macro_field.update(states)

        # ── 3. Risk gate daily reset ──────────────────────────────────
        self.risk_gate.check_new_day()

        # ── 4. Check Phase signals for tradeable instruments ──────────
        for sym in self.config.tradeable:
            state = states.get(sym)
            if state is None:
                continue

            # Skip if max concurrent trades reached
            if len(self.open_trades) >= self.config.risk.max_concurrent_trades:
                break

            # Phase III detector
            theta = self.risk_gate.theta_trade()
            signal = self.p3_detectors[sym].update(state, theta)
            if signal is not None:
                await self._process_signal(signal, state, ea_id="remora")

            # Spectral Rider (Phase II probe)
            theta_p = self.risk_gate.theta_probe()
            probe   = self.sr_detectors[sym].update(state, theta_p)
            if probe is not None:
                await self._process_signal(probe, state, ea_id="spectral_rider",
                                            size_factor=0.25)

        # ── 5. Monitor active trade exits ─────────────────────────────
        for trade_id in list(self.exit_engines.keys()):
            trade = self.open_trades.get(trade_id)
            if trade is None:
                continue
            state = states.get(trade.symbol)
            if state is None:
                continue
            action = self.exit_engines[trade_id].update(state, trade.pnl_r)
            if action.reason != ExitReason.HOLD:
                await self._dispatch_exit(trade_id, action, state)

    # ── Signal processing ─────────────────────────────────────────────

    async def _process_signal(self, signal: TradeSignal,
                               state: ManifoldState,
                               ea_id: str,
                               size_factor: float = 1.0):
        """Validate and route a trade signal to the appropriate EA."""

        # Risk gate
        theta = self.risk_gate.theta_trade()
        if state.S < theta:
            log.debug(
                f"[{signal.symbol}] BLOCKED: S={state.S:.1f} < θ={theta:.1f}"
            )
            return

        # Correlated exposure
        if not self.risk_gate.correlated_exposure_ok(signal.symbol,
                                                      signal.direction):
            return

        # Macro consistency
        macro_ok, macro_score, macro_reason = self.macro_field.check(
            signal, self.latest_states
        )
        if not macro_ok:
            return

        macro_adj   = self.macro_field.size_adjustment(macro_score)
        base_lots   = self.risk_gate.position_size(
            signal.symbol, state.S, signal.revival_strength, state.atr
        )
        final_lots  = round(base_lots * size_factor * macro_adj, 2)
        sl_pips     = self.risk_gate.sl_pips(signal.symbol, state.atr)

        if final_lots < 0.01:
            log.debug(f"[{signal.symbol}] Lot size too small ({final_lots}), skipping")
            return

        trade_id = str(uuid.uuid4())[:8]

        cmd = {
            "type":                "ENTER",
            "trade_id":            trade_id,
            "symbol":              signal.symbol,
            "direction":           signal.direction.value,
            "lots":                final_lots,
            "entry_type":          "MARKET",
            "initial_sl_pips":     sl_pips,
            "initial_tp_pips":     0,
            "exit_mode":           "DYNAMIC",
            "revival_strength":    round(signal.revival_strength, 3),
            "delay_bars_estimate": signal.delay_bars,
            "receiver_stability":  round(state.S, 2),
            "phase2_duration_bars": signal.phase2_duration_bars,
            "rho_env_accumulated": round(signal.rho_env_accumulated, 5),
            "macro_score":         round(macro_score, 3),
            "macro_notes":         macro_reason,
        }

        log.info(
            f"⚡ FIRING  {signal.symbol} {signal.direction.value}  "
            f"lots={final_lots}  revival={signal.revival_strength:.2f}  "
            f"S={state.S:.1f}  macro={macro_score:.2f}  ea={ea_id}"
            + (" [DRY RUN]" if self.dry_run else "")
        )

        if not self.dry_run:
            sent = await self.server.send_to_ea(ea_id, cmd)
            if not sent:
                log.warning(f"EA {ea_id} not connected — signal lost")
                return

        # Register trade locally
        initial_risk = final_lots * sl_pips * 0.1   # approximate
        trade = Trade(
            trade_id=trade_id,
            symbol=signal.symbol,
            direction=signal.direction,
            lots=final_lots,
            entry_price=state.current_price,
            entry_time=datetime.utcnow(),
            ea_id=ea_id,
            initial_risk_usd=initial_risk,
        )
        self.open_trades[trade_id]   = trade
        self.exit_engines[trade_id]  = ExitEngine(trade, state)
        self.risk_gate.register_trade(trade)

        # Reset Phase III detector cooldown
        self.p3_detectors[signal.symbol].reset()

    # ── Exit dispatch ─────────────────────────────────────────────────

    async def _dispatch_exit(self, trade_id: str, action: ExitAction,
                              state: ManifoldState):
        trade = self.open_trades.get(trade_id)
        if trade is None:
            return

        cmd: dict = {"trade_id": trade_id, "symbol": trade.symbol}

        if action.reason == ExitReason.CLOSE_ALL:
            cmd["type"] = "CLOSE_ALL"
            cmd["reason"] = action.close_reason
            log.info(f"[{trade.symbol}:{trade_id}] CLOSE ALL — {action.close_reason}")
            await self._finalise_trade(trade_id, state.current_price)

        elif action.reason == ExitReason.CLOSE_PARTIAL:
            cmd["type"]     = "CLOSE_PARTIAL"
            cmd["fraction"] = action.close_fraction
            log.info(
                f"[{trade.symbol}:{trade_id}] CLOSE PARTIAL {action.close_fraction:.0%}"
            )
            trade.lots = round(trade.lots * (1.0 - action.close_fraction), 2)
            if trade.lots < 0.01:
                await self._finalise_trade(trade_id, state.current_price)

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

    async def _finalise_trade(self, trade_id: str, close_price: float):
        trade = self.open_trades.pop(trade_id, None)
        self.exit_engines.pop(trade_id, None)
        if trade:
            direction_sign = 1.0 if trade.direction == Direction.BUY else -1.0
            pnl_pips = direction_sign * (close_price - trade.entry_price) * 10_000.0
            pnl_usd  = pnl_pips * 0.10 * trade.lots * 10.0
            self.risk_gate.close_trade(trade_id, pnl_usd)
            log.info(
                f"[{trade.symbol}:{trade_id}] CLOSED  "
                f"P&L={pnl_pips:.1f}pip  ${pnl_usd:.2f}"
            )

    # ── EA message handler ────────────────────────────────────────────

    async def _on_ea_message(self, ea_id: str, data: dict):
        """Callback from HubServer when an EA sends a message."""
        msg_type = data.get("type", "")

        if msg_type == "tick_update":
            symbol = data.get("symbol", "")
            bid    = float(data.get("bid", 0))
            ask    = float(data.get("ask", 0))
            self.observer.inject_ea_tick(symbol, bid, ask)

            # Update P&L for open trade on this symbol
            trade_id = data.get("trade_id")
            if trade_id and trade_id in self.open_trades:
                open_pnl = float(data.get("open_pnl", 0))
                self.open_trades[trade_id].pnl_usd = open_pnl
                initial_risk = self.open_trades[trade_id].initial_risk_usd
                if initial_risk > 0:
                    self.open_trades[trade_id].pnl_r = open_pnl / initial_risk
                self.risk_gate.update_trade_pnl(trade_id, open_pnl)

        elif msg_type == "heartbeat":
            pass   # last_seen handled in HubServer

        elif msg_type == "trade_opened":
            log.info(f"EA {ea_id}: trade confirmed — {data}")

        elif msg_type == "trade_closed":
            trade_id   = data.get("trade_id", "")
            close_price = float(data.get("close_price", 0))
            if trade_id:
                await self._finalise_trade(trade_id, close_price)

        else:
            log.debug(f"EA {ea_id} unknown message type: {msg_type}")

    # ── Risk monitor loop ─────────────────────────────────────────────

    async def _risk_monitor_loop(self):
        """Dedicated high-frequency DD check. Runs every 5 s."""
        log.info("Risk monitor started.")
        while True:
            try:
                status = self.risk_gate.status_dict()

                # Daily DD at 95% → close all + lock session
                if (status["daily_dd_used"] >=
                        self.risk_gate.daily_dd_hard * 0.95):
                    await self._emergency_close_all("DAILY_DD_95pct")
                    self.risk_gate.session_locked = True

                # Max DD at 92% → close all + critical alert
                if (status["max_dd_used"] >=
                        self.risk_gate.max_dd_hard * 0.92):
                    await self._emergency_close_all("MAX_DD_92pct")
                    log.critical(
                        f"⛔ MAX DD APPROACHING: "
                        f"${status['max_dd_used']:.2f} / "
                        f"${self.risk_gate.max_dd_hard:.2f}"
                    )

            except Exception as e:
                log.error(f"Risk monitor error: {e}")
            await asyncio.sleep(5)

    async def _emergency_close_all(self, reason: str):
        log.warning(f"EMERGENCY CLOSE ALL — reason: {reason}")
        cmd = {"type": "CLOSE_ALL", "reason": reason}
        if not self.dry_run:
            await self.server.broadcast(cmd)
        for trade_id in list(self.open_trades.keys()):
            state = self.latest_states.get(
                self.open_trades[trade_id].symbol
            )
            price = state.current_price if state else 0.0
            await self._finalise_trade(trade_id, price)

    # ── Status loop ───────────────────────────────────────────────────

    async def _status_loop(self):
        """Log a status summary every 60 s."""
        while True:
            await asyncio.sleep(60)
            risk = self.risk_gate.status_dict()
            eas  = self.server.connected_eas()
            phases = {
                sym: states.phase.value
                for sym, states in self.latest_states.items()
                if sym in self.config.tradeable
            } if self.latest_states else {}

            log.info(
                f"STATUS | open={len(self.open_trades)}  "
                f"daily_DD=${risk['daily_dd_used']:.2f}/{risk['daily_dd_hard']:.0f}  "
                f"max_DD=${risk['max_dd_used']:.2f}/{risk['max_dd_hard']:.0f}  "
                f"EAs={eas}  "
                f"phases={phases}"
            )
            self.server.set_hub_state({
                "open_trades": len(self.open_trades),
                "risk": risk,
                "phases": phases,
                "ngrok_url": self.ngrok_url,
                "dry_run": self.dry_run,
            })


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IMM AGI Trading Hub")
    parser.add_argument("--config", default="imm_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without sending commands to EAs")
    args = parser.parse_args()

    try:
        config = Config.from_yaml(args.config)
    except FileNotFoundError:
        log.warning(f"Config file {args.config} not found — using defaults")
        config = Config()

    hub = IMMHub(config, dry_run=args.dry_run)
    asyncio.run(hub.start())


if __name__ == "__main__":
    main()
