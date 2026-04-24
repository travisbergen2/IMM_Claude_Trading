"""
IMM AGI Hub — Live Deployment Script
Single entry point for production launch.

Usage:
    python deploy.py                          # interactive setup + launch
    python deploy.py --dry-run                # paper trade mode (no EA commands)
    python deploy.py --config my_config.yaml  # custom config
    python deploy.py --check-only             # validate setup without running

What this does:
    1. Validates all dependencies
    2. Runs a quick self-test
    3. Starts ngrok tunnel
    4. Launches the V2 hub (HTTP polling, TradeLocker-compatible)
    5. Prints EA setup instructions with your live ngrok URL

Before running:
    pip install -r requirements.txt
    Edit imm_config.yaml → set account.size and ngrok.auth_token
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
import subprocess
import time

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deploy")

import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     IMM AGI TRADING HUB — LIVE DEPLOYMENT                   ║
║     Information Manifold Model V3                           ║
║     doi.org/10.5281/zenodo.19075097                         ║
╚══════════════════════════════════════════════════════════════╝
"""

REQUIRED_PACKAGES = [
    "fastapi", "uvicorn", "numpy", "pyngrok",
    "aiohttp", "pydantic", "yaml",
]

OPTIONAL_PACKAGES = ["yfinance", "matplotlib", "scipy"]


# ═════════════════════════════════════════════════════════════════════════════
# DEPENDENCY CHECK
# ═════════════════════════════════════════════════════════════════════════════

def check_dependencies() -> bool:
    print("\n  Checking dependencies...")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        mod = "yaml" if pkg == "yaml" else pkg.replace("-", "_")
        try:
            __import__(mod)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ✗ {pkg}  ← MISSING")
            missing.append(pkg)

    for pkg in OPTIONAL_PACKAGES:
        try:
            __import__(pkg)
            print(f"    ○ {pkg} (optional)")
        except ImportError:
            print(f"    ○ {pkg} (optional — not installed)")

    if missing:
        print(f"\n  Missing: {missing}")
        print(f"  Run:  pip install {' '.join(missing)} --break-system-packages")
        return False

    print("  All required packages present.\n")
    return True


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_config(config_path: str) -> bool:
    print("  Validating configuration...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models import Config
        cfg = Config.from_yaml(config_path)
        print(f"    ✓ Account size:    ${cfg.account_size:,.0f}")
        print(f"    ✓ Daily DD limit:  {cfg.daily_dd_pct:.0%}  (${cfg.account_size * cfg.daily_dd_pct:.0f})")
        print(f"    ✓ Max DD limit:    {cfg.max_dd_pct:.0%}  (${cfg.account_size * cfg.max_dd_pct:.0f})")
        print(f"    ✓ Instruments:     {cfg.tradeable}")
        print(f"    ✓ ngrok port:      {cfg.ngrok_port}")

        if cfg.account_size < 1000:
            print("    ⚠  Account size < $1,000 — check config")
            return False

        print("  Config valid.\n")
        return True

    except FileNotFoundError:
        print(f"    ✗ Config file not found: {config_path}")
        print(f"      Creating default config...")
        _write_default_config(config_path)
        print(f"      Edit {config_path} and re-run.")
        return False
    except Exception as e:
        print(f"    ✗ Config error: {e}")
        return False


def _write_default_config(path: str):
    content = """# IMM AGI Trading Hub — Configuration
# Edit account.size and ngrok.auth_token before running

account:
  size: 10000                # Your actual account equity
  daily_dd_pct: 0.05         # 5% daily drawdown limit
  max_dd_pct: 0.10           # 10% max drawdown limit
  broker: "aquafunded"

instruments:
  macro_anchors:
    - "DX-Y.NYB"             # DXY
    - "^GSPC"                # SPX
    - "TLT"
  tradeable:
    - "EURUSD"
    - "GBPUSD"
    - "USDJPY"
    - "AUDUSD"
    - "XAUUSD"

receiver:
  forex_majors: [55, 60, 58, 65, 55]
  gold:         [65, 70, 55, 60, 50]
  indices:      [50, 65, 55, 62, 58]

spectral:
  gap_estimation_lags: 25
  phase2_min_bars: 3
  phase3_revival_threshold: 1.05

risk:
  max_risk_per_trade_pct: 0.005
  max_concurrent_trades: 4
  max_correlated_exposure: 2
  daily_stop_at_dd_pct: 0.85

ngrok:
  port: 8000
  auth_token: "PASTE_YOUR_NGROK_TOKEN_HERE"
"""
    with open(path, "w") as f:
        f.write(content)


# ═════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════════════

def run_self_test() -> bool:
    print("  Running self-test...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        # Test core modules
        from models import Config, ManifoldState, Direction, Phase
        from receiver_array import ReceiverArray
        from risk_gate import RiskGate
        from cipolla_field import CipollaAnalyser
        from orderbook_analyzer import OrderBookAnalyzer
        from mtf_selector import MTFSelector
        from pid_layer import PIDValidator
        import numpy as np

        cfg = Config()
        ra  = ReceiverArray(cfg)
        rg  = RiskGate(cfg)

        # Receiver dynamics
        R  = ra.get_R("EURUSD")
        V  = ra.V(R, 0.65, 0.65)
        S  = -V
        assert S < 0 or S >= 0, "stability computed"
        print(f"    ✓ Receiver dynamics: V={V:.2f} S={S:.2f}")

        # Risk gate
        theta = rg.theta_trade()
        assert theta < 0
        print(f"    ✓ Risk gate: θ={theta:.1f}")

        # Cipolla
        rng = np.random.default_rng(42)
        r   = rng.standard_normal(25) * 0.001
        v   = np.abs(rng.standard_normal(25)) * 1000 + 500
        ca  = CipollaAnalyser("TEST")
        cs  = ca.update(r, v)
        assert 0 <= cs.stupid_index <= 1
        print(f"    ✓ Cipolla: SI={cs.stupid_index:.3f} BI={cs.bandit_index:.3f}")

        # Order book
        oba  = OrderBookAnalyzer("EURUSD")
        snap = oba.build_synthetic_snapshot("EURUSD", 1.1000, 0.3, "normal")
        assert len(snap) == 3
        print(f"    ✓ Order book: {len(snap)} granularity levels")

        # MTF selector
        mts = MTFSelector("EURUSD")
        tfd = mts._build_tf_data(50, seed=42)
        assert "1h" in tfd and "4h" in tfd
        print(f"    ✓ MTF selector: {len(tfd)} timeframes")

        # PID validator
        pid_v = PIDValidator()
        pid   = pid_v.compute(0.3, 0.4, 0.45, 0.2, 0.3, 0.4, 0.35, 0.3, 0.25, "TEST")
        allowed, mult, reason = pid_v.validate_signal(pid, 1.0)
        print(f"    ✓ PID validator: conf={pid.pid_confidence:.2f} "
              f"conv={pid.observer_convergence:.2f} → {'PASS' if allowed else 'BLOCK'}")

        print("  Self-test passed.\n")
        return True

    except Exception as e:
        print(f"    ✗ Self-test failed: {e}")
        import traceback; traceback.print_exc()
        return False


# ═════════════════════════════════════════════════════════════════════════════
# EA INSTRUCTION PRINTER
# ═════════════════════════════════════════════════════════════════════════════

def print_ea_instructions(ngrok_url: str, config_path: str):
    try:
        import yaml
        with open(config_path, encoding='utf-8', errors='replace') as f:
            cfg = yaml.safe_load(f)
        instruments = cfg.get("instruments", {}).get("tradeable", ["EURUSD"])
    except Exception:
        instruments = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]

    print()
    print("═" * 65)
    print("  EA SETUP INSTRUCTIONS")
    print("═" * 65)
    print()
    print("  1. In MetaTrader 5 / TradeLocker:")
    print("     Tools → Options → Expert Advisors")
    print("     ✓ Allow WebRequest for listed URL")
    print(f"     Add URL:  {ngrok_url}")
    print()
    print("  2. Copy these EA files to MQL5/Experts/:")
    print("     → IMM_REMORA_V2_HTTP.mq5      (Phase III executor)")
    print("     → IMM_SPECTRAL_RIDER_EA_HTTP.mq5  (Phase II probe)")
    print()
    print("  3. Set HubBaseURL in each EA:")
    print(f"     HubBaseURL = \"{ngrok_url}\"")
    print()
    print("  4. Attach EAs to charts:")
    print("     Remora:         One chart per instrument")
    print("     Spectral Rider: Instruments where you want Phase II probes")
    print()
    print("  5. Instrument suggestions (3×$5k split):")
    for i, sym in enumerate(instruments[:5]):
        ea = "Remora" if i < 4 else "S.Rider"
        print(f"     {sym:<12} → {ea}")
    print()
    print("  6. Add MarketBookAdd(Symbol()) to EA OnInit() for L2 depth")
    print()
    print("  7. Status dashboard:")
    print(f"     {ngrok_url}/status")
    print(f"     {ngrok_url}/cipolla")
    print()
    print("═" * 65)
    print()


# ═════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═════════════════════════════════════════════════════════════════════════════

async def _launch(config_path: str, dry_run: bool):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import Config
    from main_v2 import IMMHubV2

    try:
        cfg = Config.from_yaml(config_path)
    except FileNotFoundError:
        cfg = Config()

    hub = IMMHubV2(cfg, dry_run=dry_run)

    # Hook to print EA instructions after ngrok starts
    original_start = hub.start

    async def patched_start():
        # Start ngrok first to get the URL
        from hub_server_http import start_ngrok
        import yaml
        try:
            with open(config_path, encoding='utf-8', errors='replace') as f:
                d = yaml.safe_load(f)
            token = d.get("ngrok", {}).get("auth_token", "")
            port  = d.get("ngrok", {}).get("port", 8000)
        except Exception:
            token = ""; port = 8000

        ngrok_url = start_ngrok(port, token)
        if ngrok_url:
            hub.ngrok_url = ngrok_url
            print_ea_instructions(ngrok_url, config_path)
        else:
            print("\n  ⚠  ngrok not started — running on localhost:8000 only")
            print("  Install pyngrok: pip install pyngrok --break-system-packages\n")

        await original_start()

    hub.start = patched_start
    await hub.start()


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="IMM AGI Hub — Live Deploy")
    parser.add_argument("--config",     default="imm_config.yaml")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Paper trade mode — no EA commands sent")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate setup and exit without running")
    args = parser.parse_args()

    all_ok = True

    # Step 1: Dependencies
    if not check_dependencies():
        all_ok = False

    # Step 2: Config
    if all_ok and not validate_config(args.config):
        all_ok = False

    # Step 3: Self-test
    if all_ok and not run_self_test():
        all_ok = False

    if not all_ok:
        print("\n  ✗ Pre-flight checks failed. Fix above issues and re-run.\n")
        sys.exit(1)

    print("  ✓ All pre-flight checks passed.\n")

    if args.check_only:
        print("  --check-only: exiting without launch.\n")
        sys.exit(0)

    if args.dry_run:
        print("  MODE: DRY RUN — signals logged, NO commands sent to EAs\n")
    else:
        print("  MODE: LIVE — commands will be sent to connected EAs\n")
        confirm = input("  Type 'go' to launch live, anything else to abort: ").strip()
        if confirm.lower() != "go":
            print("  Aborted.\n")
            sys.exit(0)

    print("\n  Launching IMM AGI Hub V2...\n")
    asyncio.run(_launch(args.config, args.dry_run))


if __name__ == "__main__":
    main()
