# IMM AGI Trading Hub

**Information Manifold Model — Structural Detection Engine**
*Travis Bergen | Zenodo: doi.org/10.5281/zenodo.19075097*

---

## What This Is

A Python AGI trading hub that detects Phase III coherence revival events in
the market's informational manifold and fires trade commands to TradeLocker EAs
via an ngrok WebSocket bridge.

**Not a prediction engine. A structural detection engine.**

The market conserves informational energy (N_tot = ψ² + ρ_env). Phase II
compresses visible price action while accumulating background load ρ_env.
When κ·ρ_env > Γ, the revival is forced to complete. We fire there.

---

## File Structure

```
imm_hub/
├── main.py                    ← IMMHub orchestrator (entry point)
├── models.py                  ← All data classes
├── receiver_array.py          ← IMM receiver dynamics per instrument
├── manifold_observer.py       ← Market data + ManifoldState computation
├── phase_detector.py          ← Phase III signal engine
├── exit_engine.py             ← Dynamic phase-aware exit management
├── risk_gate.py               ← AquaFunded DD constraints as θ_trade
├── macro_field.py             ← DXY/SPX/TLT cross-instrument validation
├── hub_server.py              ← FastAPI WebSocket + ngrok bridge
├── IMM_REMORA_EA.mq5          ← MT5/TradeLocker Phase III executor EA
├── IMM_SPECTRAL_RIDER_EA.mq5  ← MT5/TradeLocker Phase II probe EA
├── requirements.txt
└── imm_config.yaml
```

---

## Setup

### 1. Python environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. ngrok
- Sign up at ngrok.com (free tier is fine)
- Copy your auth token into `imm_config.yaml` → `ngrok.auth_token`

### 3. MT5 / TradeLocker EAs
- Copy `IMM_REMORA_EA.mq5` and `IMM_SPECTRAL_RIDER_EA.mq5`
  into your MT5 `MQL5/Experts/` folder
- Compile both in MetaEditor
- Set `HubURL` input to your ngrok URL (printed on hub startup)
- Attach REMORA to whichever chart/symbol you want it to trade
- Attach SPECTRAL_RIDER to charts for Phase II probe instruments

### 4. Configuration
Edit `imm_config.yaml`:
- `account.size`: your actual account equity
- `instruments.tradeable`: symbols your broker offers
  (must match exactly what's in your MT5 symbol list)
- `spectral.phase3_revival_threshold`: 1.15 is conservative, lower to 1.05
  for more signals, raise to 1.30 for highest-conviction only

### 5. Run
```bash
# Full live mode
python main.py --config imm_config.yaml

# Dry run (no EA commands sent, useful for observation)
python main.py --dry-run
```

---

## How Signals Flow

```
yfinance (macro: DXY, SPX, TLT)
        +
EA tick feeds (EURUSD, GBPUSD, ...)
        ↓
ManifoldObserver.update_all()
  → per-instrument: spectral gap Δ, entropy H, ρ_env, ψ², phase
        ↓
ReceiverArray.step()
  → R adapts under V(R,E) gradient flow
        ↓
MacroField.update() + check()
  → validates signal against DXY/SPX/TLT phase consistency
        ↓
PhaseIIIDetector.update()
  → fires TradeSignal when κ·ρ_env > Γ × threshold
        ↓
RiskGate checks:
  → S(R,E) > θ_trade  (stability threshold)
  → correlated_exposure_ok()
  → position_size()
        ↓
HubServer.send_to_ea("remora", ENTER_cmd)
        ↓
ExitEngine.update() each cycle
  → trails stop as ψ²/N_tot depletes
  → detects nested Phase III for extended runners
  → closes when coherence basis gone
```

---

## AquaFunded Constraints

Embedded in `RiskGate`:
- Daily DD hard limit: 5% ($500 on $10k)
- Max DD hard limit: 10% ($1000 on $10k)
- Trading blocked at 85% of daily limit used
- Emergency close at 95% of daily limit
- All limits scale automatically with `account.size`

---

## Scaling

To scale to $100k, $1M, or more:
1. Change `account.size` in `imm_config.yaml`
2. Increase `risk.max_concurrent_trades` (8-12 for $1M+)
3. Everything else scales automatically
4. For $10M+: run multiple hub instances with separate instrument groups

---

## Live Data Notes

Macro anchors (DXY, SPX, TLT) use `yfinance` with 5-minute bars.
This introduces a small latency — acceptable for the manifold observer
which operates on structural timescales, not tick scalping.

For lower latency macro feeds:
- Replace `_yf_fetch()` in `manifold_observer.py` with your broker's API
- The `inject_ea_tick()` method accepts real-time price updates from EAs

For full tick-level precision on tradeable pairs:
- EAs already push tick updates via WebSocket every `OnTick()`
- These immediately update the rolling buffers in ManifoldObserver

---

## Tuning Parameters

| Parameter | File | Effect |
|-----------|------|--------|
| `phase3_revival_threshold` | config | Higher = fewer, stronger signals |
| `phase2_min_bars` | config | Min compression bars before Phase III |
| `THETA_BASE` | risk_gate.py | Baseline stability requirement |
| `REVIVAL_FADING_FRAC` | exit_engine.py | When to start tightening trail |
| `NESTED_WAIT_BARS` | exit_engine.py | How long to wait for nested cycle |
| `PASS_THRESHOLD` | macro_field.py | Macro consistency bar |

---

*Built on IMM V3 — doi.org/10.5281/zenodo.19075097*
