# ATLAS v4 Mega Upgrade — Design Document

## Overview
Massive expansion: global market coverage, simulation controller, report generation,
custom strategies, GPU-accelerated RL, multi-asset portfolio optimization, SaaS layer,
Oracle Cloud deployment.

## Architecture Changes

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ATLAS v4 Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─── Markets (v4: 10 markets) ──────────────────────────────────┐  │
│  │ KRX │ US │ CRYPTO │ JPX │ SSE │ HKEX │ EURONEXT │ LSE │BOND │  │
│  │                     │ COMMODITIES                              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─── Simulation Controller ─┴──────────────────────────────────┐  │
│  │ • Custom date range (start_date → end_date)                   │  │
│  │ • Infinite mode (continuous until stopped)                    │  │
│  │ • Scenario injection (add events at any timing)               │  │
│  │ • Pause / Resume / Stop controls                              │  │
│  │ • Per-simulation isolated state                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─── Agent Engine ──────────┴──────────────────────────────────┐  │
│  │ 4 LLMs × 2 Archs + Custom Strategy Builder                   │  │
│  │ • User prompt customization via StrategyTemplate              │  │
│  │ • Per-user strategy storage (SaaS)                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─── Portfolio Engine ──────┴──────────────────────────────────┐  │
│  │ • Multi-asset optimizer (correlation-based allocation)        │  │
│  │ • Cross-market hedging signals                                │  │
│  │ • GPU-accelerated RL position sizer (PyTorch)                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─── Report Engine ────────┴───────────────────────────────────┐  │
│  │ • Excel (.xlsx) with charts via openpyxl                      │  │
│  │ • Word (.docx) formatted report via python-docx               │  │
│  │ • PDF summary via reportlab                                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─── SaaS Layer ───────────┴───────────────────────────────────┐  │
│  │ • Multi-tenant: tenant_id on all records                      │  │
│  │ • User auth: JWT + API key per tenant                         │  │
│  │ • Per-user portfolios, strategies, simulation history         │  │
│  │ • Usage metering (API calls, compute time)                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─── Deployment ───────────────────────────────────────────────┐  │
│  │ Oracle Cloud: ARM A1 (4 OCPU, 24 GB) Always Free             │  │
│  │ Docker Compose: app + postgres + redis                        │  │
│  │ Vercel: Next.js frontend (unchanged)                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. Global Market Expansion
**Files**: `src/core/types.py`, `src/core/constants.py`, `config/markets.yaml`
- Add Market enum: JPX, SSE, HKEX, EURONEXT, LSE, BOND, COMMODITIES
- Add data adapters: jpx_adapter, sse_adapter, hkex_adapter, euronext_adapter, lse_adapter, bond_adapter, commodities_adapter
- Add currencies: JPY, CNY, HKD, EUR, GBP
- Add macro adapters: macro_jp, macro_cn, macro_eu

### 2. Simulation Controller
**Files**: `src/simulator/simulation_controller.py`
- SimulationConfig: start_date, end_date (None=infinite), interval, markets, models
- SimulationState: CREATED → RUNNING → PAUSED → COMPLETED / STOPPED
- ScenarioEvent: inject custom events (news, price shocks) at any timestamp
- Full async lifecycle with pause/resume/stop

### 3. Report Engine
**Files**: `src/reports/excel_report.py`, `src/reports/word_report.py`, `src/reports/pdf_report.py`
- ExcelReportGenerator: multi-sheet workbook with charts
- WordReportGenerator: formatted professional report
- PDFReportGenerator: summary with embedded charts

### 4. Custom Strategy System
**Files**: `src/agents/custom_strategy.py`
- StrategyTemplate: user-defined prompt template with variables
- StrategyBuilder: validate + compile custom strategies
- StrategyStore: CRUD for per-user strategies (SaaS-aware)

### 5. GPU-Accelerated RL
**Files**: `src/rl/gpu_position_sizer.py`
- DQN with PyTorch (falls back to CPU if no GPU)
- Experience replay buffer
- Target network with soft updates
- Same interface as RLPositionSizer (drop-in replacement)

### 6. Multi-Asset Portfolio Optimizer
**Files**: `src/simulator/multi_asset_optimizer.py`
- Correlation matrix from cross-market returns
- Mean-Variance Optimization (Markowitz)
- Risk parity allocation
- Black-Litterman with LLM views as inputs

### 7. SaaS Multi-Tenant Layer
**Files**: `src/saas/tenant.py`, `src/saas/auth.py`, `src/saas/usage.py`
- Tenant model with JWT auth
- API key management
- Usage metering and limits
- Per-tenant data isolation

### 8. Oracle Cloud Deployment
**Files**: `deploy/oracle/`, `docker-compose.prod.yml`, `deploy/oracle/setup.sh`
- ARM-optimized Docker images
- Terraform for OCI resources
- Systemd service files
- Auto-SSL with Let's Encrypt
