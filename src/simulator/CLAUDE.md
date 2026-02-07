# Simulator Module (Virtual Trading Simulator) Rules

## Role

Receive TradingSignals, execute virtual orders, and manage portfolio state.

## Rules

- All trades execute at signal-time close price + slippage
- Commission rates managed in config/markets.yaml — no hardcoding
- Insufficient cash -> reject order (no margin/leverage)
- Short positions not supported (long only)
- Max position weight: 30% of portfolio
- Min cash reserve: 20% of portfolio
- P&L calculation FX: KRW portfolios in KRW, USD/USDT portfolios in USD
- Portfolio snapshot saved to DB every cycle (for time-series analysis)
- Initial capital: KRX 100M KRW, US $100K, Crypto $100K (configurable)

## Commission Schedule

- KRX: buy 0.015%, sell 0.015% + tax 0.18%
- US: 0% (zero-commission broker assumed)
- Crypto: 0.1% maker/taker (Binance default)
- Default slippage: 0.1% linear
