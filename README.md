FX Pairs Trading: EUR/USD – GBP/USD

This repository presents a pairs trading backtest on the EUR/USD and GBP/USD FX spot rates using rolling hedge ratios, spread mean-reversion signals, and regime filtering.
The project is designed as a quantitative research study, with a focus on methodology, robustness, and bias control rather than performance optimization.

Overview

Assets: EUR/USD and GBP/USD (daily close prices)

Period: 2020–2024

Strategy class: Statistical arbitrage / pairs trading

Goal: Evaluate whether a mean-reverting spread can be exploited under realistic assumptions

The analysis highlights the regime-dependent nature of FX pairs trading and documents cases where the strategy is inactive or unprofitable.

Methodology
1. Data preprocessing

Daily close prices

Time zones normalized and removed

Alignment on common trading days

2. Hedge ratio estimation

Rolling OLS regression on log-prices

Window length: 120 trading days

Dynamic hedge ratio (beta_t)

3. Spread construction
Spreadt=log⁡(PtEURUSD)−βtlog⁡(PtGBPUSD)
Spread
t
	​

=log(P
t
EURUSD
	​

)−β
t
	​

log(P
t
GBPUSD
	​

)
4. Signal generation

Rolling z-score of the spread

Entry: |z| > 2.5

Exit: z = 0

Stop: |z| > 3.5

5. Regime filter (key component)

Mean-reversion speed estimated via half-life

Trades allowed only when estimated half-life < 60 days

Prevents trading during trend-dominated regimes

6. Backtesting assumptions

Positions executed with one-day lag (no look-ahead bias)

PnL computed on log-returns (consistent with hedge estimation)

Transaction costs applied on both legs (beta-weighted)

Out-of-sample evaluation via time-based split (70% / 30%)

Results Summary

Average rolling return correlation: ~0.7

Estimated half-lives: often > 1 year

Strategy activation: ~25% of the sample

Number of trades: very limited

Out-of-sample trades: none

The results indicate that EUR/USD–GBP/USD does not exhibit sufficiently fast mean reversion over the 2020–2024 period to support a viable daily pairs trading strategy.

Interpretation

Despite high liquidity and economic similarity, the EUR/USD–GBP/USD spread shows slow and unstable mean reversion during the sample period.
This leads to low strategy engagement and weak out-of-sample performance.

Rather than forcing parameter choices to improve returns, the project demonstrates the importance of:

regime awareness,

diagnostic metrics beyond correlation,

and negative results in quantitative research.

Key Takeaways

High return correlation is not sufficient for pairs trading

Mean-reversion speed is a critical diagnostic

Regime filtering can correctly prevent trading when conditions are unfavorable

Not all economically intuitive pairs generate tradable statistical arbitrage opportunities

How to run
pip install -r requirements.txt
python src/long_short_pairs_backtest.py


CSV files must contain at least:

Date, Close

Disclaimer

This project is for educational and research purposes only.
It does not constitute investment advice or a trading recommendation.
