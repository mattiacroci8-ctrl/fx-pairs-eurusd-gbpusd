#FX Pairs Trading: EUR/USD – GBP/USD

This repository contains a pairs trading backtest on the EUR/USD and GBP/USD FX spot rates.
The project is presented as a quantitative research study, focusing on methodology, robustness, and regime dependence rather than performance optimization.

Overview

Assets: EUR/USD and GBP/USD (daily close prices)

Period: 2020–2024

Strategy type: Statistical arbitrage / pairs trading

Objective: assess whether a mean-reverting spread can be exploited under realistic assumptions

Methodology
Data

Daily close prices

Time zones normalized and removed

Series aligned on common trading days

Hedge ratio

Rolling OLS regression on log-prices

Window length: 120 trading days

Time-varying hedge ratio (beta)

Spread

Constructed as log(EUR/USD) minus beta times log(GBP/USD)

Signals

Rolling z-score of the spread

Entry when absolute z-score exceeds 2.5

Exit at mean reversion (z = 0)

Stop-loss when absolute z-score exceeds 3.5

Regime filter

Mean-reversion speed estimated via half-life

Trades allowed only when estimated half-life is below 60 days

Prevents trading during trend-dominated regimes

Backtest assumptions

One-day execution lag (no look-ahead bias)

PnL computed on log-returns (consistent with hedge estimation)

Transaction costs applied on both legs (beta-weighted)

Time-based out-of-sample split (70% / 30%)

Results Summary

Average rolling return correlation: approximately 0.7

Estimated half-lives often exceed one year

Strategy activation around 25% of the sample

Very limited number of trades

No trades generated in the out-of-sample period

These results indicate that the EUR/USD–GBP/USD spread does not exhibit sufficiently fast or stable mean reversion over the 2020–2024 period.

Interpretation

Despite high liquidity and economic similarity, the EUR/USD–GBP/USD pair shows slow and unstable mean-reverting behavior in recent years.
The regime filter correctly prevents trading in unfavorable conditions, leading to low exposure and weak out-of-sample performance.

This highlights the regime-dependent nature of FX pairs trading and the importance of diagnostic metrics beyond correlation.

Key Takeaways

High return correlation alone is not sufficient for pairs trading

Mean-reversion speed is a critical diagnostic

Regime filters help avoid structurally unfavorable periods

Negative results are informative in quantitative research

How to run
pip install -r requirements.txt
python src/long_short_pairs_backtest.py


CSV files must include at least the following columns:

Date, Close

Disclaimer

This project is for educational and research purposes only.
It does not constitute investment advice.
