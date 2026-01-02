import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

TRANSACTION_COST = 0.0005
WINDOW = 120
ENTRY_Z = 2.5
EXIT_Z = 0.0
STOP_Z = 3.5
ANNUAL_FACTOR = 252

HALF_LIFE_WINDOW = 252
HALF_LIFE_MAX = 60


def load_data(file1, file2):
    df1 = pd.read_csv(file1, usecols=["Date", "Close"])
    df2 = pd.read_csv(file2, usecols=["Date", "Close"])

    df1["Date"] = pd.to_datetime(df1["Date"], dayfirst=True, utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    df2["Date"] = pd.to_datetime(df2["Date"], dayfirst=True, utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()

    df1 = df1.dropna(subset=["Date"]).rename(columns={"Close": "Price1"})
    df2 = df2.dropna(subset=["Date"]).rename(columns={"Close": "Price2"})

    df = (
        pd.merge(df1, df2, on="Date", how="inner")
        .sort_values("Date")
        .set_index("Date")
    )

    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="first")]

    df["LogP1"] = np.log(df["Price1"].astype(float))
    df["LogP2"] = np.log(df["Price2"].astype(float))
    return df


def calculate_spread_zscore(df, window=WINDOW):
    betas = np.full(len(df), np.nan)

    for i in range(window, len(df)):
        y = df["LogP1"].iloc[i - window:i]
        x = add_constant(df["LogP2"].iloc[i - window:i])
        betas[i] = OLS(y, x).fit().params.iloc[1]

    df["Beta"] = betas
    df["Spread"] = df["LogP1"] - df["Beta"] * df["LogP2"]

    mean = df["Spread"].rolling(window).mean()
    std = df["Spread"].rolling(window).std().replace(0, np.nan)
    df["ZScore"] = (df["Spread"] - mean) / std
    return df


def estimate_half_life(spread_window: pd.Series) -> float:
    s = spread_window.dropna()
    if len(s) < 20:
        return np.nan

    lag = s.shift(1).iloc[1:]
    delta = s.diff().iloc[1:]
    if lag.std() == 0:
        return np.nan

    b = np.polyfit(lag.values, delta.values, 1)[0]
    if b >= 0:
        return np.inf

    hl = -np.log(2) / b
    return float(hl)


def add_half_life_filter(df, hl_window=HALF_LIFE_WINDOW, hl_max=HALF_LIFE_MAX):
    hl = df["Spread"].rolling(hl_window).apply(estimate_half_life, raw=False)
    df["HalfLife"] = hl
    df["RegimeOK"] = (df["HalfLife"] > 0) & (df["HalfLife"] < hl_max)
    return df


def generate_signals(df):
    pos = 0
    out = []

    for z, ok in zip(df["ZScore"], df["RegimeOK"]):
        if (not ok) or np.isnan(z):
            pos = 0
        elif pos == 0:
            pos = -1 if z > ENTRY_Z else (1 if z < -ENTRY_Z else 0)
        elif pos == 1:
            if z >= EXIT_Z or z < -STOP_Z:
                pos = 0
        else:
            if z <= EXIT_Z or z > STOP_Z:
                pos = 0

        out.append(pos)

    df["Position"] = out
    return df


def backtest_strategy(df, cost=TRANSACTION_COST):
    df = df.copy()

    df["LR1"] = df["LogP1"].diff()
    df["LR2"] = df["LogP2"].diff()

    pos = df["Position"].shift(1).fillna(0.0)
    beta = df["Beta"].shift(1).fillna(0.0)

    pos_change = (pos - pos.shift(1).fillna(0.0)).abs()

    raw = pos * (df["LR1"] - beta * df["LR2"])

    traded_notional = pos_change * (1.0 + beta.abs())
    costs = cost * traded_notional

    df["Strategy_Returns"] = (raw - costs).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["Equity"] = (1.0 + df["Strategy_Returns"]).cumprod()

    executed_pos = pos.astype(int)
    prev_exe = executed_pos.shift(1).fillna(0).astype(int)
    df["Trade_Entry"] = ((prev_exe == 0) & (executed_pos != 0)).astype(int)

    df["Exposure"] = executed_pos.abs()
    df["Turnover"] = traded_notional.fillna(0.0)

    trade_returns, trade_durations = [], []
    in_trade, cum_trade, dur = False, 1.0, 0

    for i in range(len(df)):
        p = executed_pos.iloc[i]
        r = df["Strategy_Returns"].iloc[i]

        if not in_trade and p != 0:
            in_trade, cum_trade, dur = True, 1.0, 0

        if in_trade:
            cum_trade *= (1.0 + r)
            dur += 1
            if p == 0:
                trade_returns.append(cum_trade - 1.0)
                trade_durations.append(dur)
                in_trade = False

    if in_trade:
        trade_returns.append(cum_trade - 1.0)
        trade_durations.append(dur)

    return df, trade_returns, trade_durations


def calculate_metrics(df, trade_returns, trade_durations):
    rets = df["Strategy_Returns"]
    eq = df["Equity"]

    vol = rets.std(ddof=0)
    sharpe = np.sqrt(ANNUAL_FACTOR) * (rets.mean() / vol) if vol > 0 else np.nan

    downside = rets[rets < 0]
    down_vol = downside.std(ddof=0)
    sortino = np.sqrt(ANNUAL_FACTOR) * (rets.mean() / down_vol) if down_vol > 0 else np.nan

    total_days = max((df.index[-1] - df.index[0]).days, 1)
    cagr = (eq.iloc[-1]) ** (365.0 / total_days) - 1.0

    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    max_dd = dd.min()

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else np.nan

    tr = np.array(trade_returns, dtype=float) if trade_returns else np.array([])
    hit_rate = float((tr > 0).mean()) if tr.size else np.nan
    avg_win = float(tr[tr > 0].mean()) if (tr > 0).any() else np.nan
    avg_loss = float(tr[tr < 0].mean()) if (tr < 0).any() else np.nan
    profit_factor = (
        float(tr[tr > 0].sum() / abs(tr[tr < 0].sum()))
        if (tr > 0).any() and (tr < 0).any()
        else np.nan
    )

    num_trades = int(df["Trade_Entry"].sum())
    avg_dur = float(np.mean(trade_durations)) if trade_durations else 0.0

    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "CAGR": cagr,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Total Trades": num_trades,
        "Hit Rate": hit_rate,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Profit Factor": profit_factor,
        "Avg Trade Duration (days)": avg_dur,
        "Avg Exposure": float(df["Exposure"].mean()),
        "Avg Turnover (proxy)": float(df["Turnover"].mean()),
        "Avg HalfLife (days)": float(df["HalfLife"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if "HalfLife" in df else np.nan,
        "RegimeOK %": float(df["RegimeOK"].mean()) if "RegimeOK" in df else np.nan,
    }


def plot_results(df, trade_returns):
    eq = df["Equity"]
    dd = (eq - eq.cummax()) / eq.cummax()

    plt.figure(figsize=(12, 5))
    plt.plot(df.index, eq, label="Equity")
    plt.title("Equity Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, dd, label="Drawdown")
    plt.title("Drawdown")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if trade_returns:
        plt.figure(figsize=(10, 4))
        plt.hist(trade_returns, bins=30)
        plt.title("Trade Return Distribution")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def split_train_test(df, split_ratio=0.7):
    cut = int(len(df) * split_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


if __name__ == "__main__":
    file1 = "data/eurusd.csv"
    file2 = "data/gbpusd.csv"

    df = load_data(file1, file2)

    r1 = df["LogP1"].diff()
    r2 = df["LogP2"].diff()
    rolling_corr = r1.rolling(WINDOW).corr(r2)
    print(f"Average rolling return correlation ({WINDOW}d): {rolling_corr.mean():.2f}")

    train_raw, test_raw = split_train_test(df, split_ratio=0.7)

    def run_pipeline(d):
        d = calculate_spread_zscore(d, window=WINDOW)
        d = add_half_life_filter(d, hl_window=HALF_LIFE_WINDOW, hl_max=HALF_LIFE_MAX)
        d = generate_signals(d)
        d = d.dropna(subset=["Beta", "ZScore", "HalfLife"]).copy()
        bt, tr, td = backtest_strategy(d, cost=TRANSACTION_COST)
        m = calculate_metrics(bt, tr, td)
        return bt, tr, td, m

    bt_train, tr_train, td_train, m_train = run_pipeline(train_raw.copy())
    bt_test, tr_test, td_test, m_test = run_pipeline(test_raw.copy())

    bt_full = pd.concat([bt_train, bt_test]).sort_index()
    trade_returns_full = tr_train + tr_test
    trade_durations_full = td_train + td_test
    m_full = calculate_metrics(bt_full, trade_returns_full, trade_durations_full)

    print("\nBacktest metrics (full sample):")
    for k, v in m_full.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nTrain metrics (first 70%):")
    for k, v in m_train.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nTest metrics (last 30% OOS):")
    for k, v in m_test.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    plot_results(bt_full, trade_returns_full)
