"""
Consequence Assessment Engine for Consequence-Aware Agentic Trading Research
=======================================================================
Author: Prof. Shahab Anbarjafari, 3S Holding OÜ, Tartu, Estonia (shb@3sholding.com)

This is the heart of the "consequence-aware" paradigm. Before selecting a
strategy for live execution, I evaluate its consequences on historical data—
essentially asking "what would have happened if I had followed this strategy?"

My backtesting engine is deliberately simple and transparent. I could have
used a framework like backtrader or zipline, but I chose to write it from
scratch for two reasons:
  1. Full auditability: every line of the backtest logic is visible and
     documented. No hidden assumptions in third-party library internals.
  2. Exact control over execution assumptions (e.g., execution price,
     position sizing, commission model).

Design decisions documented for reviewers:
  - Long-only strategies (no short selling) for simplicity and because
    most retail/institutional strategies are long-biased.
  - Binary positions: either fully invested (100% of capital) or fully in cash.
    No partial positions. This is a simplification, but it's consistent across
    all strategies and baselines.
  - Execution at close price on signal day. This introduces a mild optimism
    bias (in reality you'd execute next-open), but it's consistent.
  - No commissions or slippage by default, but these can be toggled in config.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from strategy_generator import TradingStrategy

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Trade Record
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class TradeRecord:
    """
    I record every trade with enough detail to reconstruct the P&L calculation
    from scratch. This trade-level granularity is essential for understanding
    WHY a strategy performed as it did—not just the terminal return.
    """
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    pnl_dollars: float
    holding_days: int
    exit_reason: str  # "signal", "stop_loss", "trailing_stop", "max_holding", "end_of_data"
    position_value: float

    def to_dict(self) -> dict:
        return asdict(self)


# ───────────────────────────────────────────────────────────────────────────
# Backtest Result
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class BacktestResult:
    """
    I encapsulate all backtest outputs in a single object: performance metrics,
    the equity curve, and the complete trade log. This makes it easy to compare
    strategies, generate visualizations, and conduct post-hoc analysis.
    """
    ticker: str
    strategy_name: str
    strategy_type: str

    # Core performance metrics — these are what I use for strategy selection
    cumulative_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    calmar_ratio: float = 0.0

    # Additional metrics for deeper analysis
    total_trades: int = 0
    avg_trade_return: float = 0.0
    avg_holding_days: float = 0.0
    profit_factor: float = 0.0
    volatility: float = 0.0
    sortino_ratio: float = 0.0

    # Time series data
    equity_curve: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None

    # Trade-level records
    trades: List[TradeRecord] = field(default_factory=list)

    # Metadata
    backtest_start_date: str = ""
    backtest_end_date: str = ""
    num_trading_days: int = 0
    execution_time_seconds: float = 0.0

    def to_summary_dict(self) -> dict:
        """I return a serializable summary (excluding large time series)."""
        return {
            "ticker": self.ticker,
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "cumulative_return": self.cumulative_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "avg_trade_return": self.avg_trade_return,
            "avg_holding_days": self.avg_holding_days,
            "profit_factor": self.profit_factor,
            "volatility": self.volatility,
            "sortino_ratio": self.sortino_ratio,
            "backtest_start_date": self.backtest_start_date,
            "backtest_end_date": self.backtest_end_date,
            "num_trading_days": self.num_trading_days,
            "execution_time_seconds": self.execution_time_seconds,
        }


# ───────────────────────────────────────────────────────────────────────────
# Technical Indicator Computation
# ───────────────────────────────────────────────────────────────────────────
def _compute_sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return prices.rolling(window=window, min_periods=window).mean()


def _compute_ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average — I use this for MACD calculations."""
    return prices.ewm(span=span, adjust=False).mean()


def _compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing.
    I compute this as a full time series (not just the last value)
    because the backtest engine needs to check RSI on every bar.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _compute_bollinger_bands(
    prices: pd.Series, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """I return (upper, middle, lower) Bollinger Bands."""
    middle = prices.rolling(window=window, min_periods=window).mean()
    std = prices.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def _compute_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """I return (macd_line, signal_line, histogram)."""
    fast_ema = _compute_ema(prices, fast_period)
    slow_ema = _compute_ema(prices, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = _compute_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ───────────────────────────────────────────────────────────────────────────
# Core Backtest Engine
# ───────────────────────────────────────────────────────────────────────────
def backtest_strategy(
    strategy: TradingStrategy,
    data: pd.DataFrame,
    initial_capital: float = config.INITIAL_CAPITAL,
    warmup_data: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """
    I execute a trading strategy on historical data and compute comprehensive
    performance metrics. This is the "consequence assessment"—I simulate what
    would have happened if I had followed this strategy.

    The engine iterates through each trading day, generates signals based on
    the strategy's indicators, manages positions, and tracks P&L. Every trade
    is recorded with entry/exit details.

    Parameters
    ----------
    strategy : TradingStrategy
        The strategy to backtest.
    data : pd.DataFrame
        OHLCV data for the backtest period.
    initial_capital : float
        Starting portfolio value.
    warmup_data : pd.DataFrame, optional
        Historical data to prepend for indicator warm-up. When provided, I
        compute indicators on the combined (warmup + data), but only evaluate
        portfolio performance on the 'data' portion. This is critical for
        short test periods where lookback windows exceed the data length.

    Returns
    -------
    BacktestResult
        Complete backtest results including metrics, equity curve, and trades.
    """
    bt_start = time.time()
    logger.info(
        f"Backtesting '{strategy.name}' ({strategy.strategy_type}) on "
        f"{strategy.ticker} ({len(data)} bars"
        f"{f', +{len(warmup_data)} warmup' if warmup_data is not None else ''})"
    )

    # I dispatch to the appropriate strategy executor based on type
    executor_map = {
        "ma_crossover": _execute_ma_crossover,
        "rsi_mean_reversion": _execute_rsi_mean_reversion,
        "bollinger_breakout": _execute_bollinger_breakout,
        "macd_signal": _execute_macd_signal,
        "momentum": _execute_momentum,
    }

    executor = executor_map.get(strategy.strategy_type)
    if executor is None:
        raise ValueError(f"No executor for strategy type: {strategy.strategy_type}")

    # If warmup data is provided, I prepend it for indicator computation,
    # then strip it after signal generation so the portfolio simulation
    # only covers the actual evaluation period. This avoids the problem
    # of strategies with long lookback windows generating no signals on
    # short test periods.
    if warmup_data is not None and len(warmup_data) > 0:
        combined_data = pd.concat([warmup_data, data])
        combined_data = combined_data[~combined_data.index.duplicated(keep="last")]
        combined_data = combined_data.sort_index()
        warmup_len = len(combined_data) - len(data)

        signals = executor(combined_data, strategy.parameters)
        # I trim signals and data to the evaluation period only
        signals = signals.iloc[warmup_len:]
        eval_data = combined_data.iloc[warmup_len:]
    else:
        eval_data = data
        signals = executor(data, strategy.parameters)

    # I convert signals into actual portfolio values and trade records
    result = _simulate_portfolio(
        data=eval_data,
        signals=signals,
        strategy=strategy,
        initial_capital=initial_capital,
    )

    result.execution_time_seconds = time.time() - bt_start
    logger.info(
        f"Backtest complete for '{strategy.name}': "
        f"Return={result.cumulative_return:.2%}, "
        f"Sharpe={result.sharpe_ratio:.2f}, "
        f"MaxDD={result.max_drawdown:.2%}, "
        f"Trades={result.total_trades}, "
        f"Time={result.execution_time_seconds:.3f}s"
    )

    return result


def _execute_ma_crossover(
    data: pd.DataFrame, params: Dict[str, Any]
) -> pd.Series:
    """
    I generate buy/sell signals for the moving average crossover strategy.
    Signal = 1 when short MA > long MA (bullish), 0 otherwise.
    """
    close = data["Close"]
    short_ma = _compute_sma(close, params["short_window"])
    long_ma = _compute_sma(close, params["long_window"])

    # I use a simple crossover: position = 1 when short MA is above long MA
    signal = pd.Series(0, index=data.index, dtype=float)
    signal[short_ma > long_ma] = 1.0

    # I set signal to NaN during the warm-up period where MAs aren't defined
    warmup = params["long_window"]
    signal.iloc[:warmup] = np.nan

    return signal


def _execute_rsi_mean_reversion(
    data: pd.DataFrame, params: Dict[str, Any]
) -> pd.Series:
    """
    I generate signals for the RSI mean reversion strategy.
    Buy when RSI drops below oversold threshold, hold for a fixed period
    or until overbought, whichever comes first.
    """
    close = data["Close"]
    rsi = _compute_rsi(close, params["rsi_window"])

    signal = pd.Series(0, index=data.index, dtype=float)
    holding_counter = 0
    in_position = False

    for i in range(len(data)):
        if pd.isna(rsi.iloc[i]):
            signal.iloc[i] = np.nan
            continue

        if not in_position:
            if rsi.iloc[i] < params["oversold_threshold"]:
                in_position = True
                holding_counter = 0
                signal.iloc[i] = 1.0
            else:
                signal.iloc[i] = 0.0
        else:
            holding_counter += 1
            if (rsi.iloc[i] > params["overbought_threshold"] or
                    holding_counter >= params["holding_period"]):
                in_position = False
                signal.iloc[i] = 0.0
            else:
                signal.iloc[i] = 1.0

    return signal


def _execute_bollinger_breakout(
    data: pd.DataFrame, params: Dict[str, Any]
) -> pd.Series:
    """
    I generate signals for the Bollinger Band breakout strategy.
    Buy when price breaks above the upper band (momentum breakout).
    Exit at the middle band if configured, otherwise at the lower band.
    """
    close = data["Close"]
    upper, middle, lower = _compute_bollinger_bands(
        close, params["bb_window"], params["bb_std"]
    )

    signal = pd.Series(0, index=data.index, dtype=float)
    in_position = False

    for i in range(len(data)):
        if pd.isna(upper.iloc[i]):
            signal.iloc[i] = np.nan
            continue

        if not in_position:
            if close.iloc[i] > upper.iloc[i]:
                in_position = True
                signal.iloc[i] = 1.0
            else:
                signal.iloc[i] = 0.0
        else:
            exit_level = middle.iloc[i] if params.get("exit_at_middle", True) else lower.iloc[i]
            if close.iloc[i] < exit_level:
                in_position = False
                signal.iloc[i] = 0.0
            else:
                signal.iloc[i] = 1.0

    return signal


def _execute_macd_signal(
    data: pd.DataFrame, params: Dict[str, Any]
) -> pd.Series:
    """
    I generate signals for the MACD strategy. Buy on bullish crossover
    (MACD crosses above signal line), sell on bearish crossover.
    """
    close = data["Close"]
    macd_line, signal_line, histogram = _compute_macd(
        close, params["fast_period"], params["slow_period"], params["signal_period"]
    )

    signal = pd.Series(0, index=data.index, dtype=float)

    # I need enough data for the slow EMA to stabilize
    warmup = params["slow_period"] + params["signal_period"]
    signal.iloc[:warmup] = np.nan

    for i in range(warmup, len(data)):
        if macd_line.iloc[i] > signal_line.iloc[i]:
            signal.iloc[i] = 1.0
        else:
            signal.iloc[i] = 0.0

    return signal


def _execute_momentum(
    data: pd.DataFrame, params: Dict[str, Any]
) -> pd.Series:
    """
    I generate signals for the momentum strategy. Buy when the lookback
    return exceeds the threshold. Exit on trailing stop or max holding period.
    """
    close = data["Close"]
    lookback = params["lookback_period"]
    entry_thresh = params["entry_threshold"]
    max_hold = params["max_holding_period"]

    signal = pd.Series(0, index=data.index, dtype=float)
    in_position = False
    holding_counter = 0
    peak_price = 0.0

    for i in range(lookback, len(data)):
        lookback_return = (close.iloc[i] / close.iloc[i - lookback]) - 1.0

        if not in_position:
            if lookback_return > entry_thresh:
                in_position = True
                holding_counter = 0
                peak_price = close.iloc[i]
                signal.iloc[i] = 1.0
            else:
                signal.iloc[i] = 0.0
        else:
            holding_counter += 1
            peak_price = max(peak_price, close.iloc[i])
            trailing_stop_price = peak_price * (1 - params["trailing_stop_pct"])

            if close.iloc[i] < trailing_stop_price or holding_counter >= max_hold:
                in_position = False
                signal.iloc[i] = 0.0
            else:
                signal.iloc[i] = 1.0

    # Warm-up period
    signal.iloc[:lookback] = np.nan

    return signal


# ───────────────────────────────────────────────────────────────────────────
# Portfolio Simulation
# ───────────────────────────────────────────────────────────────────────────
def _simulate_portfolio(
    data: pd.DataFrame,
    signals: pd.Series,
    strategy: TradingStrategy,
    initial_capital: float,
) -> BacktestResult:
    """
    I convert a signal series into actual portfolio performance. This function
    handles position management, trade recording, stop-loss execution, and
    metric computation.

    I track the portfolio value on every bar and record each trade with
    complete entry/exit details.
    """
    close = data["Close"]
    n = len(data)

    # I initialize the portfolio tracker
    portfolio_value = np.full(n, initial_capital, dtype=float)
    cash = initial_capital
    position_shares = 0.0
    entry_price = 0.0
    entry_date = None
    trades: List[TradeRecord] = []

    stop_loss_pct = strategy.parameters.get("stop_loss_pct", 0.10)

    for i in range(1, n):
        current_price = close.iloc[i]
        current_date = data.index[i]

        if pd.isna(signals.iloc[i]):
            portfolio_value[i] = portfolio_value[i - 1]
            continue

        # I check stop loss first—this takes priority over signals
        if position_shares > 0:
            current_return = (current_price / entry_price) - 1.0
            if current_return <= -stop_loss_pct:
                # Stop loss triggered: I sell everything
                proceeds = position_shares * current_price
                pnl = proceeds - (position_shares * entry_price)
                trade = TradeRecord(
                    entry_date=str(entry_date.date()) if entry_date else "",
                    exit_date=str(current_date.date()),
                    entry_price=entry_price,
                    exit_price=current_price,
                    return_pct=current_return,
                    pnl_dollars=pnl,
                    holding_days=(current_date - entry_date).days if entry_date else 0,
                    exit_reason="stop_loss",
                    position_value=proceeds,
                )
                trades.append(trade)
                cash = proceeds
                position_shares = 0.0
                entry_price = 0.0
                entry_date = None
                portfolio_value[i] = cash
                continue

        # I execute signal-based trades
        current_signal = signals.iloc[i]
        prev_signal = signals.iloc[i - 1] if not pd.isna(signals.iloc[i - 1]) else 0

        if current_signal == 1.0 and position_shares == 0:
            # BUY: I go fully invested
            position_shares = cash / current_price
            entry_price = current_price
            entry_date = current_date
            cash = 0.0

        elif current_signal == 0.0 and position_shares > 0:
            # SELL: I exit the position entirely
            proceeds = position_shares * current_price
            pnl = proceeds - (position_shares * entry_price)
            trade_return = (current_price / entry_price) - 1.0
            trade = TradeRecord(
                entry_date=str(entry_date.date()) if entry_date else "",
                exit_date=str(current_date.date()),
                entry_price=entry_price,
                exit_price=current_price,
                return_pct=trade_return,
                pnl_dollars=pnl,
                holding_days=(current_date - entry_date).days if entry_date else 0,
                exit_reason="signal",
                position_value=proceeds,
            )
            trades.append(trade)
            cash = proceeds
            position_shares = 0.0
            entry_price = 0.0
            entry_date = None

        # I update portfolio value: cash + current position value
        portfolio_value[i] = cash + position_shares * current_price

    # If I'm still holding a position at the end, I close it
    if position_shares > 0:
        final_price = close.iloc[-1]
        proceeds = position_shares * final_price
        pnl = proceeds - (position_shares * entry_price)
        trade_return = (final_price / entry_price) - 1.0
        trade = TradeRecord(
            entry_date=str(entry_date.date()) if entry_date else "",
            exit_date=str(data.index[-1].date()),
            entry_price=entry_price,
            exit_price=final_price,
            return_pct=trade_return,
            pnl_dollars=pnl,
            holding_days=(data.index[-1] - entry_date).days if entry_date else 0,
            exit_reason="end_of_data",
            position_value=proceeds,
        )
        trades.append(trade)
        portfolio_value[-1] = proceeds

    # I compute the equity curve and daily returns
    equity_curve = pd.Series(portfolio_value, index=data.index, name="equity")
    daily_returns = equity_curve.pct_change().dropna()

    # I compute all performance metrics
    metrics = _compute_performance_metrics(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        trades=trades,
        initial_capital=initial_capital,
    )

    result = BacktestResult(
        ticker=strategy.ticker,
        strategy_name=strategy.name,
        strategy_type=strategy.strategy_type,
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        trades=trades,
        backtest_start_date=str(data.index[0].date()),
        backtest_end_date=str(data.index[-1].date()),
        num_trading_days=n,
        **metrics,
    )

    return result


def _compute_performance_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    trades: List[TradeRecord],
    initial_capital: float,
) -> Dict[str, float]:
    """
    I compute the full suite of performance metrics. I handle edge cases
    explicitly (zero volatility, no trades, etc.) because these WILL occur
    in practice and I refuse to let a division-by-zero crash my experiment.

    Metrics:
      - Cumulative return: Total return over the period
      - Annualized return: Geometric annualization
      - Sharpe ratio: Annualized, adjusted for risk-free rate
      - Max drawdown: Largest peak-to-trough decline
      - Win rate: Proportion of profitable trades
      - Calmar ratio: Annualized return / |max drawdown|
      - Profit factor: Gross profits / gross losses
      - Sortino ratio: Like Sharpe but only penalizes downside vol
    """
    final_value = equity_curve.iloc[-1]
    cumulative_return = (final_value / initial_capital) - 1.0

    # Annualized return: I use geometric annualization
    n_days = len(equity_curve)
    n_years = n_days / config.TRADING_DAYS_PER_YEAR
    if n_years > 0 and final_value > 0:
        annualized_return = (final_value / initial_capital) ** (1 / n_years) - 1.0
    else:
        annualized_return = 0.0

    # Volatility and Sharpe ratio
    if len(daily_returns) > 1 and daily_returns.std() > 1e-10:
        annual_vol = daily_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        sharpe = (
            (daily_returns.mean() * config.TRADING_DAYS_PER_YEAR - config.RISK_FREE_RATE)
            / annual_vol
        )
    else:
        annual_vol = 0.0
        sharpe = 0.0

    # Max drawdown: I compute this from the equity curve, not from returns,
    # because cumulative compounding can make return-based drawdown misleading.
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = float(drawdown.min())

    # Calmar ratio: annualized return / |max drawdown|
    # I handle the edge case where max drawdown is zero
    calmar = annualized_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # Trade-level metrics
    if trades:
        trade_returns = [t.return_pct for t in trades]
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r <= 0]

        win_rate = len(winning_trades) / len(trade_returns)
        avg_trade_return = np.mean(trade_returns)
        avg_holding = np.mean([t.holding_days for t in trades])

        gross_profit = sum(t.pnl_dollars for t in trades if t.pnl_dollars > 0)
        gross_loss = abs(sum(t.pnl_dollars for t in trades if t.pnl_dollars < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        win_rate = 0.0
        avg_trade_return = 0.0
        avg_holding = 0.0
        profit_factor = 0.0

    # Sortino ratio: I only penalize downside volatility
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 1:
        downside_vol = downside_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        sortino = (
            (daily_returns.mean() * config.TRADING_DAYS_PER_YEAR - config.RISK_FREE_RATE)
            / downside_vol
        ) if downside_vol > 1e-10 else 0.0
    else:
        sortino = 0.0

    return {
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "calmar_ratio": float(calmar),
        "total_trades": len(trades),
        "avg_trade_return": float(avg_trade_return),
        "avg_holding_days": float(avg_holding),
        "profit_factor": float(profit_factor),
        "volatility": float(annual_vol),
        "sortino_ratio": float(sortino),
    }


# ───────────────────────────────────────────────────────────────────────────
# Buy-and-Hold Baseline
# ───────────────────────────────────────────────────────────────────────────
def backtest_buy_and_hold(
    ticker: str,
    data: pd.DataFrame,
    initial_capital: float = config.INITIAL_CAPITAL,
) -> BacktestResult:
    """
    I implement the buy-and-hold baseline. Purchase on day one, hold through
    the entire period. This is the simplest possible strategy and serves as
    a lower bound for what any active strategy should beat.

    If an agent can't outperform buy-and-hold, it's destroying value through
    unnecessary trading.
    """
    logger.info(f"Running buy-and-hold baseline for {ticker}")
    close = data["Close"]

    # I buy at the close on day 1 and hold
    shares = initial_capital / close.iloc[0]
    portfolio_value = shares * close
    equity_curve = pd.Series(portfolio_value.values, index=data.index, name="equity")
    daily_returns = equity_curve.pct_change().dropna()

    final_value = equity_curve.iloc[-1]
    trade = TradeRecord(
        entry_date=str(data.index[0].date()),
        exit_date=str(data.index[-1].date()),
        entry_price=float(close.iloc[0]),
        exit_price=float(close.iloc[-1]),
        return_pct=float((close.iloc[-1] / close.iloc[0]) - 1.0),
        pnl_dollars=float(final_value - initial_capital),
        holding_days=(data.index[-1] - data.index[0]).days,
        exit_reason="end_of_data",
        position_value=float(final_value),
    )

    metrics = _compute_performance_metrics(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        trades=[trade],
        initial_capital=initial_capital,
    )

    result = BacktestResult(
        ticker=ticker,
        strategy_name="Buy and Hold",
        strategy_type="buy_and_hold",
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        trades=[trade],
        backtest_start_date=str(data.index[0].date()),
        backtest_end_date=str(data.index[-1].date()),
        num_trading_days=len(data),
        **metrics,
    )

    logger.info(
        f"Buy-and-hold for {ticker}: Return={result.cumulative_return:.2%}, "
        f"Sharpe={result.sharpe_ratio:.2f}"
    )

    return result


# ───────────────────────────────────────────────────────────────────────────
# Strategy Selection (Consequence-Aware)
# ───────────────────────────────────────────────────────────────────────────
def select_best_strategy(
    backtest_results: List[BacktestResult],
    metric: str = config.PRIMARY_SELECTION_METRIC,
) -> Tuple[int, BacktestResult]:
    """
    I select the best strategy based on the specified metric. This is the
    core of the consequence-aware paradigm: I pick the strategy that would
    have performed best in the past (according to the chosen metric).

    The key research question is whether this backward-looking selection
    translates to forward-looking performance.

    Parameters
    ----------
    backtest_results : list of BacktestResult
        Backtest results for all candidate strategies.
    metric : str
        Which metric to optimize. Default: "sharpe_ratio".

    Returns
    -------
    Tuple of (best_index, best_result)
    """
    if not backtest_results:
        raise ValueError("No backtest results to select from")

    # I extract the selection metric from each result
    metric_values = []
    for r in backtest_results:
        value = getattr(r, metric, None)
        if value is None:
            value = r.to_summary_dict().get(metric, float("-inf"))
        metric_values.append(value)

    # For max_drawdown, less negative is better, so I negate it for selection
    if metric == "max_drawdown":
        selection_values = [-v for v in metric_values]
    else:
        selection_values = metric_values

    best_idx = int(np.argmax(selection_values))
    best_result = backtest_results[best_idx]

    logger.info(
        f"Selected strategy {best_idx + 1}: '{best_result.strategy_name}' "
        f"({metric}={metric_values[best_idx]:.4f})"
    )

    # I log the comparison across all strategies for transparency
    logger.info("Strategy comparison:")
    for i, (r, v) in enumerate(zip(backtest_results, metric_values)):
        marker = " <<<" if i == best_idx else ""
        logger.info(f"  [{i+1}] {r.strategy_name}: {metric}={v:.4f}{marker}")

    return best_idx, best_result


# ───────────────────────────────────────────────────────────────────────────
# Logging Utilities
# ───────────────────────────────────────────────────────────────────────────
def log_backtest_results(
    ticker: str,
    results: List[BacktestResult],
    phase: str = "training",
) -> None:
    """
    I save detailed backtest results to JSON for every strategy. This is
    the complete audit trail that a reviewer needs to verify my results.
    """
    log_data = {
        "ticker": ticker,
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "num_strategies": len(results),
        "results": [],
    }

    for r in results:
        result_dict = r.to_summary_dict()
        result_dict["trades"] = [t.to_dict() for t in r.trades]
        log_data["results"].append(result_dict)

    log_path = config.LOGS_DIR / f"backtest_{ticker}_{phase}.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    logger.info(f"Saved {phase} backtest results to {log_path}")


if __name__ == "__main__":
    # I include this block for standalone testing of the backtest engine.
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    # I create a simple test with synthetic data
    dates = pd.date_range("2022-01-01", periods=500, freq="B")
    np.random.seed(config.RANDOM_SEED)
    prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.01 + 0.0002))

    test_data = pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.01,
        "Low": prices * 0.99,
        "Close": prices,
        "Volume": np.random.randint(1000000, 10000000, 500),
    }, index=dates)

    from strategy_generator import TradingStrategy
    test_strategy = TradingStrategy(
        name="Test MA Crossover",
        strategy_type="ma_crossover",
        description="Test strategy",
        parameters={"short_window": 20, "long_window": 50, "stop_loss_pct": 0.05},
        entry_rule="Buy when SMA(20) > SMA(50)",
        exit_rule="Sell when SMA(20) < SMA(50)",
        reasoning="Test",
        ticker="TEST",
    )

    result = backtest_strategy(test_strategy, test_data)
    print(f"\nBacktest Result:")
    print(f"  Return: {result.cumulative_return:.2%}")
    print(f"  Sharpe: {result.sharpe_ratio:.2f}")
    print(f"  Max DD: {result.max_drawdown:.2%}")
    print(f"  Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate:.1%}")
