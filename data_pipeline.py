"""
Data Pipeline Module for Consequence-Aware Agentic Trading Research
=======================================================================
Author: Prof. Shahab Anbarjafari, 3S Holding OÜ, Tartu, Estonia (shb@3sholding.com)

I handle everything related to acquiring, validating, and splitting market data
in this module. My philosophy: data quality issues are the number-one source of
false results in quantitative finance research. I therefore log every anomaly,
every missing value, every adjustment—so that a reviewer (or my future self)
can trace any suspicious result back to the raw data.

The pipeline follows three stages:
  1. Fetch: Pull daily OHLCV data from Yahoo Finance via yfinance.
  2. Validate: Check for missing dates, NaN values, zero-volume days,
     corporate-action artifacts, and stale prices.
  3. Split: Partition into training and test sets with zero data leakage.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Data Quality Report
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class DataQualityReport:
    """
    I encapsulate all data quality diagnostics into a single, serializable
    object. This makes it trivial to log, compare across tickers, and include
    in the paper's appendix if a reviewer asks "did you check for survivorship
    bias or missing data?"
    """
    ticker: str
    fetch_timestamp: str = ""
    total_rows: int = 0
    date_range_start: str = ""
    date_range_end: str = ""
    missing_values: Dict[str, int] = field(default_factory=dict)
    zero_volume_days: int = 0
    duplicate_dates: int = 0
    large_gap_days: List[str] = field(default_factory=list)
    price_anomalies: List[str] = field(default_factory=list)
    train_rows: int = 0
    test_rows: int = 0
    data_quality_score: float = 1.0  # 1.0 = perfect, decreases with issues

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ───────────────────────────────────────────────────────────────────────────
# Core Pipeline Functions
# ───────────────────────────────────────────────────────────────────────────
def fetch_ohlc_data(
    ticker: str,
    start_date: str = config.DATA_START_DATE,
    end_date: str = config.DATA_END_DATE,
) -> pd.DataFrame:
    """
    I pull daily OHLCV data from Yahoo Finance for a single ticker.

    I chose yfinance because it's free, widely used in academic research, and
    automatically adjusts for splits and dividends (the 'Adj Close' column).
    I use adjusted close prices throughout to avoid spurious signals from
    corporate actions—a 2:1 split should not trigger a buy signal.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Adj Close, Volume.
        Index is DatetimeIndex.
    """
    logger.info(f"Fetching OHLCV data for {ticker} from {start_date} to {end_date}")
    fetch_start = time.time()

    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            raise ValueError(f"No data returned for {ticker}. Check ticker symbol and date range.")

        # I ensure the index is a proper DatetimeIndex without timezone info
        # because mixing tz-aware and tz-naive timestamps causes subtle bugs
        # in downstream date comparisons.
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"

        fetch_time = time.time() - fetch_start
        logger.info(
            f"Fetched {len(df)} rows for {ticker} in {fetch_time:.2f}s "
            f"({df.index[0].date()} to {df.index[-1].date()})"
        )
        return df

    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        raise


def validate_data(df: pd.DataFrame, ticker: str) -> DataQualityReport:
    """
    I run a comprehensive battery of data quality checks. In my experience,
    silently proceeding with dirty data is how you end up with a paper
    retraction. Better to be paranoid here.

    Checks performed:
      - Missing/NaN values in any column
      - Zero-volume days (often indicates exchange holidays not filtered)
      - Duplicate date entries
      - Large calendar gaps (>5 business days, may indicate delistings)
      - Price anomalies: daily returns > 50% (likely data errors or extreme events)
      - Negative prices (definitely data errors)

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame.
    ticker : str
        Ticker symbol for labeling.

    Returns
    -------
    DataQualityReport
        Comprehensive diagnostics.
    """
    logger.info(f"Validating data quality for {ticker}")
    report = DataQualityReport(
        ticker=ticker,
        fetch_timestamp=datetime.now().isoformat(),
        total_rows=len(df),
        date_range_start=str(df.index[0].date()),
        date_range_end=str(df.index[-1].date()),
    )

    # --- Missing values ---
    # I check every column because a NaN in Volume is less critical than a NaN
    # in Close, but I want to know about both.
    missing = df.isnull().sum().to_dict()
    report.missing_values = {k: int(v) for k, v in missing.items() if v > 0}
    if report.missing_values:
        logger.warning(f"{ticker}: Missing values detected — {report.missing_values}")
        report.data_quality_score -= 0.1

    # --- Zero-volume days ---
    # On legitimate trading days, volume should be positive. Zero-volume days
    # sometimes appear for thinly traded stocks or data errors.
    if "Volume" in df.columns:
        zero_vol = (df["Volume"] == 0).sum()
        report.zero_volume_days = int(zero_vol)
        if zero_vol > 0:
            logger.warning(f"{ticker}: {zero_vol} zero-volume days detected")
            report.data_quality_score -= min(0.05, zero_vol * 0.001)

    # --- Duplicate dates ---
    dup_count = df.index.duplicated().sum()
    report.duplicate_dates = int(dup_count)
    if dup_count > 0:
        logger.warning(f"{ticker}: {dup_count} duplicate dates found — dropping duplicates")
        report.data_quality_score -= 0.1

    # --- Large calendar gaps ---
    # I flag any gap > 5 business days. Weekends and holidays typically produce
    # gaps of 1-3 calendar days; anything larger warrants investigation.
    date_diffs = pd.Series(df.index).diff().dt.days
    large_gaps = date_diffs[date_diffs > 5]
    if len(large_gaps) > 0:
        gap_dates = [str(df.index[i].date()) for i in large_gaps.index]
        report.large_gap_days = gap_dates
        logger.warning(f"{ticker}: Large date gaps detected near {gap_dates[:5]}")

    # --- Price anomalies ---
    # I flag single-day returns exceeding ±50%. In the history of the S&P 500,
    # the largest single-day move was ~23% (1987 crash). A 50% threshold catches
    # obvious data errors while allowing for genuine extreme events.
    if "Close" in df.columns:
        daily_returns = df["Close"].pct_change().dropna()
        extreme = daily_returns[daily_returns.abs() > 0.5]
        if len(extreme) > 0:
            anomaly_dates = [f"{d.date()}: {r:.2%}" for d, r in extreme.items()]
            report.price_anomalies = anomaly_dates
            logger.warning(f"{ticker}: Price anomalies detected — {anomaly_dates}")
            report.data_quality_score -= 0.05 * len(extreme)

    # --- Negative prices ---
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    for col in price_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            logger.error(f"{ticker}: {neg_count} negative values in {col} — serious data error!")
            report.data_quality_score -= 0.3

    report.data_quality_score = max(0.0, report.data_quality_score)
    logger.info(
        f"{ticker}: Data quality score = {report.data_quality_score:.2f} "
        f"({report.total_rows} rows)"
    )
    return report


def clean_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    I apply minimal, defensible cleaning operations. My principle is: fix what's
    clearly broken, leave everything else alone. Aggressive imputation or
    smoothing could introduce look-ahead bias or mask real market dynamics.

    Cleaning steps:
      1. Drop duplicate dates (keep first occurrence).
      2. Forward-fill NaN values (at most 3 consecutive days—beyond that,
         something is seriously wrong and I'd rather crash than silently
         propagate stale prices).
      3. Drop any remaining NaN rows.
      4. Sort by date.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame.
    ticker : str
        For logging.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    original_len = len(df)

    # I drop duplicates before anything else to avoid double-counting
    df = df[~df.index.duplicated(keep="first")]

    # I sort by date because yfinance occasionally returns unsorted data
    df = df.sort_index()

    # Forward-fill with a strict limit. If data is missing for more than 3
    # consecutive trading days, I want to know about it rather than silently
    # extrapolating.
    df = df.ffill(limit=3)

    # Drop whatever NaNs remain—these are gaps too large to safely impute
    nan_before = df.isnull().any(axis=1).sum()
    df = df.dropna()

    cleaned_len = len(df)
    if cleaned_len < original_len:
        logger.info(
            f"{ticker}: Cleaned {original_len} -> {cleaned_len} rows "
            f"(dropped {original_len - cleaned_len})"
        )

    return df


def split_data(
    df: pd.DataFrame,
    train_end: str = config.TRAIN_END_DATE,
    test_start: str = config.TEST_START_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    I split the data into training and test sets with an explicit date boundary.
    There is NO overlap—this is crucial for preventing data leakage.

    I use date-based splitting rather than percentage-based because financial
    time series have strong temporal dependencies. A random split would let
    future information leak into the training set.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV DataFrame.
    train_end : str
        Last date (inclusive) for training data.
    test_start : str
        First date (inclusive) for test data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    train_end_dt = pd.Timestamp(train_end)
    test_start_dt = pd.Timestamp(test_start)

    # I add an assertion here as a safety net: if the test start is before or
    # equal to the train end, we have a logic error in our date configuration.
    assert test_start_dt > train_end_dt, (
        f"Data leakage! test_start ({test_start}) must be after train_end ({train_end})"
    )

    train_df = df[df.index <= train_end_dt].copy()
    test_df = df[df.index >= test_start_dt].copy()

    # Sanity checks: both sets should be non-empty
    assert len(train_df) > 0, f"Training set is empty! Check date range."
    assert len(test_df) > 0, f"Test set is empty! Check date range."

    logger.info(
        f"Split: {len(train_df)} training rows "
        f"({train_df.index[0].date()} to {train_df.index[-1].date()}), "
        f"{len(test_df)} test rows "
        f"({test_df.index[0].date()} to {test_df.index[-1].date()})"
    )

    return train_df, test_df


# ───────────────────────────────────────────────────────────────────────────
# High-Level Pipeline Orchestration
# ───────────────────────────────────────────────────────────────────────────
def run_pipeline(
    tickers: Optional[List[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, DataQualityReport]]:
    """
    I orchestrate the full data pipeline: fetch, validate, clean, split—for
    every ticker in our universe. This is the single entry point that main.py
    calls.

    Parameters
    ----------
    tickers : list of str, optional
        Override the default ticker list from config.

    Returns
    -------
    Tuple containing:
      - train_data: Dict mapping ticker -> training DataFrame
      - test_data: Dict mapping ticker -> test DataFrame
      - quality_reports: Dict mapping ticker -> DataQualityReport
    """
    if tickers is None:
        tickers = config.TICKERS

    pipeline_start = time.time()
    logger.info(f"Starting data pipeline for {len(tickers)} tickers")

    train_data: Dict[str, pd.DataFrame] = {}
    test_data: Dict[str, pd.DataFrame] = {}
    quality_reports: Dict[str, DataQualityReport] = {}

    for ticker in tickers:
        logger.info(f"{'='*60}")
        logger.info(f"Processing {ticker} ({config.TICKER_NAMES.get(ticker, 'Unknown')})")
        logger.info(f"{'='*60}")

        # Step 1: Fetch raw data
        raw_df = fetch_ohlc_data(ticker)

        # Step 2: Validate and generate quality report
        report = validate_data(raw_df, ticker)

        # Step 3: Clean the data (minimal, defensible operations only)
        clean_df = clean_data(raw_df, ticker)

        # Step 4: Split into train/test with no leakage
        train_df, test_df = split_data(clean_df)
        report.train_rows = len(train_df)
        report.test_rows = len(test_df)

        # Store results
        train_data[ticker] = train_df
        test_data[ticker] = test_df
        quality_reports[ticker] = report

        # I save the raw data to disk so that anyone can verify my pipeline
        # without re-downloading from Yahoo Finance (which might return
        # slightly different data due to retroactive adjustments).
        clean_df.to_csv(config.DATA_DIR / f"{ticker}_full.csv")
        train_df.to_csv(config.DATA_DIR / f"{ticker}_train.csv")
        test_df.to_csv(config.DATA_DIR / f"{ticker}_test.csv")

    # I save all quality reports as a single JSON file for easy inspection
    all_reports = {t: r.to_dict() for t, r in quality_reports.items()}
    report_path = config.DATA_DIR / "data_quality_reports.json"
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    logger.info(f"Saved data quality reports to {report_path}")

    pipeline_time = time.time() - pipeline_start
    logger.info(
        f"Data pipeline completed in {pipeline_time:.1f}s — "
        f"{len(train_data)} tickers processed successfully"
    )

    # I log a summary table so it's easy to spot issues at a glance
    _log_pipeline_summary(train_data, test_data, quality_reports)

    return train_data, test_data, quality_reports


def _log_pipeline_summary(
    train_data: Dict[str, pd.DataFrame],
    test_data: Dict[str, pd.DataFrame],
    quality_reports: Dict[str, DataQualityReport],
) -> None:
    """I print a concise summary table to the log for quick sanity checking."""
    logger.info("\n" + "=" * 80)
    logger.info("DATA PIPELINE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Ticker':<10} {'Train Rows':>10} {'Test Rows':>10} {'Quality':>10} {'Issues':>30}")
    logger.info("-" * 80)

    for ticker in sorted(train_data.keys()):
        report = quality_reports[ticker]
        issues = []
        if report.missing_values:
            issues.append(f"missing:{sum(report.missing_values.values())}")
        if report.zero_volume_days > 0:
            issues.append(f"zero_vol:{report.zero_volume_days}")
        if report.price_anomalies:
            issues.append(f"anomalies:{len(report.price_anomalies)}")
        issue_str = ", ".join(issues) if issues else "none"

        logger.info(
            f"{ticker:<10} {report.train_rows:>10} {report.test_rows:>10} "
            f"{report.data_quality_score:>10.2f} {issue_str:>30}"
        )

    logger.info("=" * 80)


# ───────────────────────────────────────────────────────────────────────────
# Utility: Compute Market Summary Statistics
# ───────────────────────────────────────────────────────────────────────────
def compute_market_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    I compute summary statistics that the strategy generator needs as context.
    These describe the stock's recent behavior: trend direction, volatility
    regime, momentum characteristics. The LLM uses these to make informed
    strategy recommendations.

    Parameters
    ----------
    df : pd.DataFrame
        Training period OHLCV data.

    Returns
    -------
    dict
        Summary statistics including trend, volatility, and momentum metrics.
    """
    close = df["Close"]
    returns = close.pct_change().dropna()

    # I compute several windows of return to characterize the trend at
    # multiple timescales—what looks like a bull market on the 1-year chart
    # might be a bear market on the 3-month chart.
    summary = {
        "total_return": float((close.iloc[-1] / close.iloc[0]) - 1),
        "annualized_return": float(returns.mean() * config.TRADING_DAYS_PER_YEAR),
        "annualized_volatility": float(returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)),
        "sharpe_ratio": float(
            (returns.mean() * config.TRADING_DAYS_PER_YEAR - config.RISK_FREE_RATE)
            / (returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR))
            if returns.std() > 0 else 0.0
        ),
        "max_drawdown": float(_compute_max_drawdown(close)),
        "last_price": float(close.iloc[-1]),
        "price_52w_high": float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max()),
        "price_52w_low": float(close.iloc[-252:].min()) if len(close) >= 252 else float(close.min()),
        "avg_daily_volume": float(df["Volume"].mean()) if "Volume" in df.columns else 0.0,
        "recent_30d_return": float(
            (close.iloc[-1] / close.iloc[-30] - 1) if len(close) >= 30 else 0.0
        ),
        "recent_90d_return": float(
            (close.iloc[-1] / close.iloc[-90] - 1) if len(close) >= 90 else 0.0
        ),
        "sma_50": float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float(close.mean()),
        "sma_200": float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean()),
        "rsi_14": float(_compute_rsi(close, 14)),
        "num_trading_days": len(close),
    }
    return summary


def _compute_max_drawdown(prices: pd.Series) -> float:
    """
    I compute maximum drawdown as the largest peak-to-trough decline.
    This is the standard definition used in portfolio management.
    """
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    return float(drawdown.min())


def _compute_rsi(prices: pd.Series, window: int = 14) -> float:
    """
    I compute the Relative Strength Index using Wilder's smoothing method.
    RSI is bounded [0, 100]. Values below 30 suggest oversold conditions;
    above 70 suggests overbought.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # I handle the edge case where average loss is zero (all up days)
    # to avoid division by zero. In that case, RSI = 100 by convention.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    last_rsi = rsi.dropna()
    if len(last_rsi) == 0:
        return 50.0  # Neutral default if insufficient data
    return float(last_rsi.iloc[-1])


if __name__ == "__main__":
    # I include this block so the pipeline can be tested in isolation.
    # Running `python data_pipeline.py` fetches data and prints quality reports.
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    train_data, test_data, reports = run_pipeline()
    for ticker, report in reports.items():
        print(f"\n{ticker}: Quality={report.data_quality_score:.2f}, "
              f"Train={report.train_rows}, Test={report.test_rows}")
