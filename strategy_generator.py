"""
Strategy Generation Module for Consequence-Aware Agentic Trading Research
=======================================================================
Author: Prof. Shahab Anbarjafari, 3S Holding OÜ, Tartu, Estonia (shb@3sholding.com)

This module is the "brain" of the agentic system. I interface with an LLM
(OpenAI GPT-5.4) to generate diverse, executable trading strategies for each
stock based on its historical market characteristics.

My design philosophy:
  1. The LLM is a *strategy proposer*, not a black box. Every strategy it
     generates must be parseable into a structured format that my backtesting
     engine can execute mechanically.
  2. I constrain the strategy space to well-known technical analysis patterns
     (MA crossover, RSI, Bollinger, MACD, momentum) with configurable
     parameters. This keeps strategies interpretable and backtestable.
  3. If no LLM API key is available, I fall back to a deterministic rule-based
     generator. This ensures full reproducibility without API access—critical
     for peer review.

Strategy Representation:
  Each strategy is a dictionary with:
    - name: Human-readable name
    - type: One of the supported strategy types
    - description: Plain-English description of the logic
    - parameters: Dict of numeric parameters for the backtest engine
    - entry_rule: Natural language entry condition
    - exit_rule: Natural language exit condition
    - reasoning: Why this strategy was proposed for this stock
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Strategy Data Structure
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class TradingStrategy:
    """
    I represent a single trading strategy as a structured, serializable object.
    Every field is typed and documented so that the backtester knows exactly
    what to do with it. The LLM's creativity is channeled through the
    'parameters' dict—it can tune windows, thresholds, and stop-losses, but
    the execution logic is mine.
    """
    name: str
    strategy_type: str
    description: str
    parameters: Dict[str, Any]
    entry_rule: str
    exit_rule: str
    reasoning: str
    ticker: str = ""
    generation_timestamp: str = ""
    generation_method: str = ""  # "llm" or "rule_based"
    raw_llm_output: str = ""    # I keep the raw output for auditability

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def validate(self) -> bool:
        """
        I check that this strategy has all required fields and that the
        parameters make sense (e.g., short MA window < long MA window).
        Returns True if valid, raises ValueError if not.
        """
        if self.strategy_type not in config.STRATEGY_TYPES:
            raise ValueError(
                f"Unknown strategy type '{self.strategy_type}'. "
                f"Must be one of {config.STRATEGY_TYPES}"
            )

        if not self.parameters:
            raise ValueError("Strategy parameters cannot be empty")

        # I perform type-specific validation here
        if self.strategy_type == "ma_crossover":
            short_w = self.parameters.get("short_window", 0)
            long_w = self.parameters.get("long_window", 0)
            if short_w >= long_w:
                raise ValueError(
                    f"MA crossover: short_window ({short_w}) must be < long_window ({long_w})"
                )

        if self.strategy_type == "rsi_mean_reversion":
            oversold = self.parameters.get("oversold_threshold", 30)
            overbought = self.parameters.get("overbought_threshold", 70)
            if oversold >= overbought:
                raise ValueError(
                    f"RSI: oversold ({oversold}) must be < overbought ({overbought})"
                )

        return True


# ───────────────────────────────────────────────────────────────────────────
# LLM-Based Strategy Generator
# ───────────────────────────────────────────────────────────────────────────
def _build_llm_prompt(ticker: str, market_summary: Dict[str, float]) -> str:
    """
    I craft a detailed prompt that gives the LLM enough context to make
    informed strategy recommendations, while constraining its output to
    a parseable JSON format.

    I include the stock's market summary statistics so the LLM can reason
    about whether to propose trend-following vs. mean-reversion strategies
    based on recent volatility and momentum.
    """
    prompt = f"""You are an expert quantitative analyst. I need you to generate exactly 5 diverse trading strategies for {ticker} ({config.TICKER_NAMES.get(ticker, '')}).

## Current Market Context for {ticker}
- Total historical return: {market_summary.get('total_return', 0):.2%}
- Annualized return: {market_summary.get('annualized_return', 0):.2%}
- Annualized volatility: {market_summary.get('annualized_volatility', 0):.2%}
- Historical Sharpe ratio: {market_summary.get('sharpe_ratio', 0):.2f}
- Maximum drawdown: {market_summary.get('max_drawdown', 0):.2%}
- Last price: ${market_summary.get('last_price', 0):.2f}
- 52-week high: ${market_summary.get('price_52w_high', 0):.2f}
- 52-week low: ${market_summary.get('price_52w_low', 0):.2f}
- 30-day return: {market_summary.get('recent_30d_return', 0):.2%}
- 90-day return: {market_summary.get('recent_90d_return', 0):.2%}
- SMA(50): ${market_summary.get('sma_50', 0):.2f}
- SMA(200): ${market_summary.get('sma_200', 0):.2f}
- RSI(14): {market_summary.get('rsi_14', 0):.1f}

## Requirements
Generate exactly 5 strategies, one of each type:
1. "ma_crossover" — Moving average crossover
2. "rsi_mean_reversion" — RSI-based mean reversion
3. "bollinger_breakout" — Bollinger Band strategy
4. "macd_signal" — MACD-based strategy
5. "momentum" — Price momentum with stops

Each strategy MUST be a JSON object with these exact fields:
- "name": string (creative, descriptive name)
- "strategy_type": string (one of the 5 types above)
- "description": string (2-3 sentences explaining the logic)
- "parameters": object with type-specific parameters (see below)
- "entry_rule": string (precise entry condition)
- "exit_rule": string (precise exit condition)
- "reasoning": string (why this strategy suits {ticker} given the market context)

## Parameter Specifications by Type

### ma_crossover
- "short_window": int (5-50 days)
- "long_window": int (20-200 days, must be > short_window)
- "stop_loss_pct": float (0.01-0.15)

### rsi_mean_reversion
- "rsi_window": int (7-21 days)
- "oversold_threshold": float (20-40)
- "overbought_threshold": float (60-80)
- "holding_period": int (3-20 days)
- "stop_loss_pct": float (0.01-0.15)

### bollinger_breakout
- "bb_window": int (10-30 days)
- "bb_std": float (1.5-3.0 standard deviations)
- "exit_at_middle": bool (true/false)
- "stop_loss_pct": float (0.01-0.15)

### macd_signal
- "fast_period": int (8-16)
- "slow_period": int (20-30)
- "signal_period": int (5-12)
- "stop_loss_pct": float (0.01-0.15)

### momentum
- "lookback_period": int (10-60 days)
- "entry_threshold": float (0.02-0.15, minimum return to trigger entry)
- "trailing_stop_pct": float (0.03-0.15)
- "max_holding_period": int (10-40 days)

Respond with ONLY a JSON array of 5 strategy objects. No markdown, no explanation outside the JSON."""

    return prompt


def generate_strategies_llm(
    ticker: str,
    market_summary: Dict[str, float],
) -> List[TradingStrategy]:
    """
    I call the OpenAI GPT-5.4 API to generate trading strategies. The LLM
    receives market context and returns structured strategy definitions that
    my backtester can execute.

    I parse the response carefully, validate each strategy, and log the raw
    LLM output for reproducibility and debugging.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed. Falling back to rule-based generator.")
        return generate_strategies_rule_based(ticker, market_summary)

    if not config.OPENAI_API_KEY:
        logger.info("No OPENAI_API_KEY set. Using rule-based strategy generator.")
        return generate_strategies_rule_based(ticker, market_summary)

    logger.info(f"Generating strategies for {ticker} via LLM ({config.LLM_MODEL})")
    prompt = _build_llm_prompt(ticker, market_summary)

    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert quantitative analyst. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )

        raw_output = response.choices[0].message.content
        logger.info(f"LLM response received for {ticker} ({len(raw_output)} chars)")

        # I attempt to parse the JSON response. The LLM sometimes wraps its
        # output in markdown code blocks, so I strip those if present.
        clean_output = raw_output.strip()
        if clean_output.startswith("```"):
            clean_output = clean_output.split("\n", 1)[1]
            if clean_output.endswith("```"):
                clean_output = clean_output[:-3]
            clean_output = clean_output.strip()

        strategy_dicts = json.loads(clean_output)

        strategies = []
        for i, sd in enumerate(strategy_dicts):
            strategy = TradingStrategy(
                name=sd.get("name", f"Strategy_{i+1}"),
                strategy_type=sd.get("strategy_type", ""),
                description=sd.get("description", ""),
                parameters=sd.get("parameters", {}),
                entry_rule=sd.get("entry_rule", ""),
                exit_rule=sd.get("exit_rule", ""),
                reasoning=sd.get("reasoning", ""),
                ticker=ticker,
                generation_timestamp=datetime.now().isoformat(),
                generation_method="llm",
                raw_llm_output=raw_output,
            )
            try:
                strategy.validate()
                strategies.append(strategy)
            except ValueError as e:
                logger.warning(f"Strategy {i+1} for {ticker} failed validation: {e}")
                # I fall back to a rule-based strategy for this slot
                fallback = _generate_single_rule_based(
                    ticker, market_summary, config.STRATEGY_TYPES[i]
                )
                strategies.append(fallback)

        # I ensure we always have exactly 5 strategies
        while len(strategies) < config.NUM_STRATEGIES_PER_STOCK:
            idx = len(strategies)
            stype = config.STRATEGY_TYPES[idx % len(config.STRATEGY_TYPES)]
            strategies.append(
                _generate_single_rule_based(ticker, market_summary, stype)
            )

        return strategies[:config.NUM_STRATEGIES_PER_STOCK]

    except Exception as e:
        logger.error(f"LLM generation failed for {ticker}: {e}. Falling back to rule-based.")
        return generate_strategies_rule_based(ticker, market_summary)


# ───────────────────────────────────────────────────────────────────────────
# Deterministic Rule-Based Strategy Generator (Fallback)
# ───────────────────────────────────────────────────────────────────────────
def generate_strategies_rule_based(
    ticker: str,
    market_summary: Dict[str, float],
) -> List[TradingStrategy]:
    """
    I generate five deterministic strategies based on the stock's historical
    characteristics. This is my fallback when no LLM API key is available,
    but it's also useful as a reproducibility baseline—the same market summary
    always produces the same strategies.

    I adapt parameters to the stock's volatility and trend characteristics:
    - High-volatility stocks get wider stops and shorter windows
    - Trending stocks get trend-following strategies with tighter parameters
    - Mean-reverting stocks get wider RSI bands and longer holding periods

    This is NOT random parameter selection—I'm encoding domain knowledge about
    how technical indicators should be tuned for different market regimes.
    """
    logger.info(f"Generating rule-based strategies for {ticker}")
    rng = np.random.RandomState(config.RANDOM_SEED + hash(ticker) % 10000)

    strategies = []
    for stype in config.STRATEGY_TYPES:
        strategy = _generate_single_rule_based(ticker, market_summary, stype, rng)
        strategies.append(strategy)

    return strategies


def _generate_single_rule_based(
    ticker: str,
    market_summary: Dict[str, float],
    strategy_type: str,
    rng: Optional[np.random.RandomState] = None,
) -> TradingStrategy:
    """
    I generate a single strategy of a given type, with parameters adapted
    to the stock's characteristics. I use the market summary to make
    intelligent parameter choices—this is what a human quant would do.
    """
    if rng is None:
        rng = np.random.RandomState(config.RANDOM_SEED + hash(ticker + strategy_type) % 10000)

    vol = market_summary.get("annualized_volatility", 0.25)
    trend = market_summary.get("total_return", 0.0)
    rsi = market_summary.get("rsi_14", 50.0)

    # I scale stop losses with volatility: higher-vol stocks need wider stops
    # to avoid being stopped out by normal fluctuations.
    base_stop = min(0.15, max(0.03, vol * 0.15))

    if strategy_type == "ma_crossover":
        # For trending stocks, I use shorter windows to catch trends earlier.
        # For choppy stocks, I use longer windows to filter noise.
        if abs(trend) > 0.5:
            short_w = rng.choice([10, 15, 20])
            long_w = rng.choice([50, 60, 75])
        else:
            short_w = rng.choice([20, 30, 40])
            long_w = rng.choice([100, 150, 200])

        return TradingStrategy(
            name=f"MA Crossover ({short_w}/{long_w})",
            strategy_type="ma_crossover",
            description=(
                f"I buy {ticker} when the {short_w}-day SMA crosses above the "
                f"{long_w}-day SMA (golden cross) and sell on the reverse (death cross). "
                f"This is a classic trend-following approach tuned for this stock's "
                f"volatility regime."
            ),
            parameters={
                "short_window": int(short_w),
                "long_window": int(long_w),
                "stop_loss_pct": round(float(base_stop), 3),
            },
            entry_rule=f"Buy when SMA({short_w}) crosses above SMA({long_w})",
            exit_rule=f"Sell when SMA({short_w}) crosses below SMA({long_w}) or stop loss at {base_stop:.1%}",
            reasoning=(
                f"Given {ticker}'s annualized volatility of {vol:.1%} and total return of "
                f"{trend:.1%}, I chose {'shorter' if abs(trend) > 0.5 else 'longer'} "
                f"moving average windows to {'capture strong trends' if abs(trend) > 0.5 else 'filter market noise'}."
            ),
            ticker=ticker,
            generation_timestamp=datetime.now().isoformat(),
            generation_method="rule_based",
        )

    elif strategy_type == "rsi_mean_reversion":
        # I widen the RSI bands for volatile stocks—they'll naturally hit
        # more extreme RSI values, so I need wider bands to avoid false signals.
        if vol > 0.35:
            oversold = rng.choice([20, 22, 25])
            overbought = rng.choice([75, 78, 80])
            holding = rng.choice([5, 7, 10])
        else:
            oversold = rng.choice([28, 30, 32])
            overbought = rng.choice([68, 70, 72])
            holding = rng.choice([7, 10, 15])

        rsi_window = rng.choice([10, 12, 14])

        return TradingStrategy(
            name=f"RSI Mean Reversion ({rsi_window}d, {oversold}/{overbought})",
            strategy_type="rsi_mean_reversion",
            description=(
                f"I buy {ticker} when RSI({rsi_window}) drops below {oversold} (oversold) "
                f"and sell when RSI rises above {overbought} (overbought) or after "
                f"{holding} trading days. This exploits short-term overreactions."
            ),
            parameters={
                "rsi_window": int(rsi_window),
                "oversold_threshold": float(oversold),
                "overbought_threshold": float(overbought),
                "holding_period": int(holding),
                "stop_loss_pct": round(float(base_stop), 3),
            },
            entry_rule=f"Buy when RSI({rsi_window}) < {oversold}",
            exit_rule=f"Sell when RSI({rsi_window}) > {overbought} or after {holding} days or stop loss",
            reasoning=(
                f"With current RSI at {rsi:.1f} and volatility at {vol:.1%}, "
                f"I set {'wider' if vol > 0.35 else 'standard'} RSI bands to "
                f"capture mean-reversion opportunities without excessive false signals."
            ),
            ticker=ticker,
            generation_timestamp=datetime.now().isoformat(),
            generation_method="rule_based",
        )

    elif strategy_type == "bollinger_breakout":
        if vol > 0.35:
            bb_std = rng.choice([2.5, 2.75, 3.0])
            bb_window = rng.choice([15, 20, 25])
        else:
            bb_std = rng.choice([1.75, 2.0, 2.25])
            bb_window = rng.choice([18, 20, 22])

        return TradingStrategy(
            name=f"Bollinger Breakout ({bb_window}d, {bb_std}σ)",
            strategy_type="bollinger_breakout",
            description=(
                f"I buy {ticker} when price breaks above the upper Bollinger Band "
                f"({bb_window}-day, {bb_std}σ) as a momentum breakout signal. "
                f"I exit at the middle band or on stop loss."
            ),
            parameters={
                "bb_window": int(bb_window),
                "bb_std": float(bb_std),
                "exit_at_middle": True,
                "stop_loss_pct": round(float(base_stop), 3),
            },
            entry_rule=f"Buy when Close > Upper BB({bb_window}, {bb_std}σ)",
            exit_rule=f"Sell when Close < Middle BB({bb_window}) or stop loss",
            reasoning=(
                f"Bollinger Bands with {bb_std}σ width are calibrated for {ticker}'s "
                f"volatility ({vol:.1%}). Wider bands for high-vol stocks reduce "
                f"false breakout signals."
            ),
            ticker=ticker,
            generation_timestamp=datetime.now().isoformat(),
            generation_method="rule_based",
        )

    elif strategy_type == "macd_signal":
        if abs(trend) > 0.5:
            fast = rng.choice([8, 10, 12])
            slow = rng.choice([20, 22, 26])
            signal = rng.choice([7, 8, 9])
        else:
            fast = rng.choice([10, 12, 14])
            slow = rng.choice([24, 26, 28])
            signal = rng.choice([8, 9, 10])

        return TradingStrategy(
            name=f"MACD Signal ({fast}/{slow}/{signal})",
            strategy_type="macd_signal",
            description=(
                f"I buy {ticker} when MACD({fast},{slow}) crosses above its "
                f"{signal}-period signal line (bullish crossover) and sell on "
                f"bearish crossover. Classic momentum indicator."
            ),
            parameters={
                "fast_period": int(fast),
                "slow_period": int(slow),
                "signal_period": int(signal),
                "stop_loss_pct": round(float(base_stop), 3),
            },
            entry_rule=f"Buy when MACD({fast},{slow}) crosses above Signal({signal})",
            exit_rule=f"Sell when MACD crosses below Signal or stop loss",
            reasoning=(
                f"MACD parameters are tuned for {ticker}'s trend characteristics "
                f"(total return: {trend:.1%}). "
                f"{'Faster' if abs(trend) > 0.5 else 'Standard'} settings to "
                f"{'capture strong momentum' if abs(trend) > 0.5 else 'reduce whipsaws'}."
            ),
            ticker=ticker,
            generation_timestamp=datetime.now().isoformat(),
            generation_method="rule_based",
        )

    elif strategy_type == "momentum":
        if vol > 0.35:
            lookback = rng.choice([15, 20, 25])
            entry_thresh = round(float(rng.uniform(0.05, 0.12)), 3)
            trailing = round(float(rng.uniform(0.06, 0.12)), 3)
            max_hold = rng.choice([15, 20, 25])
        else:
            lookback = rng.choice([20, 30, 40])
            entry_thresh = round(float(rng.uniform(0.03, 0.08)), 3)
            trailing = round(float(rng.uniform(0.04, 0.08)), 3)
            max_hold = rng.choice([20, 25, 30])

        return TradingStrategy(
            name=f"Momentum ({lookback}d, {entry_thresh:.1%} threshold)",
            strategy_type="momentum",
            description=(
                f"I buy {ticker} when its {lookback}-day return exceeds {entry_thresh:.1%}, "
                f"riding the momentum with a {trailing:.1%} trailing stop. "
                f"Maximum holding period of {max_hold} days prevents overstaying."
            ),
            parameters={
                "lookback_period": int(lookback),
                "entry_threshold": float(entry_thresh),
                "trailing_stop_pct": float(trailing),
                "max_holding_period": int(max_hold),
            },
            entry_rule=f"Buy when {lookback}-day return > {entry_thresh:.1%}",
            exit_rule=f"Sell on {trailing:.1%} trailing stop or after {max_hold} days",
            reasoning=(
                f"Momentum parameters reflect {ticker}'s volatility ({vol:.1%}). "
                f"{'Shorter' if vol > 0.35 else 'Standard'} lookback and "
                f"{'wider' if vol > 0.35 else 'tighter'} stops for the current regime."
            ),
            ticker=ticker,
            generation_timestamp=datetime.now().isoformat(),
            generation_method="rule_based",
        )

    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


# ───────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────
def generate_strategies(
    ticker: str,
    market_summary: Dict[str, float],
    use_llm: bool = True,
) -> List[TradingStrategy]:
    """
    I generate a set of trading strategies for a given ticker. This is the
    main entry point that experiment_runner.py calls.

    I first attempt LLM generation (if use_llm=True and API key is available),
    then fall back to rule-based generation. Either way, I validate all
    strategies and log the complete generation record.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    market_summary : dict
        Market statistics from data_pipeline.compute_market_summary().
    use_llm : bool
        Whether to attempt LLM generation.

    Returns
    -------
    List[TradingStrategy]
        Exactly NUM_STRATEGIES_PER_STOCK strategies.
    """
    gen_start = time.time()
    logger.info(f"Generating {config.NUM_STRATEGIES_PER_STOCK} strategies for {ticker}")

    if use_llm and config.OPENAI_API_KEY:
        strategies = generate_strategies_llm(ticker, market_summary)
    else:
        strategies = generate_strategies_rule_based(ticker, market_summary)

    # I validate every strategy regardless of generation method
    valid_strategies = []
    for i, s in enumerate(strategies):
        try:
            s.validate()
            valid_strategies.append(s)
        except ValueError as e:
            logger.warning(f"Strategy {i+1} validation failed for {ticker}: {e}")

    # I ensure exactly the right number of strategies
    assert len(valid_strategies) == config.NUM_STRATEGIES_PER_STOCK, (
        f"Expected {config.NUM_STRATEGIES_PER_STOCK} strategies for {ticker}, "
        f"got {len(valid_strategies)}"
    )

    gen_time = time.time() - gen_start
    logger.info(
        f"Generated {len(valid_strategies)} strategies for {ticker} "
        f"in {gen_time:.2f}s (method: {valid_strategies[0].generation_method})"
    )

    # I log the full strategy set for auditability
    _log_strategies(ticker, valid_strategies)

    return valid_strategies


def _log_strategies(ticker: str, strategies: List[TradingStrategy]) -> None:
    """I save the complete strategy definitions to a JSON file for auditing."""
    strategy_log = {
        "ticker": ticker,
        "generation_time": datetime.now().isoformat(),
        "num_strategies": len(strategies),
        "strategies": [s.to_dict() for s in strategies],
    }

    log_path = config.LOGS_DIR / f"strategies_{ticker}.json"
    with open(log_path, "w") as f:
        json.dump(strategy_log, f, indent=2, default=str)
    logger.info(f"Saved strategy definitions to {log_path}")


def validate_strategy_diversity(strategies: List[TradingStrategy]) -> bool:
    """
    I check that the generated strategies are actually diverse—if the LLM
    generated five identical strategies, that defeats the purpose of having
    multiple candidates for consequence assessment.

    A valid strategy set should have at least 3 distinct strategy types.
    """
    types = set(s.strategy_type for s in strategies)
    is_diverse = len(types) >= 3

    if not is_diverse:
        logger.warning(
            f"Low strategy diversity: only {len(types)} distinct types "
            f"({types}). Expected at least 3."
        )

    return is_diverse


if __name__ == "__main__":
    # I include this for standalone testing. It generates strategies for AAPL
    # using the rule-based generator (no API key needed).
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    mock_summary = {
        "total_return": 0.45,
        "annualized_return": 0.09,
        "annualized_volatility": 0.28,
        "sharpe_ratio": 0.65,
        "max_drawdown": -0.22,
        "last_price": 185.0,
        "price_52w_high": 199.0,
        "price_52w_low": 140.0,
        "recent_30d_return": 0.03,
        "recent_90d_return": 0.08,
        "sma_50": 180.0,
        "sma_200": 170.0,
        "rsi_14": 55.0,
        "num_trading_days": 1100,
    }

    strategies = generate_strategies("AAPL", mock_summary, use_llm=False)
    for i, s in enumerate(strategies):
        print(f"\n{'='*60}")
        print(f"Strategy {i+1}: {s.name}")
        print(f"Type: {s.strategy_type}")
        print(f"Description: {s.description}")
        print(f"Parameters: {s.parameters}")
        print(f"Entry: {s.entry_rule}")
        print(f"Exit: {s.exit_rule}")
