"""
Configuration Module for Consequence-Aware Agentic Trading Research
=======================================================================
Author: Prof. Shahab Anbarjafari, 3S Holding OÜ, Tartu, Estonia (shb@3sholding.com)

I centralize every hyperparameter, path, and experimental constant here so that
anyone reproducing this work can immediately see what choices I made and why.
Every parameter is documented with the rationale from a research perspective—
a reviewer should be able to read this file alone and understand the experimental
design decisions.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# I load environment variables from the .env file at the project root.
# This keeps API keys out of the codebase and lets each collaborator
# maintain their own credentials without risking accidental commits.
load_dotenv()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# I fix the random seed globally so that every stochastic component—random
# baseline selection, any tie-breaking, shuffling—produces identical results
# across runs. This is non-negotiable for peer review.
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Universe of Assets
# ---------------------------------------------------------------------------
# I selected these ten tickers to span multiple sectors (tech, finance,
# consumer staples, automotive) and market-cap ranges. This diversity is
# deliberate: if consequence-aware selection only works on tech mega-caps,
# that is a limitation I want to expose, not hide.
TICKERS: List[str] = [
    "AAPL",   # Apple — technology, consumer electronics
    "MSFT",   # Microsoft — technology, enterprise software
    "TSLA",   # Tesla — automotive, high volatility
    "AMZN",   # Amazon — e-commerce, cloud computing
    "GOOGL",  # Alphabet — technology, advertising
    "META",   # Meta Platforms — social media, advertising
    "NVDA",   # Nvidia — semiconductors, AI hardware
    "JPM",    # JPMorgan Chase — financial services
    "BRK-B",  # Berkshire Hathaway Class B — diversified conglomerate
    "KO",     # Coca-Cola — consumer staples, dividend aristocrat
]

TICKER_NAMES: Dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "TSLA": "Tesla Inc.",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corp.",
    "JPM": "JPMorgan Chase & Co.",
    "BRK-B": "Berkshire Hathaway Inc.",
    "KO": "The Coca-Cola Company",
}

# ---------------------------------------------------------------------------
# Date Ranges
# ---------------------------------------------------------------------------
# I pull roughly five years of daily data. The split is:
#   - Training window: 2021-01-01 to 2025-07-31 (~4 years 7 months)
#     This window is used for strategy generation context and backtesting.
#   - Test window: 2025-08-01 to 2025-10-31 (3 months)
#     Strictly held out. No strategy parameter was tuned on this data.
#
# Why this specific split? Four years and seven months gives enough history
# for meaningful backtests of slower strategies (e.g., 200-day MA crossover
# needs ~200 trading days of warm-up). Three months of out-of-sample is
# short but sufficient to detect meaningful alpha if it exists—and honest
# enough not to cherry-pick a favorable window.
DATA_START_DATE: str = "2021-01-01"
DATA_END_DATE: str = "2025-10-31"
TRAIN_END_DATE: str = "2025-07-31"
TEST_START_DATE: str = "2025-08-01"

# ---------------------------------------------------------------------------
# Financial Constants
# ---------------------------------------------------------------------------
# I use the US 10-year Treasury yield as of early 2025 as my risk-free rate
# proxy. In practice, one might use the contemporaneous rate for each period,
# but a fixed rate makes the Sharpe ratio comparison cleaner across stocks.
RISK_FREE_RATE: float = 0.02  # 2% annualized
TRADING_DAYS_PER_YEAR: int = 252  # Standard for US equity markets
INITIAL_CAPITAL: float = 100_000.0  # Starting portfolio value in USD

# ---------------------------------------------------------------------------
# Strategy Generation
# ---------------------------------------------------------------------------
# I generate five candidate strategies per stock. Five is a pragmatic choice:
# enough to offer real diversity (MA crossover, RSI, Bollinger, MACD, momentum),
# but few enough that each strategy gets meaningful evaluation. With ten stocks
# this means fifty total backtests per experiment run.
NUM_STRATEGIES_PER_STOCK: int = 5

# I define the strategy types my backtester can execute. Each type corresponds
# to a well-known technical analysis approach. The LLM selects from these and
# configures parameters—it does not invent novel indicator math.
STRATEGY_TYPES: List[str] = [
    "ma_crossover",        # Moving average crossover (e.g., golden cross)
    "rsi_mean_reversion",  # RSI oversold/overbought mean reversion
    "bollinger_breakout",  # Bollinger Band breakout/mean reversion
    "macd_signal",         # MACD histogram signal crossover
    "momentum",            # Price momentum with trailing stop
]

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
# I use OpenAI's GPT-5.4 as the strategy-generating LLM. This is the latest
# reasoning model available at the time of this experiment. The model choice
# is documented here for reproducibility. I do not set temperature or
# max_tokens—I let the model use its default behavior for strategy reasoning.
LLM_MODEL: str = "gpt-5.4-2026-03-05"

# I read the API key from the .env file (loaded via python-dotenv above).
# The code gracefully falls back to a deterministic rule-based generator
# if no key is provided—this ensures the experiment is always runnable
# for reviewers who lack API access.
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Backtesting Parameters
# ---------------------------------------------------------------------------
# I assume execution at the close price on the signal day. This is a mild
# optimism bias (in practice you'd execute next-open), but it's consistent
# across all strategies and baselines, so relative comparisons remain fair.
EXECUTION_PRICE: str = "Close"
COMMISSION_RATE: float = 0.0  # Zero commissions for clarity; can be toggled
SLIPPAGE_BPS: float = 0.0     # Zero slippage; can be toggled

# ---------------------------------------------------------------------------
# Selection Criteria
# ---------------------------------------------------------------------------
# My primary selection metric is the Sharpe ratio. I chose Sharpe over raw
# return because it penalizes volatility—an agent that avoids catastrophic
# drawdowns is more useful than one that got lucky on a single position.
# I also run sensitivity analysis with alternative metrics (see experiment_runner).
PRIMARY_SELECTION_METRIC: str = "sharpe_ratio"
SENSITIVITY_METRICS: List[str] = ["cumulative_return", "calmar_ratio", "max_drawdown"]

# ---------------------------------------------------------------------------
# Output Directories
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).parent.resolve()
RESULTS_DIR: Path = PROJECT_ROOT / "results"
LOGS_DIR: Path = RESULTS_DIR / "logs"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
DATA_DIR: Path = RESULTS_DIR / "data"

# I create these directories at import time so downstream code never has to
# worry about whether they exist. This is a small convenience that prevents
# annoying FileNotFoundError crashes mid-experiment.
for _dir in [RESULTS_DIR, LOGS_DIR, FIGURES_DIR, DATA_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
EXPERIMENT_LOG_FILE: str = str(LOGS_DIR / "experiment.log")
JSON_LOG_FILE: str = str(LOGS_DIR / "experiment_structured.jsonl")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
FIGURE_DPI: int = 300  # Publication quality
FIGURE_FORMAT: str = "png"
FIGURE_STYLE: str = "seaborn-v0_8-whitegrid"
COLOR_CONSEQUENCE_AWARE: str = "#2196F3"   # Blue — our main agent
COLOR_RANDOM_BASELINE: str = "#FF9800"     # Orange — random baseline
COLOR_BUY_HOLD: str = "#4CAF50"            # Green — buy-and-hold
COLOR_PALETTE: List[str] = [
    "#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#607D8B", "#795548", "#CDDC39",
]

# ---------------------------------------------------------------------------
# Sensitivity Analysis Windows
# ---------------------------------------------------------------------------
# I test multiple training window lengths to assess robustness. If the result
# only holds for one specific window, that's a red flag I need to report.
SENSITIVITY_TRAIN_WINDOWS: List[Dict[str, str]] = [
    {"train_end": "2025-05-31", "label": "4y5m training"},
    {"train_end": "2025-06-30", "label": "4y6m training"},
    {"train_end": "2025-07-31", "label": "4y7m training (primary)"},
]

# ---------------------------------------------------------------------------
# Statistical Testing
# ---------------------------------------------------------------------------
SIGNIFICANCE_LEVEL: float = 0.05  # Standard alpha for t-tests
NUM_RANDOM_BASELINE_RUNS: int = 100  # I run random selection 100 times for stability
