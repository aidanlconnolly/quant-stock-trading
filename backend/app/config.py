from pathlib import Path

LIVE_TRADING_ENABLED: bool = False

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SLIPPAGE_BPS = 5
BORROW_RATE_ANNUAL = 0.0025
TRADING_DAYS_PER_YEAR = 252
DEFAULT_INIT_CASH = 100_000.0
TARGET_VOL_ANNUAL = 0.10
MAX_POSITION_PCT = 0.05
MAX_DRAWDOWN_CIRCUIT = 0.15
DRAWDOWN_LOOKBACK_DAYS = 30
