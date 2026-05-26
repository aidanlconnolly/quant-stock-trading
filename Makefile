.PHONY: setup test backtest momentum factor notebook lint clean

setup:
	uv sync

test:
	uv run pytest -v

backtest:
	uv run python -m scripts.run_backtest --pair KO PEP --start 2014-01-01 --end 2024-12-31

momentum:
	uv run python -m scripts.run_momentum --start 2014-01-01 --end 2024-12-31

factor:
	uv run python -m scripts.run_factor_tilt --start 2014-01-01 --end 2024-12-31 --tilt-factor HML

notebook:
	uv run jupyter lab notebooks/

lint:
	uv run ruff check .

clean:
	rm -rf .venv .pytest_cache .ruff_cache data/cache/*.parquet data/results/*.parquet
