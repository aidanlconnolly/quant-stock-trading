"""Ken French data library fetcher (FF5 daily factors).

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Cached to Parquet so repeated runs don't re-download.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import urllib.request

from backend.app.config import CACHE_DIR


FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


def _cache_path() -> Path:
    return CACHE_DIR / "ff5_daily.parquet"


def load_ff5_daily(use_cache: bool = True) -> pd.DataFrame:
    """Return daily FF5 factor returns as decimals (not percent), indexed by date.

    Columns: Mkt-RF, SMB, HML, RMW, CMA, RF. The raw CSV is in percent; we divide by 100.
    """
    path = _cache_path()
    if use_cache and path.exists():
        return pd.read_parquet(path)

    with urllib.request.urlopen(FF5_URL, timeout=60) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv"))
        text = zf.read(csv_name).decode("latin-1")

    # The CSV has a header preamble; the data section starts after a line of column headers
    # and ends before footnote text. Robust approach: find the line that looks like the
    # header, then parse until a blank line or non-numeric date.
    lines = text.splitlines()
    header_idx = next(
        i for i, line in enumerate(lines)
        if "Mkt-RF" in line and "SMB" in line and "HML" in line
    )
    data_lines: list[str] = [lines[header_idx]]
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            break
        first_field = stripped.split(",")[0].strip()
        if not first_field.isdigit() or len(first_field) != 8:
            break
        data_lines.append(line)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d")
    df = df.set_index(date_col).sort_index()
    df.index.name = "date"
    for c in FF5_COLUMNS:
        if c not in df.columns:
            raise RuntimeError(f"Expected column {c} not found in FF5 CSV: {list(df.columns)}")
    df = df[FF5_COLUMNS].astype(float) / 100.0

    df.to_parquet(path)
    return df
