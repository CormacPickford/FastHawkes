"""
Fetch top-N USDT pairs by 24hr quote volume, then download monthly
aggTrades archives for each from data.binance.vision.

Run this in your own environment (Colab / Azure VM / local) --
Binance endpoints are not reachable from this sandbox.
"""

import requests
import zipfile
import io
import os
from datetime import date, timedelta

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"
AGGTRADES_URL_TEMPLATE = (
    "https://data.binance.vision/data/spot/daily/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{day}.zip"
)

OUTPUT_DIR = "binance_data"
TOP_N = 100
NUM_DAYS = 1
END_DATE = date(2026, 7, 1)  # the single day to fetch; change as needed


def get_top_usdt_pairs(top_n=TOP_N):
    """Query live 24hr ticker, filter to USDT pairs, sort by quoteVolume."""
    resp = requests.get(TICKER_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    usdt_pairs = [d for d in data if d["symbol"].endswith("USDT")]
    usdt_pairs.sort(key=lambda d: float(d["quoteVolume"]), reverse=True)

    return [d["symbol"] for d in usdt_pairs[:top_n]]


def get_day_range(end_date=END_DATE, num_days=NUM_DAYS):
    """Return list of ISO date strings, num_days ending at end_date (inclusive)."""
    return [
        (end_date - timedelta(days=i)).isoformat()
        for i in range(num_days - 1, -1, -1)
    ]


def download_aggtrades_day(symbol, day, out_dir):
    """Download and extract one symbol's daily aggTrades CSV."""
    url = AGGTRADES_URL_TEMPLATE.format(symbol=symbol, day=day)
    resp = requests.get(url, timeout=60)

    if resp.status_code != 200:
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        zf.extract(csv_name, path=out_dir)
        extracted_path = os.path.join(out_dir, csv_name)

    return extracted_path


def main():
    print(f"Fetching top {TOP_N} USDT pairs by 24hr quote volume...")
    symbols = get_top_usdt_pairs(TOP_N)
    print(f"Got {len(symbols)} symbols. Top 5: {symbols[:5]}\n")

    days = get_day_range()
    print(f"Downloading {NUM_DAYS} days ({days[0]} to {days[-1]}) per symbol...\n")

    for symbol in symbols:
        symbol_dir = os.path.join(OUTPUT_DIR, symbol)
        os.makedirs(symbol_dir, exist_ok=True)

        ok_count = 0
        for day in days:
            path = download_aggtrades_day(symbol, day, symbol_dir)
            if path:
                ok_count += 1

        print(f"  {symbol}: {ok_count}/{len(days)} days downloaded -> {symbol_dir}/")

    print(f"\nDone. Files saved under '{OUTPUT_DIR}/<symbol>/'.")


if __name__ == "__main__":
    main()