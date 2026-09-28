import yfinance as yf
from datetime import datetime, timedelta

from config import DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Top 200 by market cap
popular_stocks = [
    # Mega Cap & Growth
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "UNH", "JNJ", "V",
    "WMT", "JPM", "PG", "XOM", "KO", "MA", "HD", "CVX", "MRK", "ABBV",
    "PEP", "COST", "CSCO", "ACN", "CRM", "ABT", "DIS", "TMO", "MCD", "NFLX",
    "INTC", "SCHW", "IBM", "QCOM", "LLY", "GE", "BA", "ISRG", "CAT", "AMD",
    "ORCL", "TMUS", "WFC", "BAC", "AMGN", "T", "VZ", "NEE", "PM", "PGR",

    # Industrials, Tech & Semi
    "HON", "SYK", "AXP", "TXN", "INTU", "AMAT", "AZO", "LRCX", "BKNG", "NOW",
    "MU", "PANW", "ADBE", "SNPS", "CDNS", "KLAC", "ASML", "ADSK", "DE", "LMT",
    "RTX", "NOC", "TDG", "GD", "PLTR", "ANET", "ADI", "SNOW", "SHOP", "UBER",
    "ABNB", "MELI", "PYPL", "WDAY", "TEAM", "DDOG", "CRWD", "MDB", "NET", "OKTA",
    "ZS", "SE", "U", "FISV", "TTD", "AVGO", "AMR", "STX", "WDC", "NXPI",

    # Finance, Health & Consumables
    "MS", "LOW", "SPGI", "ELV", "GS", "TJX", "MDLZ", "C", "REGN", "MMC",
    "CB", "BSX", "ETN", "CI", "MCO", "AON", "AJG", "TRV", "MET", "PRU",
    "AFL", "ALL", "STT", "BK", "DHR", "ZTS", "IDXX", "IQV", "RVTY", "A",
    "MTD", "WAT", "STE", "SHW", "APD", "ECL", "FCX", "NEM", "CTVA", "DOW",
    "NUE", "ALB", "FMC", "UPS", "FDX", "NSC", "CSX", "UNP", "WM", "RSG",

    # Energy, REITs & Utilities
    "COP", "BP", "SHEL", "TTE", "SLB", "EOG", "MPC", "PSX", "VLO", "PXD",
    "CCI", "PLD", "EQIX", "DLR", "WELL", "PSA", "AVB", "EQR", "AMT", "VICI",
    "STAG", "SBAC", "NVR", "DHI", "KMB", "SPG", "O", "VTR", "VRTX", "ILMN",
    "BIIB", "CTSH", "LIN", "CVS", "PFE", "GILD", "EW", "SJM", "CPB", "KHC",
    "ADM", "MOS", "CF", "MTCH", "PINS", "Z", "URI", "PWR", "D", "AEP",
]

popular_stocks = sorted(set(popular_stocks))
print(f"Total unique stocks: {len(popular_stocks)}")

# 10 years + warmup
end_date = datetime(2026, 1, 1)
start_date = end_date - timedelta(days=365 * 10 + 180)
print(f"Downloading data from {start_date.date()} to {end_date.date()}...")

failed_stocks = []
for i, stock in enumerate(popular_stocks, 1):
    try:
        print(f"[{i}/{len(popular_stocks)}] Downloading {stock}...", end=" ")
        df = yf.download(
            stock, start=start_date, end=end_date, progress=False, auto_adjust=False
        )
        if df.empty:
            raise ValueError("no rows returned")
        df = df.reset_index()
        df.to_parquet(DATA_DIR / f"{stock}.parquet")
        print("done")
    except Exception as e:
        print(f"failed ({e})")
        failed_stocks.append(stock)

print(f"\nDownloaded {len(popular_stocks) - len(failed_stocks)}/{len(popular_stocks)} stocks")
if failed_stocks:
    print(f"Failed: {', '.join(failed_stocks)}")