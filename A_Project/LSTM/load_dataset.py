import yfinance as yf


def load_dataset(year: int = 2000, ticker: str = "GC=F"):
    data = yf.download(ticker, start=f"{year}-01-01")

    data = data.reset_index()
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
    data = data.rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

    data.columns = data.columns.droplevel(1)

    data.to_csv("data/data.csv", index=False)

    return "data/data.csv"
