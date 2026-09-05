"""Data-loading adapters for local CSV files and Yahoo Finance."""

from pathlib import Path

import pandas as pd
import yfinance as yf


class CsvDataHandler:
    def __init__(self, datasources, data_dir="data"):
        self.datasources = datasources
        self.data_dir = Path(data_dir)

    def getDataFromSource(self, source, formatOut="dataframe"):
        try:
            filename = self.datasources[source]
        except KeyError as exc:
            raise KeyError(f"Unknown data source: {source}") from exc

        path = self.data_dir / filename
        try:
            data = pd.read_csv(path, index_col=0, parse_dates=True)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise ValueError(f"Cannot read source {path}") from exc
        return self._format(data, formatOut)

    @staticmethod
    def _format(data, formatOut):
        result = data if formatOut.lower() == "dataframe" else data.to_dict(orient="index")
        if formatOut.lower() not in {"dataframe", "dictionary"}:
            raise ValueError("formatOut must be 'dataframe' or 'dictionary'")
        return data.index, result


class YahooFinanceDataHandler:
    def __init__(self, tickers):
        self.tickers = tickers

    def getDataFromSource(self, startDate, endDate, columns=("Adj Close",), formatOut="dataframe"):
        data = yf.download(self.tickers, start=startDate, end=endDate)
        data = data[list(columns)]
        # Naive timestamps keep Portfolio date keys consistent across sources.
        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)
        if isinstance(data.columns, pd.MultiIndex):
            # Columns arrive as (price field, ticker). When a single field was
            # requested, collapse to ticker-only columns. When multiple fields
            # were requested for multiple tickers, keep the (field, ticker)
            # MultiIndex instead of dropping the field level (which would
            # produce duplicated ticker columns).
            fields = data.columns.get_level_values(0).unique()
            if len(fields) == 1:
                data.columns = data.columns.droplevel(0)
        return CsvDataHandler._format(data, formatOut)


# Backwards-compatible names for the original public API.
csvDataHandler = CsvDataHandler
yfDataHandler = YahooFinanceDataHandler
