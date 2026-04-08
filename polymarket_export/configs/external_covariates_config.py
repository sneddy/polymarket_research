from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalCovariateSpec:
    series_id: str
    provider: str
    provider_symbol: str
    interval: str
    category: str
    units: str
    description: str
    value_field: str = "close"
    include_in_default: bool = True


EXTERNAL_COVARIATE_SPECS: tuple[ExternalCovariateSpec, ...] = (
    ExternalCovariateSpec(
        series_id="btc_usd",
        provider="binance",
        provider_symbol="BTCUSDT",
        interval="5m",
        category="crypto",
        units="usd",
        description="Bitcoin spot price versus USD via Binance BTCUSDT klines.",
        value_field="close",
    ),
    ExternalCovariateSpec(
        series_id="eth_usd",
        provider="binance",
        provider_symbol="ETHUSDT",
        interval="5m",
        category="crypto",
        units="usd",
        description="Ether spot price versus USD via Binance ETHUSDT klines.",
        value_field="close",
    ),
    ExternalCovariateSpec(
        series_id="wti_oil_usd",
        provider="fred",
        provider_symbol="DCOILWTICO",
        interval="1d",
        category="commodity",
        units="usd_per_barrel",
        description="Crude Oil Prices: West Texas Intermediate daily spot price.",
        value_field="value",
    ),
    ExternalCovariateSpec(
        series_id="brent_oil_usd",
        provider="fred",
        provider_symbol="DCOILBRENTEU",
        interval="1d",
        category="commodity",
        units="usd_per_barrel",
        description="Crude Oil Prices: Brent Europe daily spot price.",
        value_field="value",
    ),
    ExternalCovariateSpec(
        series_id="us_10y_yield",
        provider="fred",
        provider_symbol="DGS10",
        interval="1d",
        category="rates",
        units="percent",
        description="Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity.",
        value_field="value",
    ),
    ExternalCovariateSpec(
        series_id="fed_funds_effective",
        provider="fred",
        provider_symbol="DFF",
        interval="1d",
        category="rates",
        units="percent",
        description="Effective Federal Funds Rate.",
        value_field="value",
    ),
    ExternalCovariateSpec(
        series_id="eur_usd",
        provider="fred",
        provider_symbol="DEXUSEU",
        interval="1d",
        category="fx",
        units="usd_per_eur",
        description="U.S. Dollars to One Euro exchange rate.",
        value_field="value",
    ),
    ExternalCovariateSpec(
        series_id="usd_jpy",
        provider="fred",
        provider_symbol="DEXJPUS",
        interval="1d",
        category="fx",
        units="jpy_per_usd",
        description="Japanese Yen to One U.S. Dollar exchange rate.",
        value_field="value",
    ),
    ExternalCovariateSpec(
        series_id="broad_usd_index",
        provider="fred",
        provider_symbol="DTWEXBGS",
        interval="1d",
        category="fx",
        units="index",
        description="Trade Weighted U.S. Dollar Index: Broad, Goods and Services.",
        value_field="value",
    ),
    ExternalCovariateSpec(
        series_id="edgar_total_filings",
        provider="sec_edgar",
        provider_symbol="*",
        interval="1d",
        category="finance",
        units="filings_per_day",
        description="Daily count of all SEC EDGAR form-index filings.",
        value_field="value",
        include_in_default=False,
    ),
    ExternalCovariateSpec(
        series_id="edgar_8k_filings",
        provider="sec_edgar",
        provider_symbol="8-K,8-K/A",
        interval="1d",
        category="finance",
        units="filings_per_day",
        description="Daily count of SEC EDGAR 8-K and 8-K/A filings.",
        value_field="value",
        include_in_default=False,
    ),
    ExternalCovariateSpec(
        series_id="edgar_10q_filings",
        provider="sec_edgar",
        provider_symbol="10-Q,10-Q/A",
        interval="1d",
        category="finance",
        units="filings_per_day",
        description="Daily count of SEC EDGAR 10-Q and 10-Q/A filings.",
        value_field="value",
        include_in_default=False,
    ),
    ExternalCovariateSpec(
        series_id="edgar_10k_filings",
        provider="sec_edgar",
        provider_symbol="10-K,10-K/A",
        interval="1d",
        category="finance",
        units="filings_per_day",
        description="Daily count of SEC EDGAR 10-K and 10-K/A filings.",
        value_field="value",
        include_in_default=False,
    ),
)


EXTERNAL_COVARIATE_REGISTRY: dict[str, ExternalCovariateSpec] = {
    spec.series_id: spec for spec in EXTERNAL_COVARIATE_SPECS
}


DEFAULT_EXTERNAL_SERIES: tuple[str, ...] = tuple(
    spec.series_id for spec in EXTERNAL_COVARIATE_SPECS if spec.include_in_default
)
