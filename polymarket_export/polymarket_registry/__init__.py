"""Registry helpers for Polymarket market metadata tables."""

__all__ = [
    "build_pending_queue_all",
    "build_yes_probability_series_5m",
    "ensure_schema",
    "store_market_dataset",
    "table_exists",
]


def __getattr__(name: str):
    if name in {"ensure_schema", "table_exists"}:
        from polymarket_registry.schema import ensure_schema, table_exists

        return {
            "ensure_schema": ensure_schema,
            "table_exists": table_exists,
        }[name]

    if name in {"build_pending_queue_all", "build_yes_probability_series_5m", "store_market_dataset"}:
        from polymarket_registry.history import (
            build_pending_queue_all,
            build_yes_probability_series_5m,
            store_market_dataset,
        )

        return {
            "build_pending_queue_all": build_pending_queue_all,
            "build_yes_probability_series_5m": build_yes_probability_series_5m,
            "store_market_dataset": store_market_dataset,
        }[name]

    raise AttributeError(f"module 'polymarket_registry' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
