"""Registry helpers for Polymarket market metadata tables."""

__all__ = [
    "build_pending_queue",
    "build_yes_probability_series_5m",
    "ensure_schema",
    "load_markets_for_category",
    "refresh_market_registry",
    "refresh_market_registry_all_categories",
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

    if name in {"load_markets_for_category"}:
        from polymarket_registry.upsert import load_markets_for_category

        return {
            "load_markets_for_category": load_markets_for_category,
        }[name]

    if name in {"refresh_market_registry", "refresh_market_registry_all_categories"}:
        from polymarket_registry.refresh import refresh_market_registry, refresh_market_registry_all_categories

        return {
            "refresh_market_registry": refresh_market_registry,
            "refresh_market_registry_all_categories": refresh_market_registry_all_categories,
        }[name]

    if name in {"build_pending_queue", "build_yes_probability_series_5m", "store_market_dataset"}:
        from polymarket_registry.history import (
            build_pending_queue,
            build_yes_probability_series_5m,
            store_market_dataset,
        )

        return {
            "build_pending_queue": build_pending_queue,
            "build_yes_probability_series_5m": build_yes_probability_series_5m,
            "store_market_dataset": store_market_dataset,
        }[name]

    raise AttributeError(f"module 'polymarket_registry' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
