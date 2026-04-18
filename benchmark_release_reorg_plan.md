# План Реорганизации Benchmark-Слоя Для Релиза

## Жесткие ограничения

- Все data layers остаются в текущем internal package.
- `raw` source access, DB-backed ingestion, build/save `CanonicalDataset` из raw data остаются в internal слое.
- `polymarket_research.benchmarks` начинается строго на canonical boundary.
- В `benchmarks` попадает только release-facing логика:
  - benchmark schemas
  - task materialization from `CanonicalDataset`
  - frozen artifact loading
  - evaluators
  - reference baselines
- Legacy `views` и notebook helpers не переезжают в benchmark package.
- Backward-compatibility import shims не делаем.
- Dependency direction должна быть строгой:
  - `raw/data -> canonical -> benchmarks`
- Legacy tests не сохраняем. Нужны только новые release-oriented tests.

## Что это означает для текущего репозитория

- [polymarket_research/data/canonical/dataset.py](/Users/sneddy/research/polymarket_research/polymarket_research/data/canonical/dataset.py) остается canonical boundary и internal substrate.
- [polymarket_research/data/raw/dataset.py](/Users/sneddy/research/polymarket_research/polymarket_research/data/raw/dataset.py) остается internal ingestion API.
- `polymarket_research/benchmarks/` должен стать узким release package, а не смесью protocol objects, dataset helpers, notebook utilities и transport-слоя.
- [polymarket_research/views/benchmark.py](/Users/sneddy/research/polymarket_research/polymarket_research/views/benchmark.py) не должен вливаться в release package; он либо остается legacy/internal, либо удаляется отдельно, но не становится частью benchmark API.
- `tasks/` и `data/representations/` сейчас содержат логику, которую можно использовать для materialization from canonical, но она должна жить под benchmark-facing структурой, а не как отдельный публичный слой.

## Целевой package boundary

```text
polymarket_research/
  data/                         # internal only
    raw/
    canonical/
    representations/           # либо internal feature helpers, либо постепенно убрать

  benchmarks/                  # release-facing package, starts at CanonicalDataset
    __init__.py
    schemas.py
    terminal.py
    decisiveness.py
    repricing.py
    loaders.py
    evaluators.py
    builders/
      __init__.py
      terminal.py
      decisiveness.py
      repricing.py
      common.py
    baselines/
      __init__.py
      terminal.py
      decisiveness.py
      repricing.py
    integrations/
      huggingface.py           # optional, only if you want transport in release surface
```

Ключевой момент: `data/` не реорганизуется ради release API. Реорганизуется только benchmark-facing слой.

## Целевой публичный API

```python
from polymarket_research.benchmarks import (
    TerminalBenchmark,
    DecisivenessBenchmark,
    RepricingBenchmark,
    load_terminal,
    load_decisiveness,
    load_repricing,
    evaluate_terminal,
    evaluate_decisiveness,
    evaluate_repricing,
)
```

Canonical build path остается отдельным internal workflow:

```python
from polymarket_research.data.canonical import CanonicalDatasetBuilder
from polymarket_research.data.raw import RawMarketHandle
```

Benchmark package не должен становиться entrypoint для raw loading или canonical building.

## Что должно остаться вне benchmark package

- `RawMarketHandle`
- `RawMarketBundle`
- `RawExternalCovariates`
- `RawMarketSnapshot`
- `CanonicalDatasetBuilder`
- любые SQLite/db helpers
- notebook-oriented helpers
- legacy `views`
- ad hoc feature utilities, если они нужны только ноутбукам и не являются частью release contract

## Что должно войти в benchmark package

### 1. Benchmark schemas

Нужны явные release-facing структуры данных:

- `TerminalBenchmark`
- `DecisivenessBenchmark`
- `RepricingBenchmark`

Они должны описывать:

- frozen `examples`
- frozen `market_timeseries`
- task metadata
- manifest/build config
- save/load contract
- evaluate contract

### 2. Task materialization from `CanonicalDataset`

Нужны builder-функции внутри benchmark package:

- `build_terminal_from_canonical(canonical, config)`
- `build_decisiveness_from_canonical(canonical, config)`
- `build_repricing_from_canonical(canonical, config)`

Они и есть граница `canonical -> benchmarks`.

### 3. Frozen artifact loading

Нужны top-level loaders:

- `load_terminal(path)`
- `load_decisiveness(path)`
- `load_repricing(path)`

Они должны читать только frozen release artifacts, а не raw DB и не canonical caches.

### 4. Evaluators

Нужны top-level evaluators:

- `evaluate_terminal(benchmark, predictions, ...)`
- `evaluate_decisiveness(benchmark, predictions, ...)`
- `evaluate_repricing(benchmark, predictions, ...)`

### 5. Reference baselines

Нужен отдельный benchmark-local baseline layer:

- `benchmarks.baselines.terminal`
- `benchmarks.baselines.decisiveness`
- `benchmarks.baselines.repricing`

Правило: baselines работают только поверх public benchmark objects или frozen artifacts.

## Пошаговый план

### 1. Зафиксировать release contract

Перед любыми переносами нужно письменно заморозить benchmark contract.

Зафиксировать:

- какие три task objects релизятся
- какие поля обязаны быть в `manifest.json`
- какие parquet-файлы обязаны существовать
- какие loader/evaluator entrypoints считаются стабильными
- что benchmark package не умеет:
  - не ходит в SQLite
  - не строит canonical из raw
  - не экспортирует notebook views

Минимум для каждого bundle:

- `manifest.json`
- `examples.parquet`
- `market_timeseries.parquet`
- `README.md`

### 2. Сузить `polymarket_research.benchmarks` до release-facing кода

Из `benchmarks/` нужно убрать все, что не является частью release contract.

Оставить в package:

- task schemas
- build-from-canonical logic
- frozen loaders
- evaluators
- reference baselines

Не держать в public surface:

- `dataset_utils.py`
- `covariate_utils.py`
- случайные notebook helpers
- generic tabular facades, если они не нужны как часть release contract

Если какой-то код из этих файлов нужен benchmark builders, его надо либо:

- встроить в `benchmarks/builders/*`,
- либо оставить internal helper внутри `benchmarks`, но не экспортировать с верхнего уровня.

### 3. Очистить `CanonicalDataset` до роли canonical boundary

`CanonicalDataset` остается там, где он сейчас живет, но его API надо сделать односторонним.

Нужно:

- оставить в нем только canonical responsibilities:
  - canonical tables
  - `summary()`
  - `status()`
  - `save()`
  - `from_parquet()`
- убрать методы, которые поднимаются вверх в benchmark/view слой:
  - `repricing_panel(...)`
  - `decisiveness_benchmark_view(...)`
  - `decisiveness_benchmark(...)`
  - `repricing_benchmark_view(...)`
  - `repricing_benchmark(...)`
  - `*_ml_benchmark(...)`

После этого layering должен быть односторонним:

`raw/data -> canonical -> benchmarks`

То есть benchmark layer зависит на canonical boundary, а internal data layer не зависит от benchmark package.

### 4. Перенести task materialization в `benchmarks/builders`

Сейчас build logic размазана между:

- `polymarket_research/tasks/*`
- `polymarket_research/data/representations/*`
- частично самими benchmark classes

Нужно собрать ее в одном месте:

- `polymarket_research/benchmarks/builders/terminal.py`
- `polymarket_research/benchmarks/builders/decisiveness.py`
- `polymarket_research/benchmarks/builders/repricing.py`

Там должны жить только функции materialization from canonical.

Рекомендация по переносу:

- оставить внутренние feature helpers маленькими и task-specific
- не переносить notebook-facing derived views
- не тянуть в builders paper-facing summary abstractions

### 5. Сделать benchmark classes thin и release-oriented

`TerminalBenchmark`, `DecisivenessBenchmark`, `RepricingBenchmark` должны быть тонкими schema objects вокруг frozen contract.

Они должны уметь:

- `build(...)` через соответствующий builder
- `save(...)`
- `load(...)`
- `manifest()`
- `targets(...)`
- `evaluate(...)`

Они не должны:

- знать про raw data access
- знать про notebook views
- знать про альтернативные non-release abstractions

### 6. Ввести явные top-level loaders/evaluators

Даже если классы уже имеют `.load()` и `.evaluate()`, публичный API лучше сделать через top-level функции:

- `load_terminal(...)`
- `load_decisiveness(...)`
- `load_repricing(...)`
- `evaluate_terminal(...)`
- `evaluate_decisiveness(...)`
- `evaluate_repricing(...)`

Документация должна вести пользователя именно в эти entrypoints.

### 7. Не переносить `views/` и notebook helpers

Это отдельное решение, его надо сделать явно, чтобы потом не размыть release boundary.

Нужно:

- не переносить [polymarket_research/views/benchmark.py](/Users/sneddy/research/polymarket_research/polymarket_research/views/benchmark.py) в `benchmarks`
- не тащить старые view schemas в release package
- не экспортировать derived notebook-friendly tables как часть benchmark API

Если какой-то код из `views/` реально нужен release layer, его надо:

- переписать заново в терминах release contract,
- а не импортировать legacy view слой в новый benchmark package.

### 8. Вынести reference baselines в `benchmarks/baselines`

Reference baselines должны быть частью release story, но не частью data layer.

Нужно создать:

- `polymarket_research/benchmarks/baselines/terminal.py`
- `polymarket_research/benchmarks/baselines/decisiveness.py`
- `polymarket_research/benchmarks/baselines/repricing.py`

Минимальный baseline suite:

- terminal:
  - market-price baseline
  - один простой learned/tabular baseline
- decisiveness:
  - threshold-distance heuristic
  - один простой learned baseline
- repricing:
  - volatility heuristic
  - logistic/tree baseline

Правила:

- baseline code не читает raw DB
- baseline code не зависит от `CanonicalDatasetBuilder`
- baseline code не тянет legacy notebook helpers

### 9. Зафиксировать release artifact layout

Нужно стандартизовать layout frozen benchmark artifacts и не смешивать его с internal running caches.

Рекомендация:

```text
frozen_notebooks/running_artefacts/
  polymarket/
    terminal/
      v1/
    decisiveness/
      v1/
    repricing/
      v1/
```

Внутри каждого task bundle:

- `manifest.json`
- `examples.parquet`
- `market_timeseries.parquet`
- `README.md`

Опционально:

- `baseline_reports/*.json`

Не включать в public release layout:

- raw snapshots
- canonical snapshots
- notebook-only views

### 10. Переписать package exports без compatibility shims

Это должен быть clean break.

Нужно:

- переписать `polymarket_research/benchmarks/__init__.py` под новый narrow API
- убрать старые re-exports из `benchmarks/__init__.py`
- не делать compatibility imports
- не оставлять deprecated shims

Если старые пути перестают работать, это приемлемо для этого релизного рефакторинга.

### 11. Переписать документацию под release boundary

README и usage examples должны отражать новый boundary буквально.

Нужно:

- в [README.md](/Users/sneddy/research/polymarket_research/README.md) сделать benchmark usage первым benchmark-facing сценарием
- явно разделить:
  - internal data generation workflow
  - public benchmark consumption workflow
- показать только новый benchmark API:
  - `load_*`
  - `evaluate_*`
  - reference baselines

Не надо в release docs:

- вести пользователя в raw handles
- рекламировать canonical build path как основной public entrypoint
- документировать legacy views

### 12. Написать новые release-oriented tests

Legacy tests не переносим. Нужны только tests под новый benchmark contract.

Минимальный набор:

- build/save/load round-trip для каждого benchmark task
- evaluator coverage:
  - полный coverage prediction ids
  - понятный fail на missing predictions
- split integrity:
  - строгий out-of-time split
- leakage safety:
  - benchmark builders не используют post-cutoff information
- manifest stability:
  - обязательные поля всегда присутствуют

## Точный mapping текущих модулей

| Текущее место | Целевое место | Комментарий |
| --- | --- | --- |
| `polymarket_research/data/canonical/dataset.py` | остается на месте | canonical boundary, internal |
| `polymarket_research/data/raw/dataset.py` | остается на месте | raw/data ingestion, internal |
| `polymarket_research/tasks/terminal.py` | `polymarket_research/benchmarks/builders/terminal.py` | materialization from canonical |
| `polymarket_research/tasks/decisiveness.py` | `polymarket_research/benchmarks/builders/decisiveness.py` | materialization from canonical |
| `polymarket_research/tasks/repricing.py` | `polymarket_research/benchmarks/builders/repricing.py` | materialization from canonical |
| `polymarket_research/tasks/base.py` | `polymarket_research/benchmarks/builders/common.py` или удалить | только если реально нужен |
| `polymarket_research/data/representations/terminal.py` | частично перенести в `benchmarks/builders/terminal.py` | только release-relevant materialization logic |
| `polymarket_research/data/representations/repricing.py` | частично перенести в `benchmarks/builders/repricing.py` | только release-relevant materialization logic |
| `polymarket_research/data/representations/context.py` | частично встроить в relevant builders | не делать отдельным public layer |
| `polymarket_research/data/representations/external.py` | частично встроить в relevant builders | только если нужно release task materialization |
| `polymarket_research/benchmarks/common.py` | `polymarket_research/benchmarks/evaluators.py` + internal metric helpers | release-facing evaluation API |
| `polymarket_research/benchmarks/terminal.py` | оставить, но сузить до schema object | release-facing |
| `polymarket_research/benchmarks/decisiveness.py` | оставить, но сузить до schema object | release-facing |
| `polymarket_research/benchmarks/repricing.py` | оставить, но сузить до schema object | release-facing |
| `polymarket_research/benchmarks/huggingface.py` | `polymarket_research/benchmarks/integrations/huggingface.py` или удалить | optional integration |
| `polymarket_research/views/benchmark.py` | не переносить в benchmark package | legacy/internal only |

## Порядок реализации

### Волна 1. Зафиксировать benchmark boundary

1. Заморозить release contract.
2. Очистить `benchmarks/__init__.py`.
3. Добавить `load_*` и `evaluate_*`.
4. Убрать benchmark-oriented convenience methods из `CanonicalDataset`.

### Волна 2. Собрать materialization logic под benchmark package

1. Создать `benchmarks/builders/*`.
2. Перенести туда task materialization из `tasks/*`.
3. Аккуратно втянуть только нужные куски из `data/representations/*`.
4. Не переносить `views/` и notebook abstractions.

### Волна 3. Довести release surface

1. Вынести baselines в `benchmarks/baselines`.
2. Зафиксировать artifact layout и manifest schema.
3. Переписать README/examples.
4. Написать новые release-oriented tests.

## Критерии готовности

Рефакторинг можно считать завершенным, когда выполняются все условия:

- benchmark package начинаетcя от `CanonicalDataset` и выше
- benchmark package не умеет читать raw DB или строить canonical dataset
- `CanonicalDataset` не импортирует benchmark/view/baseline слои
- `benchmarks` содержит только release-facing logic
- `views` и notebook helpers не попали в benchmark package
- public API целиком помещается в `polymarket_research.benchmarks`
- frozen artifacts имеют стабильный manifest contract
- baselines работают только от benchmark objects / frozen artifacts
- документация отражает новый boundary без legacy paths
- tests покрывают только новый release contract

## Практическая рекомендация

Делать это стоит как clean release refactor, а не как миграцию с длительной поддержкой старых путей. Самая частая ошибка в такой задаче — протащить в новый benchmark package старые удобные abstraction layers, notebook views и compatibility exports. Это почти гарантированно размоет границу. Здесь лучше жестко оставить `data` internal, а `benchmarks` сделать маленьким, односторонним и release-oriented.
