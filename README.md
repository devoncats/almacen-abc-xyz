# almacen — ABC/XYZ replenishment expert system

Rule-based expert system that assigns a replenishment policy to each item from its ABC/XYZ segment,
then validates the policies with a weekly lost-sales inventory simulation against a uniform baseline.

## Quick start

```bash
uv sync
uv run almacen run                                # real data in data/favorita/
uv run almacen generate                           # or: synthetic data -> data/synthetic/
uv run almacen run --data-dir data/synthetic
uv run almacen explain --run outputs/<run_id> --item <item_nbr>
uv run pytest
```

## Configuration

Everything is configured in `config/config.yaml`. The knowledge base (production rules) is kept
in `config/rules.yaml`. CLI flags override the config for a single run:
`--data-dir`, `--city`, `--test-weeks`, `--outliers`.

## Input data (Favorita layout)

`dataset.dir` must contain the three files of the Kaggle
[Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)
competition, with these columns (extra columns are ignored):

| File         | Columns                                              |
| ------------ | ---------------------------------------------------- |
| `train.csv`  | `date, store_nbr, item_nbr, unit_sales, onpromotion` |
| `items.csv`  | `item_nbr, family, class, perishable`                |
| `stores.csv` | `store_nbr, city, type`                              |

Any other dataset can be used if it is shaped like this. File names go in `dataset.files`, and
headers can be renamed with `dataset.columns` (e.g. `{qty: unit_sales}`).

`train.csv` is joined with `items.csv` on `item_nbr` and with `stores.csv` on `store_nbr`. It is read
in chunks of `dataset.chunk_rows`, and only the stores of `dataset.city` and the dates from
`dataset.date_from` are kept in memory. The full Kaggle file (~125M rows) takes about 3 minutes for
Quito. Sales whose item or store is missing from the lookup tables are dropped, and the count is
reported in `run.json` (`rows_unmatched_<table>`). Empty `onpromotion` values are read as `False`.

`almacen generate` writes synthetic data in this same layout, so one config serves both.

## Outlier capping

`outliers.method` in the config (or `--outliers none|percentile|mad` on `almacen run`) caps each
item's weekly TRAINING demand from above before segmentation and parameter estimation; the test
window is always simulated with real demand. Capped weeks are listed in `tables/outliers_capped.*`
and summarized in `run.json`. On Favorita (Quito), MAD capping catches the April 2016 earthquake
spikes but also recurring Christmas peaks, and it lowered fill rate while reducing inventory, so it
is off by default.

## Knowledge base

Rules live in `config/rules.yaml` (format documented at the top of the file). Segment rules create
the base policy; modifier rules (new item, perishable, weighted, discontinuation) adjust it.
Conflicts are resolved by salience, and each rule fires at most once per item.

## Outputs (`outputs/<run_id>/`)

| Path                                               | Content                                                                               |
| -------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `tables/metrics_summary.*`                         | Expert vs. baseline: fill rate, stockouts, average inventory, days of cover           |
| `tables/metrics_by_segment.*`, `metrics_by_abc.*`  | Same metrics per segment / ABC class                                                  |
| `tables/abc_xyz_matrix.*`, `segmentation.*`        | Segmentation results                                                                  |
| `tables/policy_by_segment.*`, `recommendations*.*` | Inferred policies and parameters (s, S, Q, SS)                                        |
| `charts/*.png`                                     | Pareto curve, ABC/XYZ heatmap, CV vs. volume, metrics by segment, inventory timelines |
| `explanations.jsonl`                               | Fired-rule trace per item                                                             |
| `run.json`                                         | Config, preprocessing report, periods, rules, overall metrics                         |
