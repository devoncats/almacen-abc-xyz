# almacen — ABC/XYZ replenishment expert system

Rule-based expert system that assigns a replenishment policy to each item from its ABC/XYZ segment,
then validates the policies with a weekly lost-sales inventory simulation against a uniform baseline
and writes a decision report for the warehouse manager. The segment can come from the classic
ABC/XYZ cuts or from a LightGBM classifier (`--segmenter`).

## Quick start

```bash
uv sync
uv run almacen run                                # real data in data/favorita/
uv run almacen generate                           # or: synthetic data -> data/synthetic/
uv run almacen run --data-dir data/synthetic
uv run almacen run --segmenter lightgbm           # segments predicted by LightGBM
uv run almacen explain --run outputs/<run_id> --item <item_nbr>
uv run pytest
```

## Configuration

Everything is configured in `config/config.yaml`. The knowledge base (production rules) is kept
in `config/rules.yaml`. CLI flags override the config for a single run:
`--data-dir`, `--city`, `--test-weeks`, `--outliers`, `--segmenter`.

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

## Segmenters

The expert system consumes one of nine segments (AX ... CZ) per item. Who assigns it is set by
`segmenter.method` (or `--segmenter`):

- `abc-xyz` (default): classic cuts on the training window, ABC by cumulative volume share and XYZ
  by the coefficient of variation of weekly demand.
- `lightgbm`: a multiclass `LGBMClassifier` that predicts the segment each item will have over the
  next `simulation.test_weeks` weeks. Training samples come from rolling origins inside the training
  window (every `stride_weeks`): features summarize the previous `window_weeks` of demand (volume
  rank and share, mean, std, CV, recent CV and trend, share of zero weeks, peak ratio, promotion
  share, age, family, perishable, weighted, current classic class), and the label is the classic
  segment of the following weeks. The last origin is held out to measure the model
  (`tables/model_metrics*`, `confusion_matrix`, `feature_importance` and the `model_*.png` charts);
  the model is then refitted on all origins and applied at the training cutoff. The test window is
  never used for training.

In both modes the policy parameters use the item's historical weekly mean and standard deviation,
so only the segment differs. The probability of the assigned segment is passed to the rules as
`seg_confidence` (1.0 for the classic method); `R-LOWCONF` sends uncertain items to manual
validation.

To add another model (random forest, CV-only, ...), subclass `Segmenter` in
`src/almacen/segmenters/` (return the segment columns plus `confidence`) and add it to `SEGMENTERS`.

## Knowledge base

Rules live in `config/rules.yaml` (format documented at the top of the file). Segment rules create
the base policy; modifier rules (new item, uncertain ML segment, perishable, weighted,
discontinuation) adjust it.
Conflicts are resolved by salience, and each rule fires at most once per item.

## Recommendation for the warehouse manager

Each run writes `recommendation.md` and prints its verdict. Every segment is judged on simulated
fill rate and average inventory against the uniform baseline (`decision` in the config): ADOPT
when the fill rate drops at most `fill_rate_tolerance_pp` while inventory does not grow, or the
fill rate improves with at most `max_inventory_increase_pct` more inventory; REVIEW when the
trade-off is larger in either direction; otherwise KEEP BASELINE. The overall decision (ADOPT,
ADOPT SELECTIVELY, REVIEW, KEEP BASELINE) combines the warehouse-level result with the segments.
The report also lists the items for manual validation, uncertain segments, discontinuation
candidates and new items, the parameters of the top items, and the model quality in LightGBM mode.

## Outputs (`outputs/<run_id>/`)

| Path                                               | Content                                                                               |
| -------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `tables/metrics_summary.*`                         | Expert vs. baseline: fill rate, stockouts, average inventory, days of cover           |
| `tables/metrics_by_segment.*`, `metrics_by_abc.*`  | Same metrics per segment / ABC class                                                  |
| `tables/abc_xyz_matrix.*`, `segmentation.*`        | Segmentation results                                                                  |
| `tables/policy_by_segment.*`, `recommendations*.*` | Inferred policies and parameters (s, S, Q, SS)                                        |
| `tables/decision_by_segment.*`                     | Verdict per segment (adopt / review / keep baseline) and its reason                   |
| `tables/model_*`, `confusion_matrix.*`, `feature_importance.*` | LightGBM only: validation metrics of the segment classifier               |
| `charts/*.png`                                     | Pareto curve, ABC/XYZ heatmap, CV vs. volume, metrics by segment, inventory timelines (+ confusion matrix and feature importance with LightGBM) |
| `recommendation.md`                                | Decision report for the warehouse manager                                             |
| `explanations.jsonl`                               | Fired-rule trace per item                                                             |
| `run.json`                                         | Config, segmenter and its diagnostics, decision, preprocessing report, periods, rules, overall metrics |
