"""LightGBM segmenter: predicts the ABC/XYZ segment each item will have over the coming horizon.

Training samples come from rolling origins inside the training window only. At origin t the
features describe the item's demand before t, and the label is the classic ABC/XYZ segment of
the following `horizon` weeks (the simulation horizon). The last origin is held out to measure
model quality; the model is then refitted on every origin and applied at the train cutoff.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from almacen.config import AppConfig
from almacen.schemas import SchemaError
from almacen.segmentation import (
    ABC_ORDER,
    SEGMENT_ORDER,
    XYZ_ORDER,
    classify_abc,
    classify_xyz,
    segment,
)
from almacen.segmenters.base import RESULT_COLUMNS, SegmentationResult, Segmenter

NUMERIC_FEATURES = [
    "volume_rank",  # cumulative volume share preceding the item (0 = top seller)
    "share",
    "mean_w",
    "std_w",
    "cv",
    "cv_13",
    "trend_4",  # mean of the last 4 weeks / window mean
    "trend_13",
    "zero_share",  # share of weeks without sales
    "peak_ratio",  # max week / mean week
    "promo_share",
    "history_weeks",
    "age_weeks",
    "abc_now",  # classic class on the feature window, as ordinal
    "xyz_now",
    "perishable",
    "is_weighted",
]
FEATURES = [*NUMERIC_FEATURES, "family"]


def build_features(
    weekly: pd.DataFrame,
    items: pd.DataFrame,
    origin: pd.Timestamp,
    window_weeks: int,
    families: list[str],
    cfg: AppConfig,
) -> pd.DataFrame:
    """One row per item with history before `origin`; features use at most `window_weeks`."""
    hist = weekly[weekly["week"] < origin]
    window = hist[hist["week"] >= origin - pd.Timedelta(weeks=window_weeks)]
    if window.empty:
        return pd.DataFrame(columns=["item_nbr", *FEATURES])

    abc = classify_abc(window, cfg.abc)
    if abc["total_units"].sum() <= 0:
        return pd.DataFrame(columns=["item_nbr", *FEATURES])
    abc["volume_rank"] = abc["cum_share"] - abc["share"]
    xyz = classify_xyz(window, cfg.xyz)
    feats = abc[["item_nbr", "share", "volume_rank", "abc"]].merge(
        xyz[["item_nbr", "mean_w", "std_w", "cv", "xyz"]], on="item_nbr"
    )

    g = window.groupby("item_nbr")
    stats = pd.DataFrame(
        {
            "zero_share": g["units"].apply(lambda s: float((s <= 0).mean())),
            "peak_ratio_max": g["units"].max(),
            "promo_share": g["promo_share"].mean(),
            "history_weeks": g["week"].size(),
        }
    )
    last4 = window[window["week"] >= origin - pd.Timedelta(weeks=4)]
    last13 = window[window["week"] >= origin - pd.Timedelta(weeks=13)].groupby(
        "item_nbr"
    )["units"]
    stats["mean_4"] = last4.groupby("item_nbr")["units"].mean()
    stats["mean_13"] = last13.mean()
    stats["std_13"] = last13.std()
    feats = feats.merge(stats.reset_index(), on="item_nbr", how="left")

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = feats["mean_w"].where(feats["mean_w"] > 0)
        feats["trend_4"] = feats["mean_4"] / mean
        feats["trend_13"] = feats["mean_13"] / mean
        feats["peak_ratio"] = feats["peak_ratio_max"] / mean
        feats["cv_13"] = feats["std_13"] / feats["mean_13"].where(feats["mean_13"] > 0)
    feats["cv"] = feats["cv"].replace([np.inf, -np.inf], np.nan)
    feats["abc_now"] = feats["abc"].map({c: i for i, c in enumerate(ABC_ORDER)})
    feats["xyz_now"] = feats["xyz"].map({c: i for i, c in enumerate(XYZ_ORDER)})

    attrs = items[["item_nbr", "family", "perishable", "is_weighted", "first_week"]]
    feats = feats.merge(attrs, on="item_nbr", how="left")
    feats["age_weeks"] = (origin - feats["first_week"]).dt.days // 7
    feats["perishable"] = feats["perishable"].astype(float)
    feats["is_weighted"] = feats["is_weighted"].astype(float)
    feats["family"] = pd.Categorical(feats["family"], categories=families)
    return feats[["item_nbr", *FEATURES]]


def future_labels(
    weekly: pd.DataFrame, origin: pd.Timestamp, horizon: int, cfg: AppConfig
) -> pd.Series:
    """Classic segment of each item over [origin, origin + horizon), as a SEGMENT_ORDER index."""
    future = weekly[
        (weekly["week"] >= origin)
        & (weekly["week"] < origin + pd.Timedelta(weeks=horizon))
    ]
    seg = segment(future, cfg.abc, cfg.xyz)
    return seg.set_index("item_nbr")["segment"].map(SEGMENT_ORDER.index)


class LightGBMSegmenter(Segmenter):
    name = "lightgbm"
    label = "LightGBM"

    def _samples(
        self,
        weekly: pd.DataFrame,
        items: pd.DataFrame,
        origins: list[pd.Timestamp],
        families: list[str],
    ) -> pd.DataFrame:
        p, horizon = self.cfg.segmenter.lightgbm, self.cfg.sim.test_weeks
        frames = []
        for origin in origins:
            feats = build_features(
                weekly, items, origin, p.window_weeks, families, self.cfg
            )
            labels = future_labels(weekly, origin, horizon, self.cfg)
            feats["label"] = feats["item_nbr"].map(labels)
            feats["origin"] = origin
            frames.append(feats.dropna(subset=["label"]))
        out = pd.concat(frames, ignore_index=True)
        out["label"] = out["label"].astype(int)
        return out

    def _model(self):
        import lightgbm as lgb

        p = self.cfg.segmenter.lightgbm
        return lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=p.n_estimators,
            learning_rate=p.learning_rate,
            num_leaves=p.num_leaves,
            min_child_samples=p.min_child_samples,
            subsample=p.subsample,
            subsample_freq=1 if p.subsample < 1 else 0,
            colsample_bytree=p.colsample_bytree,
            class_weight=p.class_weight,
            random_state=self.cfg.seed,
            n_jobs=1,  # deterministic results
            verbose=-1,
        )

    def fit_predict(
        self, weekly_train: pd.DataFrame, items: pd.DataFrame
    ) -> SegmentationResult:
        p, horizon = self.cfg.segmenter.lightgbm, self.cfg.sim.test_weeks
        weeks = pd.DatetimeIndex(np.sort(weekly_train["week"].unique()))
        last = len(weeks) - horizon  # last origin whose label window fits in training
        idx = list(range(last, p.min_history_weeks - 1, -p.stride_weeks))[::-1]
        if len(idx) < 2:
            raise SchemaError(
                f"lightgbm needs at least {p.min_history_weeks + horizon + p.stride_weeks} "
                f"training weeks (min_history_weeks + test_weeks + stride_weeks), "
                f"found {len(weeks)}"
            )
        origins = [weeks[i] for i in idx]
        families = sorted(items["family"].astype(str).unique())
        samples = self._samples(weekly_train, items, origins, families)

        # model quality on the last origin (never seen in fitting)
        is_val = samples["origin"] == origins[-1]
        fit_rows, val_rows = samples[~is_val], samples[is_val]
        val_model = self._model().fit(fit_rows[FEATURES], fit_rows["label"])
        val_pred = np.asarray(val_model.predict(val_rows[FEATURES]))
        tables, scores = _diagnostics(val_rows["label"].to_numpy(), val_pred)

        # final model on every origin, applied at the train cutoff
        model = self._model().fit(samples[FEATURES], samples["label"])
        cutoff = weeks[-1] + pd.Timedelta(weeks=1)
        feats = build_features(
            weekly_train, items, cutoff, p.window_weeks, families, self.cfg
        )
        proba = np.asarray(model.predict_proba(feats[FEATURES]))
        classes = np.asarray(model.classes_)
        pred = pd.DataFrame(
            {
                "item_nbr": feats["item_nbr"].to_numpy(),
                "segment": [SEGMENT_ORDER[c] for c in classes[proba.argmax(axis=1)]],
                "confidence": proba.max(axis=1),
            }
        )

        # historical statistics stay those of the classic method (policy parameters use them)
        hist = classify_abc(weekly_train, self.cfg.abc).merge(
            classify_xyz(weekly_train, self.cfg.xyz), on="item_nbr"
        )
        seg = hist.drop(columns=["abc", "xyz"]).merge(pred, on="item_nbr", how="left")
        seg["abc"], seg["xyz"] = seg["segment"].str[0], seg["segment"].str[1]

        importance = pd.DataFrame(
            {"feature": FEATURES, "gain": model.booster_.feature_importance("gain")}
        )
        importance["gain_pct"] = 100 * importance["gain"] / importance["gain"].sum()
        tables["feature_importance"] = importance.sort_values(
            "gain", ascending=False, ignore_index=True
        )
        report = {
            **scores,
            "origins": len(origins),
            "first_origin": str(origins[0].date()),
            "validation_origin": str(origins[-1].date()),
            "train_samples": len(samples),
            "validation_samples": len(val_rows),
            "mean_confidence": round(float(pred["confidence"].mean()), 4),
        }
        return SegmentationResult(seg=seg[RESULT_COLUMNS], tables=tables, report=report)

    def plot_diagnostics(self, result: SegmentationResult, charts_dir: Path) -> None:
        from almacen import reporting

        reporting.plot_confusion_matrix(
            result.tables["confusion_matrix"], charts_dir / "model_confusion_matrix.png"
        )
        reporting.plot_feature_importance(
            result.tables["feature_importance"],
            charts_dir / "model_feature_importance.png",
        )


def _diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Validation metrics: overall scores, per-segment scores and the confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    labels = list(range(len(SEGMENT_ORDER)))
    seg_true = np.array([SEGMENT_ORDER[i] for i in y_true])
    seg_pred = np.array([SEGMENT_ORDER[i] for i in y_pred])
    present = sorted(set(y_true) | set(y_pred))
    # sklearn leaves zero_division unannotated: checkers infer `str` from its "warn" default
    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=present,
            average="macro",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=present,
            average="weighted",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
        "abc_accuracy": float(
            np.mean([t[0] == p[0] for t, p in zip(seg_true, seg_pred)])
        ),
        "xyz_accuracy": float(
            np.mean([t[1] == p[1] for t, p in zip(seg_true, seg_pred)])
        ),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )
    per_class = pd.DataFrame(
        {
            "segment": SEGMENT_ORDER,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=pd.Index(SEGMENT_ORDER, name="actual"),
        columns=[f"pred_{s}" for s in SEGMENT_ORDER],
    ).reset_index()
    metrics = pd.DataFrame(
        [{"metric": k, "value": float(v)} for k, v in scores.items()]
    )
    tables = {
        "model_metrics": metrics,
        "model_metrics_by_segment": per_class,
        "confusion_matrix": cm,
    }
    return tables, {f"validation_{k}": round(float(v), 4) for k, v in scores.items()}
