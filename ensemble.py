
"""Ensemble model that combines Random-Forest, KNN, and an Autoregressive model.

It imports the already‑implemented models and aggregates their predictions.
"""
from __future__ import annotations

from typing import List, Optional, Literal
try:
    from pyspark.sql import SparkSession  # type: ignore
except ImportError:
    SparkSession = None  # running outside Spark

import numpy as np
import pandas as pd
from scipy.stats import mode

from content_based_recommenders.random_forest import MyRecommender as RandomForestModel  # type: ignore
from content_based_recommenders.knn import KNeighborsRecommender as KNNModel  # type: ignore
from autoregressive_original import NGramAutoRegressiveRecommender as AutoregressiveModel  # type: ignore

Combination = Literal["average", "vote", "weighted"]


class EnsembleModel:
    """Lightweight ensemble that averages / votes over Random-Forest, KNN and an
    Autoregressive model.  It gracefully handles pandas **and** Spark
    DataFrame outputs so downstream code can call `.cache()` on the result.
    """

    # --------------------------------------------------------------------- #
    # helpers                                                               #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _safe_instantiate(model_cls, random_state: Optional[int], **kwargs):
        """Create *model_cls* dropping the ``random_state`` kwarg if unsupported."""
        try:
            return (
                model_cls(**kwargs)
                if random_state is None
                else model_cls(random_state=random_state, **kwargs)
            )
        except TypeError as e:
            if "random_state" in str(e):
                return model_cls(**kwargs)          # retry without seed
            raise

    # --------------------------------------------------------------------- #
    # constructor                                                           #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        combination: Combination = "average",
        weights: Optional[List[float]] = None,
        random_state: Optional[int] = None,
        **base_kwargs,
    ) -> None:
        self.combination = combination
        self.random_state = random_state

        base_classes = [RandomForestModel, KNNModel, AutoregressiveModel]
        self.models = [
            self._safe_instantiate(cls, random_state, **base_kwargs)
            for cls in base_classes
        ]

        if combination == "weighted":
            if weights is None or len(weights) != len(self.models):
                raise ValueError("Supply one weight per base learner.")
            self.weights = np.asarray(weights, dtype=float)
        else:
            self.weights = np.ones(len(self.models), dtype=float)

        # optional Spark support
        try:
            from pyspark.sql import SparkSession  # local import to avoid hard dep
            self._spark: Optional["SparkSession"] = SparkSession.getActiveSession()
        except Exception:                         # noqa: BLE001
            self._spark = None

    # --------------------------------------------------------------------- #
    # training                                                              #
    # --------------------------------------------------------------------- #
    def fit(self, *args, **kwargs):
        """Forward *all* args/kwargs to each base learner’s ``fit``."""
        for m in self.models:
            m.fit(*args, **kwargs)
        return self

    # --------------------------------------------------------------------- #
    # stacking / restoration                                                #
    # --------------------------------------------------------------------- #
    def _stack_predictions(self, preds):
        """Return (stacked ndarray, kind, index, columns)."""
        norm: list[Any] = []
        for p in preds:
            # Spark DF → pandas
            if not isinstance(p, (np.ndarray, pd.DataFrame)) and hasattr(p, "toPandas"):
                p = p.toPandas()
            norm.append(p)

        if any(isinstance(p, pd.DataFrame) for p in norm):
            first_df = next(p for p in norm if isinstance(p, pd.DataFrame))
            idx, cols = first_df.index, first_df.columns
            arrs = [p.values if isinstance(p, pd.DataFrame) else np.asarray(p) for p in norm]
            return np.stack(arrs), "dataframe", idx, cols

        return np.stack([np.asarray(p) for p in norm]), "array", None, None

    def _restore_type(self, arr, kind, index, columns):
        """Convert ndarray back to pandas or Spark DF depending on context."""
        if kind != "dataframe":
            return arr

        df = pd.DataFrame(arr, index=index, columns=columns)

        # keep id columns as int so Spark infers LongType
        for c in ("user_idx", "item_idx"):
            if c in df.columns:
                df[c] = df[c].astype("int64")

        if self._spark is not None:
            try:
                return self._spark.createDataFrame(df.reset_index(drop=True))
            except Exception:                      # noqa: BLE001
                pass                               # fallback to pandas

        return df

    # --------------------------------------------------------------------- #
    # prediction                                                            #
    # --------------------------------------------------------------------- #
    def predict(self, *args, **kwargs):
        """Aggregate predictions from base learners, preserving structure."""
        raw = [m.predict(*args, **kwargs) for m in self.models]
        stacked, kind, idx, cols = self._stack_predictions(raw)

        # ------ numeric aggregation -------------------------------------- #
        if self.combination == "average":
            agg = stacked.mean(axis=0)
            return self._restore_type(agg, kind, idx, cols)

        if self.combination == "weighted":
            w = self.weights / self.weights.sum()
            agg = np.tensordot(w, stacked, axes=1)
            return self._restore_type(agg, kind, idx, cols)

        # ------ majority vote (classification) --------------------------- #
        if kind == "dataframe":
            voted = pd.DataFrame(index=idx, columns=cols)
            for c in range(stacked.shape[2]):
                voted.iloc[:, c] = mode(stacked[:, :, c], axis=0, keepdims=False).mode
            return voted

        return mode(stacked, axis=0, keepdims=False).mode

    # --------------------------------------------------------------------- #
    # convenience                                                           #
    # --------------------------------------------------------------------- #
    def fit_predict(self, *args, **kwargs):
        return self.fit(*args, **kwargs).predict(*args, **kwargs)

