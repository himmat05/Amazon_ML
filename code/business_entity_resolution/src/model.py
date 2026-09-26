"""
LightGBM Matcher and Macro F_0.5 Threshold Optimizer.
Trains a precision-weighted gradient boosted decision tree classifier,
optimizes the decision threshold for the official evaluation metric,
and formats predictions for matching_results.tsv.
"""

from collections import defaultdict
import os
import pickle
import time
from typing import Dict, List, Optional, Set, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from metric import compute_macro_f05


try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class EntityMatchingModel:
    """LightGBM + CatBoost Dual Gradient-Boosted Ensemble for Entity Matching."""

    def __init__(
        self,
        n_estimators: int = 350,
        learning_rate: float = 0.04,
        max_depth: int = 6,
        num_leaves: int = 31,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42,
    ):
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.model_lgb: Optional[lgb.LGBMClassifier] = None
        self.model_cat: Optional[any] = None
        self.best_threshold: float = 0.70
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """Train LightGBM and CatBoost binary classifiers on feature matrix."""
        t0 = time.time()
        print(f"Training LightGBM classifier on {len(X_train):,} candidate pairs...")
        self.feature_names = feature_names

        self.model_lgb = lgb.LGBMClassifier(**self.params)
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None

        self.model_lgb.fit(
            X_train,
            y_train,
            eval_set=eval_set,
        )
        print(f"LightGBM training completed in {time.time()-t0:.2f}s")

        if CatBoostClassifier is not None:
            t_cat = time.time()
            print(f"Training CatBoost classifier on {len(X_train):,} candidate pairs on NVIDIA GPU...")
            try:
                self.model_cat = CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.04,
                    depth=6,
                    task_type="GPU",
                    verbose=0,
                    random_seed=42,
                )
                eval_cat = (X_val, y_val) if X_val is not None and y_val is not None else None
                self.model_cat.fit(X_train, y_train, eval_set=eval_cat, early_stopping_rounds=40)
                print(f"CatBoost GPU training completed in {time.time()-t_cat:.2f}s")
            except Exception as e:
                print(f"CatBoost GPU fallback to CPU: {e}")
                self.model_cat = CatBoostClassifier(
                    iterations=350,
                    learning_rate=0.05,
                    depth=6,
                    verbose=0,
                    random_seed=42,
                )
                self.model_cat.fit(X_train, y_train)
                print(f"CatBoost CPU training completed in {time.time()-t_cat:.2f}s")
        else:
            self.model_cat = None

        if XGBClassifier is not None:
            t_xgb = time.time()
            print(f"Training XGBoost classifier on {len(X_train):,} candidate pairs on NVIDIA GPU (CUDA)...")
            try:
                self.model_xgb = XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.04,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="logloss",
                    tree_method="hist",
                    device="cuda",
                    random_state=42,
                )
                self.model_xgb.fit(X_train, y_train)
                print(f"XGBoost CUDA training completed in {time.time()-t_xgb:.2f}s")
            except Exception as e:
                print(f"XGBoost CUDA fallback to CPU: {e}")
                self.model_xgb = XGBClassifier(
                    n_estimators=350,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1,
                )
                self.model_xgb.fit(X_train, y_train)
                print(f"XGBoost CPU training completed in {time.time()-t_xgb:.2f}s")
        else:
            self.model_xgb = None

        # Print top feature importances
        importances = self.model_lgb.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        print("\n--- Top Feature Importances (LightGBM) ---")
        for idx in sorted_indices[:10]:
            print(f"  {self.feature_names[idx]:26s}: {importances[idx]:,}")
        print("------------------------------------------\n")

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict blended match probabilities across LightGBM, CatBoost, and XGBoost."""
        if self.model_lgb is None:
            raise ValueError("Model is not trained yet!")
        
        preds = []
        weights = []
        
        preds.append(self.model_lgb.predict_proba(X)[:, 1])
        weights.append(0.40)
        
        if self.model_cat is not None:
            preds.append(self.model_cat.predict_proba(X)[:, 1])
            weights.append(0.35)
            
        if getattr(self, "model_xgb", None) is not None:
            if not getattr(self, "_xgb_device_cpu_set", False):
                try:
                    self.model_xgb.set_params(device="cpu")
                    self._xgb_device_cpu_set = True
                except Exception:
                    pass
            preds.append(self.model_xgb.predict_proba(X)[:, 1])
            weights.append(0.25)
            
        total_w = sum(weights)
        p_final = sum(w * p for w, p in zip(weights, preds)) / total_w
        return p_final

    def optimize_threshold(
        self,
        probabilities: np.ndarray,
        pair_ids: List[Tuple[str, str]],
        gt_map: Dict[str, Set[str]],
        s1_entity_ids: List[str],
        thresholds: Optional[np.ndarray] = None,
    ) -> float:
        """Sweep decision thresholds on validation set to maximize official Macro F_0.5."""
        if thresholds is None:
            thresholds = np.arange(0.40, 0.92, 0.02)

        print(f"Optimizing decision threshold across {len(thresholds)} values for Macro F_0.5...")

        # Pre-group pairs and probabilities by s1_id for instant threshold scoring
        s1_to_pairs = defaultdict(list)
        for i, (s1_id, cand_id) in enumerate(pair_ids):
            s1_to_pairs[s1_id].append((cand_id, probabilities[i]))

        best_score = -1.0
        best_thresh = 0.70
        best_results = {}

        for thresh in thresholds:
            pred_map = {}
            for s1_id in s1_entity_ids:
                matches = [cid for cid, p in s1_to_pairs.get(s1_id, ()) if p >= thresh]
                pred_map[s1_id] = set(matches)

            res = compute_macro_f05(gt_map, pred_map, beta=0.5)
            score = res["macro_f05"]

            if score > best_score:
                best_score = score
                best_thresh = float(thresh)
                best_results = res

        self.best_threshold = best_thresh
        print("\n" + "=" * 52)
        print(f"      THRESHOLD OPTIMIZATION RESULTS (Best: {self.best_threshold:.2f})")
        print("=" * 52)
        print(f"Best Validation Macro F_0.5 : {best_results['macro_f05']:.4f}")
        print(f"Singleton Accuracy          : {best_results['singleton_accuracy']*100:.2f}%")
        print(f"Non-Singleton F_0.5         : {best_results['non_singleton_f05']:.4f}")
        print("=" * 52 + "\n")

        return best_thresh

    def predict_matches(
        self,
        probabilities: np.ndarray,
        pair_ids: List[Tuple[str, str]],
        s1_entity_ids: List[str],
        threshold: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """Filter candidate pairs using the optimal threshold and group by Source 1 entity."""
        tau = threshold if threshold is not None else self.best_threshold
        pred_map = {sid: [] for sid in s1_entity_ids}

        for i, (s1_id, cand_id) in enumerate(pair_ids):
            if probabilities[i] >= tau:
                pred_map[s1_id].append(cand_id)

        return pred_map

    def save_model(self, path: str):
        """Save ensemble model and metadata to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "model_lgb": self.model_lgb,
            "model_cat": self.model_cat,
            "model_xgb": getattr(self, "model_xgb", None),
            "best_threshold": self.best_threshold,
            "feature_names": self.feature_names,
            "params": self.params,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved trained ensemble model to: {path}")

    @classmethod
    def load_model(cls, path: str) -> "EntityMatchingModel":
        """Load trained model and metadata from file with backwards compatibility."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model_lgb = data.get("model_lgb", data.get("model"))
        obj.model_cat = data.get("model_cat", None)
        obj.model_xgb = data.get("model_xgb", None)
        obj.best_threshold = data["best_threshold"]
        obj.feature_names = data["feature_names"]
        obj.params = data["params"]
        return obj


def save_matching_results(
    pred_map: Dict[str, List[str]],
    output_path: str,
):
    """Save matching_results.tsv according to official competition rules."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for s1_id, matches in pred_map.items():
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matches) if matches else "",
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep="\t", index=False)
    print(f"Saved {len(df_out):,} rows to matching file: {output_path}")
