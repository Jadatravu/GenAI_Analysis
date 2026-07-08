"""
=========================================================
  Generic Single-Agent ML Pipeline — phidata 2.7.10
=========================================================
Tools exposed to the agent
  1.  EDA              - eda_toolkit
  2.  Imputation       - imputation_toolkit
  3.  Outlier Treatment- outlier_toolkit
  4.  Scaling          - scaling_toolkit
  5.  Encoding         - encoding_toolkit
  6.  Modelling        - modelling_toolkit
  7.  Model Selection  - selection_toolkit
  8.  Model Pickling   - pickling_toolkit
  9.  Code Generation  - codegen_toolkit

Run:
    python ml_agent.py
"""

import json, os, pickle, textwrap, warnings
import hashlib, logging, uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Optional

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                    LabelEncoder, OneHotEncoder, OrdinalEncoder)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              mean_squared_error, r2_score, mean_absolute_error,
                              classification_report, confusion_matrix)
from sklearn.linear_model  import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree           import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble       import (RandomForestClassifier, RandomForestRegressor,
                                    GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm            import SVC, SVR
from sklearn.neighbors      import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes    import GaussianNB

try:
    from xgboost  import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from phi.agent  import Agent
from phi.tools  import Toolkit
from phi.model.openai import OpenAIChat

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  Typed pipeline state — one instance per agent run
#  Replaces the old global dict; concurrent runs are safe.
# ─────────────────────────────────────────────────────────
from dataclasses import dataclass, field as _field
from typing import Any

@dataclass
class PipelineState:
    """Validated, typed container for all pipeline artefacts.

    Pass one instance to build_ml_agent() so concurrent runs
    never share state.  Access via the module-level STATE alias.
    """
    df_raw          : Any          = None   # original DataFrame
    df_processed    : Any          = None   # after preprocessing
    X_train         : Any          = None
    X_test          : Any          = None
    y_train         : Any          = None
    y_test          : Any          = None
    problem_type    : str          = None   # "classification" | "regression"
    target_col      : str          = None
    feature_cols    : list         = None
    models          : dict         = _field(default_factory=dict)
    scores          : dict         = _field(default_factory=dict)
    best_model_name : str          = None
    best_model      : Any          = None
    output_dir      : Path         = _field(default_factory=lambda: Path("ml_output"))
    run_id          : str          = None   # UUID at first load_dataset
    pipeline_log    : list         = _field(default_factory=list)
    data_hash       : str          = None   # SHA-256 of raw file bytes
    scaler          : Any          = None
    encoders        : dict         = _field(default_factory=dict)
    imputers        : dict         = _field(default_factory=dict)
    model_pkl_path  : str          = None

    def get(self, key: str, default=None):
        """Dict-compatible .get() for drop-in compatibility."""
        return getattr(self, key, default)

    def update(self, **kwargs):
        """Dict-compatible .update() for drop-in compatibility."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def reset(self):
        """Reset all pipeline artefacts while keeping output_dir."""
        out = self.output_dir
        self.__init__()
        self.output_dir = out
        self.output_dir.mkdir(exist_ok=True)


# Module-level default instance — replaced per-run by build_ml_agent()
STATE = PipelineState()
STATE.output_dir.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════
#  AUDIT LOGGING INFRASTRUCTURE
# ══════════════════════════════════════════════════════════
def _get_audit_logger() -> logging.Logger:
    """Return (or create) the pipeline audit logger writing to audit.log."""
    logger = logging.getLogger("ml_agent.audit")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_path = STATE.get("output_dir", Path("ml_output")) / "audit.log"
    log_path.parent.mkdir(exist_ok=True)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    # Also mirror to stdout so the agent can see it
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def audit_log(event: str, details: dict = None):
    """Write one structured JSON line to audit.log."""
    record = {
        "ts"     : datetime.now(timezone.utc).isoformat(),
        "run_id" : STATE.get("run_id") or "unset",
        "event"  : event,
    }
    if details:
        record.update(details)
    try:
        _get_audit_logger().info(json.dumps(record))
    except Exception:
        pass  # never let logging crash the pipeline


def _audit_tool(fn):
    """Decorator: wrap any Toolkit method with entry/exit audit logging."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = __import__("time").perf_counter()
        # Capture positional args (skip 'self')
        arg_summary = {}
        try:
            import inspect
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            for i, val in enumerate(args[1:], 1):   # skip self
                if i - 1 < len(params):
                    arg_summary[params[i-1]] = str(val)[:120]
        except Exception:
            pass
        audit_log("tool_call_start", {
            "tool"     : fn.__qualname__,
            "args"     : arg_summary,
        })
        result = fn(*args, **kwargs)
        elapsed_ms = round((__import__("time").perf_counter() - t0) * 1000, 2)
        # Log pipeline step
        STATE["pipeline_log"].append({
            "step"        : fn.__qualname__,
            "ts"          : datetime.now(timezone.utc).isoformat(),
            "duration_ms" : elapsed_ms,
        })
        audit_log("tool_call_end", {
            "tool"        : fn.__qualname__,
            "duration_ms" : elapsed_ms,
            "result_preview": str(result)[:200] if result else "",
        })
        return result
    return wrapper


# ══════════════════════════════════════════════════════════
#  1.  EDA TOOLKIT
# ══════════════════════════════════════════════════════════
class EDATookit(Toolkit):
    def __init__(self):
        super().__init__(name="eda_toolkit")
        self.register(self.load_dataset)
        self.register(self.dataset_overview)
        self.register(self.missing_value_report)
        self.register(self.statistical_summary)
        self.register(self.correlation_heatmap)
        self.register(self.target_distribution)
        self.register(self.detect_problem_type)
        self.register(self.set_problem_type)

    @_audit_tool
    def load_dataset(self, file_path: str, target_column: str) -> str:
        """Load a CSV/Excel dataset and set the target column.

        Args:
            file_path: Path to the dataset file (.csv or .xlsx).
            target_column: Name of the target/label column.
        """
        try:
            path = Path(file_path)
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            elif path.suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                return f"ERROR: Unsupported file type '{path.suffix}'. Use .csv or .xlsx."

            if target_column not in df.columns:
                return f"ERROR: Target column '{target_column}' not found. Available: {list(df.columns)}"

            # Auto-detect and parse object/string columns that look like dates
            # Use both "object" and "string" selectors for pandas 2/3 compatibility
            str_candidates = set(df.select_dtypes(include="object").columns) | \
                             set(df.select_dtypes(include="string").columns)
            for col in str_candidates:
                if col == target_column:
                    continue
                sample = df[col].dropna().head(100).astype(str)
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    success_rate = parsed.notna().mean()
                    if success_rate >= 0.8:   # ≥80% of samples parse as dates
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass  # not a date column

            dt_cols = list(df.select_dtypes(include=["datetime64", "datetime", "datetimetz"]).columns)
            dt_cols = [c for c in dt_cols if c != target_column]
            STATE["df_raw"]       = df.copy()
            STATE["df_processed"] = df.copy()
            STATE["target_col"]   = target_column
            STATE["feature_cols"] = [c for c in df.columns if c != target_column]

            # ── Audit: run_id, data hash, structured load event ─────────
            if STATE.get("run_id") is None:
                STATE["run_id"] = str(uuid.uuid4())
            try:
                raw_bytes = Path(file_path).read_bytes()
                STATE["data_hash"] = hashlib.sha256(raw_bytes).hexdigest()
            except Exception:
                STATE["data_hash"] = "unavailable"
            audit_log("dataset_loaded", {
                "file_path"  : str(file_path),
                "sha256"     : STATE["data_hash"],
                "rows"       : int(df.shape[0]),
                "cols"       : int(df.shape[1]),
                "target_col" : target_column,
            })

            dt_note = (f" Detected {len(dt_cols)} datetime cols {dt_cols} — "
                       "call handle_datetime_features() in EncodingToolkit to extract features."
                       if dt_cols else "")
            return (f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols. "
                    f"Target='{target_column}'. Features={len(STATE['feature_cols'])}.{dt_note}")
        except Exception as e:
            return f"ERROR loading dataset: {e}"

    def dataset_overview(self) -> str:
        """Return shape, column dtypes, numeric/object column lists, and a 5-row preview.

        Requires: load_dataset() must have been called.
        Returns: JSON string with keys shape, dtypes, numeric_cols, object_cols, head.
        """
        df = STATE.get("df_raw")
        if df is None:
            return "ERROR: No dataset loaded. Call load_dataset first."
        info = {
            "shape"       : list(df.shape),
            "dtypes"      : df.dtypes.astype(str).to_dict(),
            "numeric_cols": list(df.select_dtypes(include=np.number).columns),
            "object_cols" : list(df.select_dtypes(include="object").columns),
            "head"        : df.head(5).to_dict(),
        }
        return json.dumps(info, default=str)

    def missing_value_report(self) -> str:
        """Report count and percentage of missing values per column, sorted by severity.

        Requires: load_dataset() must have been called.
        Returns: JSON string with keys missing_count and missing_pct per column,
                 or the plain string "No missing values found." if the dataset is clean.
        Call before imputation to identify which columns need treatment.
        """
        df = STATE.get("df_raw")
        if df is None:
            return "ERROR: No dataset loaded."
        miss = df.isnull().sum()
        miss_pct = (miss / len(df) * 100).round(2)
        report = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct})
        report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)
        if report.empty:
            return "✅ No missing values found."
        return report.to_json()

    def statistical_summary(self) -> str:
        """Compute descriptive statistics for all numeric and top-5 values for categoricals.

        Requires: load_dataset() must have been called.
        Returns: JSON string with two keys — "numeric" (describe() output) and
                 "categorical" (unique count + top-5 value frequencies per column).
        """
        df = STATE.get("df_raw")
        if df is None:
            return "ERROR: No dataset loaded."
        num_summary = df.describe(include=np.number).round(4).to_dict()
        cat_summary = {}
        for col in df.select_dtypes(include="object").columns:
            cat_summary[col] = {
                "unique"    : int(df[col].nunique()),
                "top_values": df[col].value_counts().head(5).to_dict(),
            }
        return json.dumps({"numeric": num_summary, "categorical": cat_summary}, default=str)

    def correlation_heatmap(self) -> str:
        """Generate and save a Pearson correlation heatmap for all numeric feature columns.

        Requires: load_dataset() must have been called. At least 2 numeric columns needed.
        Returns: plain string with the path to the saved PNG file.
        Side-effect: writes correlation_heatmap.png to output_dir.
        """
        df = STATE.get("df_raw")
        if df is None:
            return "ERROR: No dataset loaded."
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] < 2:
            return "Not enough numeric columns for a heatmap."
        plt.figure(figsize=(12, 8))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        path = STATE["output_dir"] / "correlation_heatmap.png"
        plt.savefig(path)
        plt.close()
        return f"✅ Correlation heatmap saved → {path}"

    def target_distribution(self) -> str:
        """Plot and return the distribution of the target column.

        Requires: load_dataset() must have been called.
        Returns: JSON string with keys plot_saved (file path) and stats
                 (describe() output for the target column).
        Side-effect: writes target_distribution.png to output_dir.
        Call before detect_problem_type to visually confirm class balance or skewness.
        """
        df    = STATE.get("df_raw")
        tcol  = STATE.get("target_col")
        if df is None or tcol is None:
            return "ERROR: No dataset or target column set."
        plt.figure(figsize=(8, 5))
        if df[tcol].dtype == "object" or df[tcol].nunique() <= 20:
            df[tcol].value_counts().plot(kind="bar", color="steelblue")
            plt.xlabel(tcol); plt.ylabel("Count"); plt.title("Target Distribution")
        else:
            df[tcol].hist(bins=30, color="steelblue", edgecolor="black")
            plt.xlabel(tcol); plt.ylabel("Frequency"); plt.title("Target Distribution")
        plt.tight_layout()
        path = STATE["output_dir"] / "target_distribution.png"
        plt.savefig(path); plt.close()
        stats = df[tcol].describe().to_dict()
        return json.dumps({"plot_saved": str(path), "stats": stats}, default=str)

    def detect_problem_type(self) -> str:
        """Auto-detect whether the ML problem is classification or regression.

        Requires: load_dataset() must have been called first.
        Heuristic: object/string targets → classification; numeric targets with
        ≤ 20 unique values → classification; otherwise → regression.
        If the heuristic is wrong, call set_problem_type() to override it.

        Returns: plain string confirming detected type and basis for decision.
        """
        df   = STATE.get("df_raw")
        tcol = STATE.get("target_col")
        if df is None or tcol is None:
            return "ERROR: Load dataset first."
        target = df[tcol]
        if pd.api.types.is_string_dtype(target) or target.dtype == "object":
            STATE["problem_type"] = "classification"
            basis = "string/object dtype"
        elif target.nunique() <= 20 and pd.api.types.is_integer_dtype(target):
            STATE["problem_type"] = "classification"
            basis = f"integer target with only {target.nunique()} unique values"
        else:
            STATE["problem_type"] = "regression"
            basis = f"continuous numeric target ({target.nunique()} unique values)"
        return (f"✅ Problem type detected: **{STATE['problem_type']}** "
                f"(basis: {basis}). Call set_problem_type() to override if incorrect.")

    def set_problem_type(self, problem_type: str) -> str:
        """Manually override the auto-detected problem type.

        Use this when detect_problem_type() gives the wrong result — e.g.
        an integer regression target with fewer than 20 unique values.

        Args:
            problem_type: Must be 'classification' or 'regression'.

        Returns: confirmation string, or ERROR if value is invalid.
        """
        valid = {"classification", "regression"}
        pt = problem_type.strip().lower()
        if pt not in valid:
            return f"ERROR: problem_type must be one of {valid}. Got '{problem_type}'."
        STATE["problem_type"] = pt
        return f"✅ Problem type manually set to '{pt}'."


# ══════════════════════════════════════════════════════════
#  2.  IMPUTATION TOOLKIT
# ══════════════════════════════════════════════════════════
class ImputationToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="imputation_toolkit")
        self.register(self.impute_numeric)
        self.register(self.impute_categorical)
        self.register(self.knn_impute)

    @_audit_tool
    def impute_numeric(self, strategy: str = "median") -> str:
        """Impute missing values in numeric columns.

        Args:
            strategy: 'mean', 'median', or 'most_frequent'.
        """
        df = STATE.get("df_processed")
        if df is None:
            return "ERROR: No dataset loaded."
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        missing  = [c for c in num_cols if df[c].isnull().any()]
        if not missing:
            return "✅ No missing values in numeric columns."
        # Drop columns that are entirely NaN (imputer can't infer strategy value)
        all_nan = [c for c in missing if df[c].isnull().all()]
        if all_nan:
            df.drop(columns=all_nan, inplace=True)
            missing = [c for c in missing if c not in all_nan]
        if not missing:
            return f"✅ Dropped all-NaN columns: {all_nan}. No remaining numeric missing values."
        imp = SimpleImputer(strategy=strategy)
        df[missing] = imp.fit_transform(df[missing])
        for col in missing:
            STATE["imputers"][col] = imp
        STATE["df_processed"] = df
        return f"✅ Numeric imputation ({strategy}) applied to: {missing}"

    @_audit_tool
    def impute_categorical(self, strategy: str = "most_frequent") -> str:
        """Impute missing values in categorical columns.

        Args:
            strategy: 'most_frequent' or 'constant'.
        """
        df = STATE.get("df_processed")
        if df is None:
            return "ERROR: No dataset loaded."
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        missing  = [c for c in cat_cols if df[c].isnull().any()]
        if not missing:
            return "✅ No missing values in categorical columns."
        imp = SimpleImputer(strategy=strategy, fill_value="Unknown")
        df[missing] = imp.fit_transform(df[missing])
        for col in missing:
            STATE["imputers"][col] = imp
        STATE["df_processed"] = df
        return f"✅ Categorical imputation ({strategy}) applied to: {missing}"

    @_audit_tool
    def knn_impute(self, n_neighbors: int = 5) -> str:
        """Apply KNN imputation to all numeric columns with missing values.

        Args:
            n_neighbors: Number of nearest neighbours for KNN imputer.
        """
        df = STATE.get("df_processed")
        if df is None:
            return "ERROR: No dataset loaded."
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        missing  = [c for c in num_cols if df[c].isnull().any()]
        if not missing:
            return "✅ No missing numeric values to impute with KNN."
        imp = KNNImputer(n_neighbors=n_neighbors)
        df[missing] = imp.fit_transform(df[missing])
        STATE["imputers"]["knn"] = imp
        STATE["df_processed"] = df
        return f"✅ KNN imputation (k={n_neighbors}) applied to: {missing}"


# ══════════════════════════════════════════════════════════
#  3.  OUTLIER TOOLKIT
# ══════════════════════════════════════════════════════════
class OutlierToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="outlier_toolkit")
        self.register(self.detect_outliers_iqr)
        self.register(self.treat_outliers_iqr_clip)
        self.register(self.detect_outliers_zscore)
        self.register(self.treat_outliers_zscore_remove)

    @_audit_tool
    def detect_outliers_iqr(self) -> str:
        """Detect outliers in numeric feature columns using the IQR (1.5×IQR) fence rule.

        Requires: load_dataset() and at least one preprocessing step (df_processed must exist).
        Returns: JSON dict mapping column name to {outlier_count, pct}, or the plain
                 string "No outliers detected via IQR." if the dataset is clean.
        Does NOT modify data — call treat_outliers_iqr_clip() to apply treatment.
        """
        df = STATE.get("df_processed")
        if df is None:
            return "ERROR: No dataset loaded."
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        target   = STATE.get("target_col")
        if target in num_cols and STATE.get("problem_type") == "classification":
            num_cols.remove(target)
        report = {}
        for col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR     = Q3 - Q1
            n_out   = int(((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum())
            if n_out > 0:
                report[col] = {"outlier_count": n_out, "pct": round(n_out/len(df)*100, 2)}
        return json.dumps(report) if report else "✅ No outliers detected via IQR."

    @_audit_tool
    def treat_outliers_iqr_clip(self) -> str:
        """Clip outliers to the IQR fence (Winsorization) for all numeric feature columns."""
        df = STATE.get("df_processed")
        if df is None:
            return "ERROR: No dataset loaded."
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        target   = STATE.get("target_col")
        if target in num_cols and STATE.get("problem_type") == "classification":
            num_cols.remove(target)
        treated = []
        for col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR     = Q3 - Q1
            lb, ub  = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            df[col] = df[col].clip(lb, ub)
            treated.append(col)
        STATE["df_processed"] = df
        return f"✅ IQR clipping applied to: {treated}"

    @_audit_tool
    def detect_outliers_zscore(self, threshold: float = 3.0) -> str:
        """Detect outliers in numeric feature columns using the Z-score method.

        Requires: load_dataset() must have been called; df_processed must exist.
        Args:
            threshold: Z-score threshold above which a value is flagged (default 3.0).
        Returns: JSON dict mapping column name to {outlier_count, pct}, or the plain
                 string "No outliers detected via Z-score." if clean.
        Does NOT modify data — call treat_outliers_zscore_remove() to drop flagged rows.
        """
        df = STATE.get("df_processed")
        if df is None:
            return "ERROR: No dataset loaded."
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        report = {}
        for col in num_cols:
            zscores = np.abs((df[col] - df[col].mean()) / df[col].std())
            n_out   = int((zscores > threshold).sum())
            if n_out > 0:
                report[col] = {"outlier_count": n_out, "pct": round(n_out/len(df)*100, 2)}
        return json.dumps(report) if report else "✅ No outliers detected via Z-score."

    @_audit_tool
    def treat_outliers_zscore_remove(self, threshold: float = 3.0) -> str:
        """Remove rows where any numeric feature exceeds Z-score threshold.

        Args:
            threshold: Z-score threshold above which rows are dropped.
        """
        df = STATE.get("df_processed")
        if df is None:
            return "ERROR: No dataset loaded."
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        before   = len(df)
        mask     = pd.Series([True]*len(df), index=df.index)
        for col in num_cols:
            zscores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask    = mask & (zscores <= threshold)
        df = df[mask]
        STATE["df_processed"] = df.reset_index(drop=True)
        removed = before - len(df)
        return f"✅ Z-score removal: {removed} rows removed. Remaining: {len(df)}"


# ══════════════════════════════════════════════════════════
#  4.  SCALING TOOLKIT
# ══════════════════════════════════════════════════════════
class ScalingToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="scaling_toolkit")
        self.register(self.standard_scale)
        self.register(self.minmax_scale)
        self.register(self.robust_scale)

    def _scale(self, scaler, name: str) -> str:
        df       = STATE.get("df_processed")
        tcol     = STATE.get("target_col")
        if df is None:
            return "ERROR: No dataset loaded."
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if tcol in num_cols:
            num_cols.remove(tcol)
        df[num_cols] = scaler.fit_transform(df[num_cols])
        STATE["df_processed"] = df
        STATE["scaler"]       = scaler
        return f"✅ {name} scaling applied to {len(num_cols)} numeric feature columns."

    @_audit_tool
    def standard_scale(self) -> str:
        """Apply StandardScaler (zero mean, unit variance) to numeric feature columns."""
        return self._scale(StandardScaler(), "StandardScaler")

    @_audit_tool
    def minmax_scale(self) -> str:
        """Apply MinMaxScaler (range [0,1]) to numeric feature columns."""
        return self._scale(MinMaxScaler(), "MinMaxScaler")

    @_audit_tool
    def robust_scale(self) -> str:
        """Apply RobustScaler (IQR-based, outlier-resistant) to numeric feature columns."""
        return self._scale(RobustScaler(), "RobustScaler")


# ══════════════════════════════════════════════════════════
#  5.  ENCODING TOOLKIT
# ══════════════════════════════════════════════════════════
class EncodingToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="encoding_toolkit")
        self.register(self.handle_datetime_features)
        self.register(self.onehot_encode)
        self.register(self.label_encode)
        self.register(self.ordinal_encode)
        self.register(self.encode_target)

    @_audit_tool
    def handle_datetime_features(self) -> str:
        """Extract year, month, day, dayofweek, hour features from all datetime columns, then drop the originals.

        Automatically called when the dataset contains datetime columns.
        Also parses any remaining object columns that look like dates.
        """
        df   = STATE.get("df_processed")
        tcol = STATE.get("target_col")
        if df is None:
            return "ERROR: No dataset loaded."

        extracted   = {}
        dropped_raw = []

        # Re-attempt parsing on any remaining object/string columns that look like dates
        str_candidates = set(df.select_dtypes(include="object").columns) | \
                         set(df.select_dtypes(include="string").columns)
        for col in list(str_candidates):
            if col == tcol:
                continue
            sample = df[col].dropna().astype(str).head(200)
            try:
                pd.to_datetime(sample, errors="raise")
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

        # Extract features from all datetime columns
        dt_cols = [c for c in df.select_dtypes(
            include=["datetime64", "datetime", "datetimetz"]).columns
                   if c != tcol]

        for col in dt_cols:
            prefix = col.replace(" ", "_").lower()
            df[prefix + "_year"]      = df[col].dt.year.astype("Int64")
            df[prefix + "_month"]     = df[col].dt.month.astype("Int64")
            df[prefix + "_day"]       = df[col].dt.day.astype("Int64")
            df[prefix + "_dayofweek"] = df[col].dt.dayofweek.astype("Int64")
            hour = df[col].dt.hour
            if hour.nunique() > 1:
                df[prefix + "_hour"] = hour.astype("Int64")
            extracted[col] = [prefix + "_year", prefix + "_month",
                              prefix + "_day", prefix + "_dayofweek"]
            df.drop(columns=[col], inplace=True)
            dropped_raw.append(col)

        # Compute difference (in days) between every pair of datetime cols that were parsed
        # e.g. ship_date - order_date → days_to_ship
        # (already dropped above; do it before drop if more than one datetime exists in original)

        if not dropped_raw:
            return "✅ No datetime columns found — nothing to extract."

        STATE["df_processed"] = df
        STATE["feature_cols"] = [c for c in df.columns if c != tcol]
        return (f"✅ Datetime feature extraction complete. "
                f"Dropped {dropped_raw}, extracted features: {extracted}")

    @_audit_tool
    def onehot_encode(self, max_cardinality: int = 15) -> str:
        """One-hot encode categorical columns with cardinality ≤ max_cardinality.

        Args:
            max_cardinality: Skip columns with more unique values than this.
        """
        df   = STATE.get("df_processed")
        tcol = STATE.get("target_col")
        if df is None:
            return "ERROR: No dataset loaded."
        cat_cols = [c for c in df.select_dtypes(include="object").columns if c != tcol]
        eligible = [c for c in cat_cols if df[c].nunique() <= max_cardinality]
        if not eligible:
            return "No eligible categorical columns for one-hot encoding."
        df = pd.get_dummies(df, columns=eligible, drop_first=True)
        STATE["df_processed"] = df
        STATE["feature_cols"] = [c for c in df.columns if c != tcol]
        return f"✅ One-hot encoding applied to: {eligible}. New shape: {df.shape}"

    @_audit_tool
    def label_encode(self) -> str:
        """Label-encode all remaining object columns (including high-cardinality)."""
        df   = STATE.get("df_processed")
        tcol = STATE.get("target_col")
        if df is None:
            return "ERROR: No dataset loaded."
        cat_cols = [c for c in df.select_dtypes(include="object").columns if c != tcol]
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            STATE["encoders"][col] = le
        STATE["df_processed"] = df
        return f"✅ Label encoding applied to: {cat_cols}"

    @_audit_tool
    def ordinal_encode(self, columns: str = "") -> str:
        """Ordinal-encode specified columns preserving natural order (e.g. Low < Medium < High).

        Requires: load_dataset() must have been called.

        Args:
            columns: Comma-separated column names to encode. Leave empty to encode all object columns.
        """
        df   = STATE.get("df_processed")
        tcol = STATE.get("target_col")
        if df is None:
            return "ERROR: No dataset loaded."
        if columns:
            cols = [c.strip() for c in columns.split(",")]
        else:
            cols = [c for c in df.select_dtypes(include="object").columns if c != tcol]
        enc = OrdinalEncoder()
        df[cols] = enc.fit_transform(df[cols].astype(str))
        STATE["encoders"]["ordinal"] = enc
        STATE["df_processed"]        = df
        return f"✅ Ordinal encoding applied to: {cols}"

    @_audit_tool
    def encode_target(self) -> str:
        """Label-encode the target column if it is categorical."""
        df   = STATE.get("df_processed")
        tcol = STATE.get("target_col")
        if df is None or tcol is None:
            return "ERROR: No dataset or target column set."
        is_categorical = (
            df[tcol].dtype == "object" or
            str(df[tcol].dtype) in ("string", "StringDtype", "str") or
            pd.api.types.is_string_dtype(df[tcol]) or
            pd.api.types.is_categorical_dtype(df[tcol])
        )
        if is_categorical:
            le = LabelEncoder()
            df[tcol] = le.fit_transform(df[tcol].astype(str))
            STATE["encoders"]["target"] = le
            STATE["df_processed"]       = df
            return f"✅ Target column '{tcol}' label-encoded. Classes: {list(le.classes_)}"
        return f"✅ Target column '{tcol}' is already numeric. No encoding needed."


# ══════════════════════════════════════════════════════════
#  6.  MODELLING TOOLKIT
# ══════════════════════════════════════════════════════════
class ModellingToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="modelling_toolkit")
        self.register(self.prepare_train_test_split)
        self.register(self.train_all_models)
        self.register(self.train_single_model)

    def _get_candidate_models(self, problem_type: str) -> dict:
        base = {
            "classification": {
                "LogisticRegression"    : LogisticRegression(max_iter=1000, random_state=42),
                "DecisionTree"          : DecisionTreeClassifier(random_state=42),
                "RandomForest"          : RandomForestClassifier(n_estimators=100, random_state=42),
                "GradientBoosting"      : GradientBoostingClassifier(random_state=42),
                "SVM"                   : SVC(probability=True, random_state=42),
                "KNN"                   : KNeighborsClassifier(),
                "NaiveBayes"            : GaussianNB(),
            },
            "regression": {
                "LinearRegression"      : LinearRegression(),
                "Ridge"                 : Ridge(random_state=42),
                "Lasso"                 : Lasso(random_state=42),
                "DecisionTree"          : DecisionTreeRegressor(random_state=42),
                "RandomForest"          : RandomForestRegressor(n_estimators=100, random_state=42),
                "GradientBoosting"      : GradientBoostingRegressor(random_state=42),
                "SVR"                   : SVR(),
                "KNN"                   : KNeighborsRegressor(),
            },
        }
        models = base.get(problem_type, {})
        if HAS_XGB:
            if problem_type == "classification":
                models["XGBoost"] = XGBClassifier(use_label_encoder=False,
                                                   eval_metric="logloss", random_state=42)
            else:
                models["XGBoost"] = XGBRegressor(random_state=42)
        if HAS_LGBM:
            if problem_type == "classification":
                models["LightGBM"] = LGBMClassifier(random_state=42, verbose=-1)
            else:
                models["LightGBM"] = LGBMRegressor(random_state=42, verbose=-1)
        return models

    @_audit_tool
    def prepare_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> str:
        """Split processed data into train/test sets.

        Args:
            test_size: Fraction of data for the test set (default 0.2).
            random_state: Random seed for reproducibility.
        """
        df   = STATE.get("df_processed")
        tcol = STATE.get("target_col")
        if df is None or tcol is None:
            return "ERROR: No processed dataset or target column."

        # Safety guard: drop any residual datetime / timedelta columns
        dt_residual = df.select_dtypes(
            include=["datetime64", "datetime", "datetimetz", "timedelta64", "timedelta"]
        ).columns.tolist()
        dt_residual = [c for c in dt_residual if c != tcol]
        if dt_residual:
            df.drop(columns=dt_residual, inplace=True)
            STATE["df_processed"] = df

        feat_cols = [c for c in df.columns if c != tcol]
        STATE["feature_cols"] = feat_cols
        X = df[feat_cols]
        y = df[tcol]
        stratify = y if STATE.get("problem_type") == "classification" else None
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify)
        except Exception:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
        STATE.update(X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te)
        return (f"✅ Split done — Train: {X_tr.shape}, Test: {X_te.shape}. "
                f"Features: {len(feat_cols)}")

    @_audit_tool
    def train_all_models(self) -> str:
        """Train all candidate models for the detected problem type and report CV scores."""
        if STATE.get("X_train") is None:
            return "ERROR: Call prepare_train_test_split first."
        ptype   = STATE.get("problem_type")
        if ptype is None:
            return "ERROR: Problem type not detected. Call detect_problem_type."
        models  = self._get_candidate_models(ptype)
        X_tr, y_tr = STATE["X_train"], STATE["y_train"]

        # ── Final dtype guard: sklearn cannot handle datetime / object / nullable-int ──
        # Drop any residual non-numeric columns (should have been handled upstream,
        # but this is the last safety net before model.fit()).
        bad_cols = [c for c in X_tr.columns
                    if not pd.api.types.is_numeric_dtype(X_tr[c])
                    or pd.api.types.is_datetime64_any_dtype(X_tr[c])]
        if bad_cols:
            X_tr = X_tr.drop(columns=bad_cols)
            X_te = STATE["X_test"].drop(columns=bad_cols)
            STATE["X_train"] = X_tr
            STATE["X_test"]  = X_te
            STATE["feature_cols"] = list(X_tr.columns)

        # Convert pandas nullable integer (Int64) to numpy int64
        for col in X_tr.columns:
            if hasattr(X_tr[col], "dtype") and str(X_tr[col].dtype).startswith("Int"):
                X_tr[col] = X_tr[col].astype(float)
                STATE["X_test"][col] = STATE["X_test"][col].astype(float)
        STATE["X_train"] = X_tr

        cv      = StratifiedKFold(5, shuffle=True, random_state=42) if ptype == "classification" \
                  else KFold(5, shuffle=True, random_state=42)
        scoring = "roc_auc" if ptype == "classification" else "r2"
        results = {}
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
                model.fit(X_tr, y_tr)
                STATE["models"][name] = model
                results[name]         = {"cv_mean": round(float(scores.mean()), 4),
                                         "cv_std" : round(float(scores.std()),  4)}
            except Exception as ex:
                results[name] = {"error": str(ex)}
        STATE["scores"] = results
        return json.dumps({"problem_type": ptype, "scoring": scoring, "results": results}, indent=2)

    @_audit_tool
    def train_single_model(self, model_name: str) -> str:
        """Train a specific model by name and report test-set metrics.

        Args:
            model_name: E.g. 'RandomForest', 'XGBoost', 'LogisticRegression'.
        """
        if STATE.get("X_train") is None:
            return "ERROR: Call prepare_train_test_split first."
        ptype  = STATE.get("problem_type")
        models = self._get_candidate_models(ptype)
        if model_name not in models:
            return f"ERROR: Unknown model '{model_name}'. Available: {list(models.keys())}"
        model  = models[model_name]
        X_tr   = STATE["X_train"].copy()
        # Drop non-numeric cols (last resort guard)
        bad    = [c for c in X_tr.columns
                  if not pd.api.types.is_numeric_dtype(X_tr[c])
                  or pd.api.types.is_datetime64_any_dtype(X_tr[c])]
        if bad:
            X_tr = X_tr.drop(columns=bad)
            STATE["X_test"] = STATE["X_test"].drop(columns=bad, errors="ignore")
            STATE["X_train"] = X_tr
        for col in X_tr.columns:
            if str(X_tr[col].dtype).startswith("Int"):
                X_tr[col] = X_tr[col].astype(float)
                STATE["X_test"][col] = STATE["X_test"][col].astype(float)
        model.fit(X_tr, STATE["y_train"])
        STATE["models"][model_name] = model
        y_pred = model.predict(STATE["X_test"])
        if ptype == "classification":
            metric = {"accuracy": round(accuracy_score(STATE["y_test"], y_pred), 4),
                      "f1_macro": round(f1_score(STATE["y_test"], y_pred, average="macro"), 4)}
        else:
            metric = {"r2"  : round(r2_score(STATE["y_test"], y_pred), 4),
                      "rmse": round(float(np.sqrt(mean_squared_error(STATE["y_test"], y_pred))), 4)}
        STATE["scores"][model_name] = metric
        return f"✅ {model_name} trained. Test metrics: {metric}"


# ══════════════════════════════════════════════════════════
#  7.  MODEL SELECTION TOOLKIT
# ══════════════════════════════════════════════════════════
class SelectionToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="selection_toolkit")
        self.register(self.evaluate_all_on_test)
        self.register(self.select_best_model)
        self.register(self.detailed_report)
        self.register(self.feature_importance_plot)

    @_audit_tool
    def evaluate_all_on_test(self) -> str:
        """Evaluate every trained model on the held-out test set."""
        if not STATE["models"]:
            return "ERROR: No trained models. Call train_all_models first."
        ptype   = STATE.get("problem_type")
        results = {}
        for name, model in STATE["models"].items():
            y_pred = model.predict(STATE["X_test"])
            if ptype == "classification":
                results[name] = {
                    "accuracy" : round(accuracy_score(STATE["y_test"], y_pred), 4),
                    "f1_macro" : round(f1_score(STATE["y_test"], y_pred, average="macro"), 4),
                }
                try:
                    y_prob = model.predict_proba(STATE["X_test"])
                    n_cls  = y_prob.shape[1]
                    auc    = roc_auc_score(STATE["y_test"], y_prob if n_cls > 2 else y_prob[:,1],
                                           multi_class="ovr" if n_cls > 2 else "raise", average="macro")
                    results[name]["roc_auc"] = round(auc, 4)
                except Exception:
                    pass
            else:
                results[name] = {
                    "r2"  : round(r2_score(STATE["y_test"], y_pred), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(STATE["y_test"], y_pred))), 4),
                    "mae" : round(float(mean_absolute_error(STATE["y_test"], y_pred)), 4),
                }
        STATE["scores"] = results
        return json.dumps(results, indent=2)

    @_audit_tool
    def select_best_model(self) -> str:
        """Select the best model based on primary metric (ROC-AUC / R²)."""
        scores = STATE.get("scores")
        if not scores:
            return "ERROR: No scores available. Call evaluate_all_on_test first."
        ptype   = STATE.get("problem_type")
        primary = "roc_auc" if ptype == "classification" else "r2"

        best_name, best_score = None, -np.inf
        for name, metrics in scores.items():
            if isinstance(metrics, dict) and primary in metrics:
                if metrics[primary] > best_score:
                    best_score = metrics[primary]
                    best_name  = name

        if best_name is None:
            # fallback to accuracy or r2
            fallback = "accuracy" if ptype == "classification" else "r2"
            for name, metrics in scores.items():
                if isinstance(metrics, dict) and fallback in metrics:
                    if metrics[fallback] > best_score:
                        best_score = metrics[fallback]
                        best_name  = name

        if best_name:
            STATE["best_model_name"] = best_name
            STATE["best_model"]      = STATE["models"][best_name]
            return (f"🏆 Best model: **{best_name}** "
                    f"(score={best_score:.4f} on {primary})")
        return "ERROR: Could not determine best model."

    @_audit_tool
    def detailed_report(self) -> str:
        """Generate a detailed evaluation report for the best selected model.

        Requires: select_best_model() must have been called.
        Returns: For classification — JSON with keys classification_report (text) and
                 confusion_matrix (list of lists).
                 For regression — JSON with keys r2, rmse, mae.
        """
        bm   = STATE.get("best_model")
        if bm is None:
            return "ERROR: No best model selected. Call select_best_model first."
        y_pred = bm.predict(STATE["X_test"])
        ptype  = STATE.get("problem_type")
        if ptype == "classification":
            report = classification_report(STATE["y_test"], y_pred)
            cm     = confusion_matrix(STATE["y_test"], y_pred).tolist()
            return json.dumps({"classification_report": report, "confusion_matrix": cm})
        else:
            return json.dumps({
                "r2"  : round(r2_score(STATE["y_test"], y_pred), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(STATE["y_test"], y_pred))), 4),
                "mae" : round(float(mean_absolute_error(STATE["y_test"], y_pred)), 4),
            })

    @_audit_tool
    def feature_importance_plot(self) -> str:
        """Generate and save a feature importance bar chart for the best tree-based model.

        Requires: select_best_model() must have been called.
        Returns: plain string with the path to the saved PNG, or an INFO message if
                 the best model does not expose feature_importances_.
        Side-effect: writes feature_importance.png to output_dir.
        """
        bm = STATE.get("best_model")
        if bm is None:
            return "ERROR: No best model. Call select_best_model first."
        if not hasattr(bm, "feature_importances_"):
            return f"INFO: '{STATE['best_model_name']}' does not expose feature_importances_."
        fi  = pd.Series(bm.feature_importances_, index=STATE["feature_cols"]).sort_values(ascending=False).head(20)
        plt.figure(figsize=(10, 6))
        fi.plot(kind="bar", color="teal")
        plt.title(f"Top-20 Feature Importances — {STATE['best_model_name']}")
        plt.ylabel("Importance"); plt.tight_layout()
        path = STATE["output_dir"] / "feature_importance.png"
        plt.savefig(path); plt.close()
        return f"✅ Feature importance plot saved → {path}"


# ══════════════════════════════════════════════════════════
#  8.  PICKLING TOOLKIT
# ══════════════════════════════════════════════════════════
class PicklingToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="pickling_toolkit")
        self.register(self.save_best_model)
        self.register(self.save_pipeline_artifacts)
        self.register(self.load_and_verify_model)

    @_audit_tool
    def save_best_model(self, filename: str = "") -> str:
        """Pickle the best model to disk.

        Args:
            filename: Output filename (optional). Defaults to '<model_name>_best_model.pkl'.
        """
        bm   = STATE.get("best_model")
        name = STATE.get("best_model_name", "model")
        if bm is None:
            return "ERROR: No best model. Run select_best_model first."
        fname = filename or f"{name}_best_model.pkl"
        path  = STATE["output_dir"] / fname
        # Embed audit metadata into the model object before pickling
        try:
            bm._run_id     = STATE.get("run_id", "unset")
            bm._trained_at = datetime.now(timezone.utc).isoformat()
            bm._model_name = STATE.get("best_model_name", "unknown")
        except Exception:
            pass   # some models don't allow attribute assignment

        with open(path, "wb") as f:
            pickle.dump(bm, f)
        STATE["model_pkl_path"] = str(path)
        audit_log("model_pickled", {
            "path"      : str(path),
            "model_name": STATE.get("best_model_name"),
            "run_id"    : STATE.get("run_id"),
        })
        return f"✅ Best model pickled → {path}"

    @_audit_tool
    def save_pipeline_artifacts(self) -> str:
        """Save all preprocessing artifacts (scaler, encoders, imputers) as a pipeline dict."""
        artifacts = {
            "scaler"      : STATE.get("scaler"),
            "encoders"    : STATE.get("encoders"),
            "imputers"    : STATE.get("imputers"),
            "feature_cols": STATE.get("feature_cols"),
            "target_col"  : STATE.get("target_col"),
            "problem_type": STATE.get("problem_type"),
        }
        path = STATE["output_dir"] / "pipeline_artifacts.pkl"
        with open(path, "wb") as f:
            pickle.dump(artifacts, f)
        return f"✅ Pipeline artifacts saved → {path}"

    @_audit_tool
    def load_and_verify_model(self, model_path: str = "") -> str:
        """Load a pickled model from disk and run a sanity prediction on the first 5 test rows.

        Requires: A model must have been pickled (save_best_model) or a valid path provided.
        Args:
            model_path: Absolute or relative path to the .pkl file.
                        Defaults to the last path saved by save_best_model().
        Returns: plain string with sample predictions, or ERROR if file not found.
        """
        path = model_path or STATE.get("model_pkl_path", "")
        if not path:
            return "ERROR: No model path provided."
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
        except FileNotFoundError:
            return f"ERROR: Model file not found at '{path}'."
        except Exception as e:
            return f"ERROR loading model: {e}"
        y_pred = model.predict(STATE["X_test"][:5])
        return f"✅ Model loaded from {path}. Sample predictions on 5 rows: {y_pred.tolist()}"


# ══════════════════════════════════════════════════════════
#  9.  CODE GENERATION TOOLKIT
# ══════════════════════════════════════════════════════════
class CodegenToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="codegen_toolkit")
        self.register(self.generate_inference_script)
        self.register(self.generate_training_script)

    @_audit_tool
    def generate_inference_script(self) -> str:
        """Generate a standalone Python inference script to load the pickled model and predict."""
        bm_path    = STATE.get("model_pkl_path", "ml_output/<model>_best_model.pkl")
        art_path   = str(STATE["output_dir"] / "pipeline_artifacts.pkl")
        feat_cols  = STATE.get("feature_cols", [])
        ptype      = STATE.get("problem_type", "classification")
        model_name = STATE.get("best_model_name") or "<model>"

        # Build script lines as plain strings — avoids f-string backslash restriction
        # on Python <= 3.11 (PEP 701 relaxes this in 3.12+).
        rb  = '"rb"'
        dq  = '"'
        lines = [
            '''"""''',
            "Auto-generated inference script",
            "Problem type : " + ptype,
            "Best model   : " + model_name,
            "Generated by : ML Single-Agent (phidata 2.7.10)",
            '''"""''',
            "import pickle, pandas as pd, numpy as np",
            "",
            "MODEL_PATH    = r" + dq + bm_path + dq,
            "ARTIFACT_PATH = r" + dq + art_path + dq,
            "FEATURE_COLS  = " + repr(feat_cols),
            "",
            "",
            "def load_artifacts():",
            "    with open(MODEL_PATH, " + repr("rb") + ") as f:",
            "        model = pickle.load(f)",
            "    with open(ARTIFACT_PATH, " + repr("rb") + ") as f:",
            "        artifacts = pickle.load(f)",
            "    return model, artifacts",
            "",
            "",
            "def preprocess(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:",
            "    \"\"\"Apply the same preprocessing pipeline used during training.\"\"\"",
            "    df = df.copy()",
            "",
            "    for col, imp in artifacts.get(" + repr("imputers") + ", {}).items():",
            "        if col in df.columns:",
            "            df[[col]] = imp.transform(df[[col]])",
            "",
            "    for col, enc in artifacts.get(" + repr("encoders") + ", {}).items():",
            "        if col in df.columns:",
            "            df[col] = enc.transform(df[col].astype(str))",
            "",
            "    scaler = artifacts.get(" + repr("scaler") + ")",
            "    if scaler:",
            "        num_cols = df.select_dtypes(include=" + repr("number") + ").columns.tolist()",
            "        df[num_cols] = scaler.transform(df[num_cols])",
            "",
            "    feat = artifacts.get(" + repr("feature_cols") + ", FEATURE_COLS)",
            "    return df[[c for c in feat if c in df.columns]]",
            "",
            "",
            "",
            "import logging as _logging, hashlib as _hashlib",
            "def _log_prediction(X, preds):",
            "    try:",
            "        import json, datetime",
            "        _logging.basicConfig(filename=" + repr("predictions.log") + ",",
            "            level=_logging.INFO, format=" + repr("%(message)s") + ")",
            "        _logging.getLogger().info(json.dumps({",
            "            " + repr("ts") + ": datetime.datetime.utcnow().isoformat(),",
            "            " + repr("input_hash") + ": _hashlib.sha256(str(X.values.tolist()).encode()).hexdigest()[:16],",
            "            " + repr("n_rows") + ": len(preds),",
            "            " + repr("predictions") + ": preds.tolist()[:20],",
            "        }))",
            "    except Exception:",
            "        pass",
            "",
            "",
"def predict(input_df: pd.DataFrame):",
            "    model, artifacts = load_artifacts()",
            "    X = preprocess(input_df, artifacts)",
            "    preds = model.predict(X)",
            "    _log_prediction(X, preds)",
            "    return preds",
            "",
            "",
            "def predict_proba(input_df: pd.DataFrame):",
            "    \"\"\"Returns probability scores (classification only).\"\"\"",
            "    model, artifacts = load_artifacts()",
            "    X = preprocess(input_df, artifacts)",
            "    if hasattr(model, " + repr("predict_proba") + "):",
            "        return model.predict_proba(X)",
            "    raise ValueError(" + repr("Model does not support predict_proba.") + ")",
            "",
            "",
            "if __name__ == " + repr("__main__") + ":",
            "    sample = pd.DataFrame({col: [0] for col in FEATURE_COLS})",
            "    preds  = predict(sample)",
            "    print(" + repr("Sample predictions:") + ", preds)",
        ]
        code = "\n".join(lines) + "\n"

        path = STATE["output_dir"] / "inference_script.py"
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        return "\u2705 Inference script generated -> " + str(path)

    @_audit_tool
    def generate_training_script(self) -> str:
        """Generate a reproducible standalone training script capturing the full pipeline."""
        ptype   = STATE.get("problem_type") or "classification"
        tcol    = STATE.get("target_col") or "target"
        bm_name = STATE.get("best_model_name") or "RandomForest"

        cls_import  = "RandomForestClassifier" if ptype == "classification" else "RandomForestRegressor"
        met_import  = "accuracy_score, f1_score" if ptype == "classification" else "r2_score, mean_squared_error"
        eval_line   = ("print(" + repr("Accuracy:") + ", accuracy_score(y_te, y_pred))"
                       if ptype == "classification"
                       else "print(" + repr("R2:") + ", r2_score(y_te, y_pred))")

        lines = [
            '''"""''',
            "Auto-generated training script",
            "Problem type : " + ptype,
            "Best model   : " + bm_name,
            "Generated by : ML Single-Agent (phidata 2.7.10)",
            '''"""''',
            "import pickle, warnings",
            "import pandas as pd, numpy as np",
            "from pathlib import Path",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.impute          import SimpleImputer",
            "from sklearn.preprocessing  import RobustScaler, LabelEncoder",
            "from sklearn.ensemble        import " + cls_import,
            "from sklearn.metrics         import " + met_import,
            "",
            "warnings.filterwarnings(" + repr("ignore") + ")",
            "OUTPUT_DIR = Path(" + repr("ml_output") + "); OUTPUT_DIR.mkdir(exist_ok=True)",
            "",
            "DATA_PATH  = " + repr("your_dataset.csv") + "  # <-- change this",
            "TARGET_COL = " + repr(tcol),
            "df = pd.read_csv(DATA_PATH)",
            "",
            "X = df.drop(columns=[TARGET_COL])",
            "y = df[TARGET_COL]",
            "",
            "le_target = None",
            "if y.dtype == " + repr("object") + ":",
            "    le_target = LabelEncoder(); y = le_target.fit_transform(y)",
            "",
            "X = pd.get_dummies(X, drop_first=True)",
            "",
            "imp = SimpleImputer(strategy=" + repr("median") + ")",
            "X   = pd.DataFrame(imp.fit_transform(X), columns=X.columns)",
            "",
            "scaler   = RobustScaler()",
            "X_scaled = scaler.fit_transform(X)",
            "",
            "X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)",
            "",
            "model = " + cls_import + "(n_estimators=100, random_state=42)",
            "model.fit(X_tr, y_tr)",
            "",
            "y_pred = model.predict(X_te)",
            eval_line,
            "",
            "with open(OUTPUT_DIR / " + repr("best_model.pkl") + ", " + repr("wb") + ") as f:",
            "    pickle.dump(model, f)",
            "artifacts = {" + repr("scaler") + ": scaler, " + repr("imputer") + ": imp,",
            "             " + repr("le_target") + ": le_target, " + repr("feature_cols") + ": list(X.columns)}",
            "with open(OUTPUT_DIR / " + repr("pipeline_artifacts.pkl") + ", " + repr("wb") + ") as f:",
            "    pickle.dump(artifacts, f)",
            "print(" + repr("Model and artifacts saved.") + ")",
        ]
        code = "\n".join(lines) + "\n"

        path = STATE["output_dir"] / "training_script.py"
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        return "\u2705 Training script generated -> " + str(path)



# ══════════════════════════════════════════════════════════
#  AGENT ASSEMBLY
# ══════════════════════════════════════════════════════════
def build_ml_agent(
    model_id    : str           = None,
    pipeline_state: PipelineState = None,
) -> Agent:
    """Build and return a configured ML pipeline agent.

    Args:
        model_id: OpenAI model ID. Reads OPENAI_MODEL_ID env-var; falls back to gpt-4o-mini.
        pipeline_state: Optional pre-configured PipelineState instance for this run.
                        Provide a fresh PipelineState() per run to ensure isolation.

    Returns: configured phidata Agent ready to call agent.print_response(prompt).
    """
    import os

    # ── Model ID: env-var > argument > safe default ────────────────────────
    resolved_model = model_id or os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")

    # ── Bind the provided state (or the module default) ────────────────────
    global STATE
    if pipeline_state is not None:
        STATE = pipeline_state
    STATE.output_dir.mkdir(exist_ok=True)

    agent = Agent(
        name  = "ML-Pipeline-Agent",
        model = OpenAIChat(id=resolved_model),
        tools = [
            EDATookit(),
            ImputationToolkit(),
            OutlierToolkit(),
            ScalingToolkit(),
            EncodingToolkit(),
            ModellingToolkit(),
            SelectionToolkit(),
            PicklingToolkit(),
            CodegenToolkit(),
        ],

        # ── Rich description: who the agent is and its scope ──────────────
        description = (
            "You are a senior ML engineer and data scientist specialising in supervised "
            "learning on tabular datasets. You run structured, reproducible ML pipelines "
            "covering EDA, data cleaning, feature engineering, model training, evaluation, "
            "and deployment artefact generation. You work exclusively with CSV and Excel "
            "files. You produce concise, plain-English summaries after every stage alongside "
            "the technical output. You always flag data-quality issues (missing values, "
            "class imbalance, suspicious columns) before proceeding to the next step, and "
            "you ask for clarification when a decision requires business context."
        ),

        instructions = [
            # ── Role and audience ─────────────────────────────────────────
            "Communicate as a senior ML engineer. After every completed stage write a "
            "one-sentence plain-English summary of what was done and what was found, "
            "then show the technical details.",

            # ── Output format ─────────────────────────────────────────────
            "At the end of the full pipeline, produce a markdown summary table with "
            "columns: Stage | Status | Key Metric | Notes.",

            # ── Strict step order ─────────────────────────────────────────
            "Follow this exact order for every ML pipeline run:",
            "  1. EDA      → load_dataset → dataset_overview → missing_value_report "
            "→ statistical_summary → correlation_heatmap → target_distribution "
            "→ detect_problem_type",
            "  2. Impute   → impute_numeric (strategy=median) → impute_categorical "
            "(strategy=most_frequent) drop columns if 50% more are NULL values",
            "  3. Outliers → detect_outliers_iqr → treat_outliers_iqr_clip",
            "  4. Scale    → robust_scale by default; use standard_scale only if the "
            "user explicitly requests it",
            "  5. Encode   → handle_datetime_features → encode_target "
            "→ onehot_encode → label_encode (for any remaining object columns)",
            "  6. Model    → prepare_train_test_split → train_all_models",
            "  7. Select   → evaluate_all_on_test → select_best_model "
            "→ detailed_report → feature_importance_plot",
            "  8. Pickle   → save_best_model → save_pipeline_artifacts",
            "  9. Codegen  → generate_inference_script → generate_training_script",

            # ── Error handling ────────────────────────────────────────────
            "CRITICAL: If any tool returns a string beginning with 'ERROR', stop "
            "immediately and report the error to the user. Do NOT proceed to the next "
            "stage until the error is resolved.",

            # ── Data-quality gates ────────────────────────────────────────
            "Before imputation: if missing_value_report shows any column with >50% "
            "missing values, warn the user and ask whether to drop the column or impute.",
            "Before modelling: if the target shows extreme class imbalance "
            "(minority class <5%), warn the user and suggest SMOTE or class_weight='balanced'.",
            "Before modelling: if fewer than 50 training rows remain after preprocessing, "
            "warn the user that results may be unreliable.",

            # ── Problem-type override ─────────────────────────────────────
            "If the user indicates that detect_problem_type gave the wrong result, "
            "call set_problem_type() with the correct value before continuing.",

            # ── Final check ───────────────────────────────────────────────
            "Always call detect_problem_type before any modelling step. "
            "Do not skip stages. Report results clearly after each stage.",
        ],

        show_tool_calls    = True,
        markdown           = True,
        add_history_to_messages = True,   # maintain context across turns
    )
    return agent


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════
def _generate_demo_dataset(path: str = "demo_classification.csv") -> str:
    """Create a small demo dataset if no real file is provided."""
    import numpy as np
    np.random.seed(42)
    n = 400
    df = pd.DataFrame({
        "age"       : np.random.randint(18, 70, n).astype(float),
        "income"    : np.random.exponential(40000, n).round(2),
        "tenure"    : np.random.randint(0, 20, n),
        "product"   : np.random.choice(["A", "B", "C"], n),
        "region"    : np.random.choice(["North", "South", "East", "West"], n),
        "churn"     : np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    # inject some missing values
    df.loc[np.random.choice(n, 30, replace=False), "age"]    = float("nan")
    df.loc[np.random.choice(n, 15, replace=False), "income"] = float("nan")
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    import sys

    # Each run gets its own isolated state
    run_state = PipelineState()
    agent     = build_ml_agent(pipeline_state=run_state)

    if len(sys.argv) >= 3:
        dataset_path  = sys.argv[1]
        target_column = sys.argv[2]

        # Pre-read column list to give the LLM richer context upfront
        try:
            cols = pd.read_csv(dataset_path, nrows=0).columns.tolist()
            col_hint = f"Available columns: {cols}."
        except Exception:
            col_hint = ""

        prompt = (
            f"Run the full ML pipeline on the dataset at '{dataset_path}' "
            f"with target column '{target_column}'. {col_hint} "
            f"Follow all steps in order: EDA, imputation, outlier treatment, scaling, Drop columns with more than 50% NULL values "
            f"encoding, train all models, select the best model, pickle it, and generate "
            f"the inference and training scripts. Report results clearly after each stage."
        )

    else:
        # No args — generate a demo dataset so the agent runs out of the box
        demo_path = _generate_demo_dataset("demo_classification.csv")
        print(f"No dataset provided — created demo file: {demo_path}")
        prompt = (
            f"Run the full ML pipeline on the dataset at '{demo_path}' "
            f"with target column 'churn'. "
            f"The dataset contains customer features: age, income, tenure (numeric), "
            f"product and region (categorical), and a binary churn target. "
            f"Follow all pipeline steps in order and produce a final summary table."
        )

    print("ML Agent starting...\n")
    agent.print_response(prompt, stream=True)
