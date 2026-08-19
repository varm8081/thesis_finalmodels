"""
Data loading + leakage-free preprocessing for the distress (Class 0) project.

Design rules (per PROJECT_STATE methodology):
  * Temporal split: Train <= 1399 | Val 1400-1401 | Test 1402-1403  -> NO leakage
  * Median imputation fit on TRAIN only
  * StandardScaler fit on TRAIN only
  * SMOTE applied to TRAIN only
The validation and test sets are passed through the *already-fitted* transformers.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from common import (RAW_XLSX, TARGET, ID_COLS, TRAIN_MAX_YEAR, VAL_YEARS,
                    TEST_YEARS, log)


def load_raw():
    df = pd.read_excel(RAW_XLSX, sheet_name=0)
    log(f"Loaded raw data: {df.shape[0]} rows x {df.shape[1]} cols", "OK")
    return df


def make_class0_target(df):
    """Binary target: 1 = distress (Class 0 of the 3-zone target), 0 = not."""
    y = (df[TARGET] == 0).astype(int)
    return y


def temporal_split(df):
    yr = df["Year"].astype(int)
    train = df[yr <= TRAIN_MAX_YEAR].copy()
    val = df[yr.isin(VAL_YEARS)].copy()
    test = df[yr.isin(TEST_YEARS)].copy()
    log(f"Temporal split -> Train:{train.shape[0]}  Val:{val.shape[0]}  Test:{test.shape[0]}",
        "OK")
    return train, val, test


def build_pipeline(df, feature_cols, cat_cols=None, smote=True, seed=42,
                   scale=True):
    """
    Returns a dict with fitted transformers + the split frames, ready for
    any model. Imputation/scaling fit on train only; val/test transformed.
    SMOTE applied to train only.
    """
    cat_cols = cat_cols or []
    train, val, test = temporal_split(df)
    y_train = make_class0_target(train)
    y_val = make_class0_target(val)
    y_test = make_class0_target(test)

    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()
    X_test = test[feature_cols].copy()

    # 1) Median imputation (train only)
    imp = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=feature_cols,
                               index=X_train.index)
    X_val_imp = pd.DataFrame(imp.transform(X_val), columns=feature_cols,
                             index=X_val.index)
    X_test_imp = pd.DataFrame(imp.transform(X_test), columns=feature_cols,
                              index=X_test.index)

    # 2) Scaling (train only) -- tree models ignore this, but keep for SVM/LR
    if scale:
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=feature_cols,
                                index=X_train_imp.index)
        X_val_s = pd.DataFrame(scaler.transform(X_val_imp), columns=feature_cols,
                               index=X_val_imp.index)
        X_test_s = pd.DataFrame(scaler.transform(X_test_imp), columns=feature_cols,
                                index=X_test_imp.index)
    else:
        scaler = None
        X_train_s, X_val_s, X_test_s = X_train_imp, X_val_imp, X_test_imp

    # 3) SMOTE on train only (balance the distress minority class)
    if smote:
        n_minor = int(y_train.sum())
        k_neigh = min(5, max(1, n_minor - 1))
        sm = SMOTE(sampling_strategy="auto", random_state=seed, k_neighbors=k_neigh)
        X_res, y_res = sm.fit_resample(X_train_s, y_train)
        log(f"SMOTE: train {y_train.sum()}/{len(y_train)} distress -> balanced "
            f"{int(y_res.sum())}/{len(y_res)}", "OK")
    else:
        X_res, y_res = X_train_s, y_train

    return {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "imputer": imp,
        "scaler": scaler,
        "X_train": X_train_s, "y_train": y_train,
        "X_res": X_res, "y_res": y_res,
        "X_val": X_val_s, "y_val": y_val,
        "X_test": X_test_s, "y_test": y_test,
        "raw_train": train, "raw_val": val, "raw_test": test,
    }


if __name__ == "__main__":
    from common import feature_set_sizes
    df = load_raw()
    print("Feature set sizes:")
    for k, v in feature_set_sizes().items():
        print(f"  {k}: {v}")
    split = build_pipeline(df, feature_set_sizes()["Group_B_Financial_Health"])
    print("val distress rate:", round(split["y_val"].mean(), 3))
    print("test distress rate:", round(split["y_test"].mean(), 3))
