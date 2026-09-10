from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from src.config import (
    # XGBoost
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
    XGB_MIN_CHILD_WEIGHT,
    XGB_GAMMA,
    XGB_REG_ALPHA,
    XGB_REG_LAMBDA,
    XGB_EVAL_METRIC,
    MODEL_RANDOM_STATE
)


def split_data(df):
    X = df.drop(["Class", "hour", "Amount"], axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=11,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight=None
    )

    model.fit(X_train, y_train)

    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        gamma=XGB_GAMMA,
        reg_alpha=XGB_REG_ALPHA,
        reg_lambda=XGB_REG_LAMBDA,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
        eval_metric=XGB_EVAL_METRIC
    )

    model.fit(X_train, y_train)

    return model