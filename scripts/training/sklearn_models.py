import numpy as np
import pandas as pd
import optuna as opt
import xgboost as xgb
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imlearn.Pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def _return_lr_grid (opt: opt.Trial):
    solver = opt.suggest_categorical(name="lr_solver", choices=["lbfgs", "saga"])
    
    if solver == "lbfgs":
        l1_ratio = 0
    else:
        l1_ratio = 0.50

    return {
        "l1_ratio":     l1_ratio,
        "solver":       solver,
        "max_iter":     opt.suggest_int(name="lr_max_iter", min=100, max=500, step=50),
        "tol":          opt.suggest_float(name="lr_tol", , min=1e-7, max=1e-2, log=True),
        "random_state": 42,
        "n_jobs":       50
    }

def _return_dt_grid (opt: opt.Trial):
    return {
        "criterion":                opt.suggest_categorical("dt_criterion", ["gini", "entropy"]),
        "max_depth":                opt.suggest_int("dt_max_depth", 5, 50, 5),
        "min_samples_split":        opt.suggest_int("dt_min_samples_split", 50, 100, 10),
        "min_samples_leaf":         opt.suggest_int("dt_min_samples_leaf", 60, 90, 10),
        "max_features":             opt.suggest_categorical("dt_max_features", ["sqrt", "log2", None]),
        "max_leaf_nodes":           opt.suggest_int("dt_max_leaf_nodes", 10, 100),
        "min_impurity_decrease":    opt.suggest_float("dt_min_impurity_decrease", 0.0, 0.1, log=True),
        "ccp_alpha":                opt.suggest_float("dt_ccp_alpha", 0.0, 0.05, log=True),
    }

def _return_xgb_grid (opt: opt.Trial):
    return {
        # Boosting structure
        "n_estimators":         opt.suggest_int("xgb_n_estimators", 50, 500, step=50),
        "max_depth":            opt.suggest_int("xgb_max_depth", 3, 10),
        "max_leaves":           opt.suggest_int("xgb_max_leaves", 0, 64),
 
        # Learning
        "learning_rate":        opt.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True),
        "booster":              opt.suggest_categorical("xgb_booster", ["gbtree", "dart"]),
 
        # Regularization
        "reg_alpha":            opt.suggest_float("xgb_reg_alpha", 1e-4, 10.0, log=True),   # L1
        "reg_lambda":           opt.suggest_float("xgb_reg_lambda", 1e-4, 10.0, log=True),  # L2
        "gamma":                opt.suggest_float("xgb_gamma", 0.0, 5.0),
        "min_child_weight":     opt.suggest_int("xgb_min_child_weight", 1, 20),
 
        # Sampling 
        "subsample":            opt.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree":     opt.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        "colsample_bylevel":    opt.suggest_float("xgb_colsample_bylevel", 0.5, 1.0),
        "colsample_bynode":     opt.suggest_float("xgb_colsample_bynode", 0.5, 1.0),
 
        # Tree method
        "tree_method":          opt.suggest_categorical("xgb_tree_method", ["hist", "approx"]),
        "grow_policy":          opt.suggest_categorical("xgb_grow_policy", ["depthwise", "lossguide"]),
    }

def _optuna_trial (
    opt_trial,
    model_class,
    hyperparameter_grid,
    strat_kfold,
    train_x: pd.DataFrame,
    train_y: pd.DataFrame
):
    precision_score_array = []
    model = model_class(**hyperparameter_grid)
    for tr_idx, ts_idx in strat_kfold.split(train_x, train_y):
        tr_x_subset = train_x.iloc[tr_idx, :]
        tr_y_subset = train_x.iloc[tr_idx, :]
        ts_x_subset = train_x.iloc[ts_idx, :]
        ts_y_subset = train_y.iloc[ts_idx, :]

        model.fit(tr_x_subset, tr_y_subset)
        ts_x_preds = model.predict(ts_x_subset)
        precision_score_array.append(precision_score(ts_y_subset, ts_x_preds, average="weighted"))

    return np.mean(np.asarray(precision_score_array))

def main ():
    csv_paths_dataset = [
        "../../datasets/original-datasets/labelled_2021may-ip-10-100-1-105.csv"
        "../../datasets/original-datasets/labelled_2021may-ip-10-100-1-186.csv"
        "../../datasets/original-datasets/labelled_2021may-ip-10-100-1-26.csv"
        "../../datasets/original-datasets/labelled_2021may-ip-10-100-1-4.csv"
        "../../datasets/original-datasets/labelled_2021may-ip-10-100-1-95.csv"
    ]
    cols_to_standardize = [
        "timestamp", 
        "processId", 
        "userId", 
        "mountNamespace", 
        "eventId",
        "stackAddresses",
        "argsNum"
    ]
    main_pd = pd.read_csv("../../datasets/original-datasets/labelled_training_datasets.csv")

    for dataset_path in csv_paths_dataset:
        temporary_pd = pd.read_csv(dataset_path)
        main_pd = pd.concat([main_pd, temporary_pd])

    main_ct = ColumnTransformer([
        ("ct_create_features", TargetEncoder(target_type="binary", random_state=42), ["stackAddresses"]),
        ("ct_standardize_cols", StandardScaler(), cols_to_standardize),
    ])

    main_lr_pipeline = Pipeline([
        ("sklearn_ct_preprocessors", main_ct),
        ("imblearn_preprocessor", SMOTE(random_state=42)),
        ("sklearn_logistic", LogisticRegression())
    ])

    main_dt_pipeline = Pipeline([
        ("sklearn_ct_preprocessors", main_ct),
        ("imblearn_preprocessor", SMOTE(random_state=42)),
        ("sklearn_tree", DecisionTreeClassifier())
    ])

    main_xgb_pipeline = Pipeline([
        ("sklearn_ct_preprocessors", main_ct),
        ("imblearn_preprocessor", SMOTE(random_state=42)),
        ("sklearn_forest", xgb.XGBClassier())
    ])
    
    strat_kfold = StratifiedKFold(shuffle=True, random_state=42)
    lr_study = opt.create_study("LOGISTIC REGRESSION STUDY")
    dt_study = opt.create_study("DECISION TREE CLASSIFIER STUDY")
    xgb_study = opt.create_study("XGBCLASSIFIER STUDY")

    optuna_studies = [lr_study, dt_study, xgb_study]
    pipeline_list = [main_lr_pipeline, main_dt_pipeline, main_xgb_pipeline]
    param_grid_list = [_return_lr_grid, _return_dt_grid, _return_xgb_grid]

    for stdy, pipeline, param_ref in zip(optuna_studies, pipeline_list, param_grid_ref):
        print(f"\n\nStudy: {stdy} \n\n")
        stdy.optimize(
            lambda trial: _optuna_trial(
                trial,
                param_ref(trial),
                strat_kfold,
                train_x,
                train_y
            ),
            n_trials=2000,
            n_jobs=60
        )

if __name__ == "__main__":
    main()

