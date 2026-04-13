from typing import Any
import numpy as np
import pandas as pd
import optuna as opt
import xgboost as xgb
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer

def _create_datasets (for_train: bool = False, for_actual: bool = False):
    csv_paths_dataset = [
        "../../datasets/original-datasets/labelled_2021may-ip-10-100-1-105.csv",
        "../../datasets/original-datasets/labelled_2021may-ip-10-100-1-186.csv",
        "../../datasets/original-datasets/labelled_2021may-ip-10-100-1-26.csv",
        "../../datasets/original-datasets/labelled_2021may-ip-10-100-1-4.csv",
        "../../datasets/original-datasets/labelled_2021may-ip-10-100-1-95.csv"
    ]
    columns_to_drop = [
        "threadId",
        "hostName",
        "eventName",
        "returnValue",
        "processName",
        "stackAddresses",
        "mountNamespace",
        "args",
        "sus"
    ]
    main_pd = pd.read_csv("../../datasets/original-datasets/labelled_training_data.csv")

    for dataset_path in csv_paths_dataset:
        temporary_pd = pd.read_csv(dataset_path)
        main_pd = pd.concat([main_pd, temporary_pd])

    main_pd = main_pd.drop(columns_to_drop, axis=1).reset_index(drop=True)

    main_pd_labels, label_counts = np.unique(main_pd.loc[:, "evil"], return_counts=True)
    print(f"Labels: {main_pd_labels} | Label counts: {label_counts}")

    if for_train:
        dataframe_subset = main_pd.sample(frac=0.3)
        return train_test_split(
            dataframe_subset.iloc[:, :-1],
            dataframe_subset.iloc[:, -1],
            train_size=0.5,
            test_size=0.5,
            random_state=42,
        )
    if for_actual:
        return train_test_split(
            main_pd.iloc[:, :-1],
            main_pd.iloc[:, -1],
            train_size=0.8,
            test_size=0.2,
            random_state=42,
        )

def _create_column_transformer ():
    cols_to_standardize = [
        "timestamp", 
        "processId", 
        "userId", 
        "eventId",
        "argsNum"
    ]

    return ColumnTransformer([
        ("ct_standardize_cols", StandardScaler(), cols_to_standardize),
    ])

def _return_lr_pipeline (
    opt: opt.Trial = None, 
    for_train: bool = True, 
    prod_hyperparameters: dict[str, Any] = None
):
    ct = _create_column_transformer()

    solver = opt.suggest_categorical(name="lr_solver", choices=["lbfgs", "saga"])
    
    if solver == "lbfgs":
        l1_ratio = 0
    else:
        l1_ratio = 0.50

    lr_parameter_grid = {
        "l1_ratio":     l1_ratio,
        "solver":       solver,
        "max_iter":     opt.suggest_int(name="lr_max_iter", low=100, high=500, step=50),
        "tol":          opt.suggest_float(name="lr_tol", low=1e-7, high=1e-2, log=True),
        "random_state": 42
    }

    if for_train:
        return Pipeline([
            ("sklearn_ct_preprocessors", ct),
            ("imblearn_preprocessor", SMOTE(sampling_strategy="minority", random_state=42)),
            ("sklearn_logistic", LogisticRegression(**lr_parameter_grid))
        ])
    else:
        return Pipeline([
            ("sklearn_ct_preprocessors", ct),
            ("imblearn_preprocessor", SMOTE(sampling_strategy="minority", random_state=42)),
            ("sklearn_logistic", LogisticRegression(**prod_hyperparameters))
        ])

def _return_dt_pipeline (
    opt: opt.Trial = None, 
    for_train: bool = True, 
    prod_hyperparameters: dict[str, Any] = None
):
    ct = _create_column_transformer()

    dt_parameter_grid = {
        "criterion":                opt.suggest_categorical("dt_criterion", ["gini", "entropy"]),
        "max_depth":                opt.suggest_int("dt_max_depth", 5, 50, 5),
        "min_samples_split":        opt.suggest_int("dt_min_samples_split", 50, 100, 10),
        "min_samples_leaf":         opt.suggest_int("dt_min_samples_leaf", 60, 90, 10),
        "max_features":             opt.suggest_categorical("dt_max_features", ["sqrt", "log2", None]),
        "max_leaf_nodes":           opt.suggest_int("dt_max_leaf_nodes", 10, 100),
        "min_impurity_decrease":    opt.suggest_float("dt_min_impurity_decrease", 0.0, 0.1, log=True),
        "ccp_alpha":                opt.suggest_float("dt_ccp_alpha", 0.0, 0.05, log=True),
    }

    if for_train:
        return Pipeline([
            ("sklearn_ct_preprocessors", ct),
            ("imblearn_preprocessor", SMOTE(sampling_strategy="minority", random_state=42)),
            ("sklearn_tree", DecisionTreeClassifier(**dt_parameter_grid))
        ])
    else:
        return Pipeline([
            ("sklearn_ct_preprocessors", ct),
            ("imblearn_preprocessor", SMOTE(sampling_strategy="minority", random_state=42)),
            ("sklearn_tree", DecisionTreeClassifier(**prod_hyperparameters))
        ])

def _return_xgb_pipeline (
    opt: opt.Trial = None, 
    for_train: bool = True, 
    prod_hyperparameters: dict[str, Any] = None
):
    ct = _create_column_transformer()

    xgb_parameter_grid = {
        "n_estimators":         opt.suggest_int("xgb_n_estimators", 50, 500, step=50),
        "max_depth":            opt.suggest_int("xgb_max_depth", 3, 10),
        "max_leaves":           opt.suggest_int("xgb_max_leaves", 0, 64),
        "learning_rate":        opt.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True),
        "booster":              opt.suggest_categorical("xgb_booster", ["gbtree", "dart"]),
        "reg_alpha":            opt.suggest_float("xgb_reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":           opt.suggest_float("xgb_reg_lambda", 1e-4, 10.0, log=True),
        "gamma":                opt.suggest_float("xgb_gamma", 0.0, 5.0),
        "min_child_weight":     opt.suggest_int("xgb_min_child_weight", 1, 20),
        "subsample":            opt.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree":     opt.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        "colsample_bylevel":    opt.suggest_float("xgb_colsample_bylevel", 0.5, 1.0),
        "colsample_bynode":     opt.suggest_float("xgb_colsample_bynode", 0.5, 1.0),
        "tree_method":          opt.suggest_categorical("xgb_tree_method", ["hist", "approx"]),
        "grow_policy":          opt.suggest_categorical("xgb_grow_policy", ["depthwise", "lossguide"]),
    }

    if for_train:
        return Pipeline([
            ("sklearn_ct_preprocessors", ct),
            ("imblearn_preprocessor", SMOTE(sampling_strategy="minority", random_state=42)),
            ("sklearn_forest", xgb.XGBClassifier(**xgb_parameter_grid))
        ])
    else:
        return Pipeline([
            ("sklearn_ct_preprocessors", ct),
            ("imblearn_preprocessor", SMOTE(sampling_strategy="minority", random_state=42)),
            ("sklearn_forest", xgb.XGBClassifier(**prod_hyperparameters))
        ])

def _optuna_trial (
    opt_trial,
    constructed_pipeline,
    train_x: pd.DataFrame,
    train_y: pd.DataFrame
):
    strat_kfold = StratifiedKFold(shuffle=True, random_state=42)
    precision_score_array = []

    for tr_idx, ts_idx in strat_kfold.split(train_x, train_y):
        tr_x_subset = train_x.iloc[tr_idx, :]
        tr_y_subset = train_y.iloc[tr_idx]
        ts_x_subset = train_x.iloc[ts_idx, :]
        ts_y_subset = train_y.iloc[ts_idx]

        constructed_pipeline.fit(tr_x_subset, tr_y_subset)
        ts_x_preds = constructed_pipeline.predict(ts_x_subset)
        precision_score_array.append(precision_score(ts_y_subset, ts_x_preds, average="macro"))

    return np.mean(np.asarray(precision_score_array))

def main ():
    tr_x, ts_x, tr_y, ts_y = _create_datasets(for_train=True)
    tra_x, tsa_x, tra_y, tsa_y = _create_datasets(for_actual=True)

    optuna_studies = ["LOGISTIC REGRESSION STUDY", "DT CLASSIFIER STUDY", "XGBCLASSIFIER STUDY"]
    pipeline_constructor_refs = [_return_lr_pipeline, _return_dt_pipeline, _return_xgb_pipeline]

    for stdy, constructor_ref in zip(optuna_studies, pipeline_constructor_refs):
        # Training loop
        print(f"\n\nStudy: {stdy} \n\n")
        temporary_study_instance = opt.create_study(study_name=stdy)
        temporary_study_instance.optimize(
            lambda trial: _optuna_trial(
                trial,
                constructor_ref(trial),
                tr_x,
                tr_y
            ),
            n_trials=2000,
            n_jobs=40
        )

        # Training the production model
        print(f"\nTRAINING PRODUCTION MODEL ON STUDY: {stdy}")
        temporary_prod_model = constructor_ref(
            for_train=False, 
            prod_hyperparameters=temporary_study_instance.best_params
        )
        temporary_prod_model.fit(tra_x, tra_y)
        prod_model_predictions = temporary_prod_model.predict(tsa_x)
        prod_model_score = precision_score(tsa_y, prod_model_predictions, average="macro")

        with open("a", f"{stdy}.txt") as file:
            file.write(f"Best training score: {temporary_study_instance.best_value}")
            file.write(f"Best production score: {prod_model_score}")
            file.write(f"Best parameters: {temporary_study_instance.best_params}")

if __name__ == "__main__":
    main()

