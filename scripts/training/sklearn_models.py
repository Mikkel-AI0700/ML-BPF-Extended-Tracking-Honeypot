import numpy as np
import pandas as pd
import optuna as opt
from imblearn.over_sampling import SMOTE
from imlearn.Pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def _create_features (X: pd.DataFrame, Y: pd.DataFrame):
    stck_len_zero = X.loc[:, "stackAddresses"].str.count(",") == 0
    stck_len_one = X.loc[:, "stackAddresses"].str.count(",") == 1
    stck_len_greater = X.loc[:, "stackAddresses"].str.count(",") > 2
    stck_mask_len_list = [stck_len_zero, stck_len_one, stck_len_greater]

    for mask_cond, mask_len in zip(range(0, 3), stck_mask_len_list):
        X.loc[mask_len, "stackAddresses"] = mask_cond

    return X, Y

def main ():
    cols_to_standardize = [
        "timestamp", 
        "processId", 
        "userId", 
        "mountNamespace", 
        "eventId",
        "stackAddresses",
        "argsNum"
    ]
    converted_feature_function = FunctionTransformer(_create_features)

    main_ct = ColumnTransformer([
        ("ct_create_features", converted_feature_function(), ["stackAddresses"]),
        ("ct_standardize_cols", StandardScaler(), cols_to_standardize),
        ("ct_oversample_minority", SMOTE(), )
    ])

