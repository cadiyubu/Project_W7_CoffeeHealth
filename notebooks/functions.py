"""
functions.py - Shared helper functions for ML Project Week 7

All reusable logic lives here. Import into notebooks with:
    from functions import *
"""

import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -- Data loading ---------------------------------------------------------------

def load_raw_data(path, **kwargs):
    """
    Load raw CSV. Uses sep=None + python engine to handle
    Windows CRLF/BOM exports safely.
    """
    return pd.read_csv(path, sep=None, engine="python", **kwargs)


# -- Config / data source loader ------------------------------------------------

def read_file(yaml_path, inp_data_section, file_name):
    try:
        with open(yaml_path, "r") as file:
            cfg = yaml.safe_load(file)
    except:
        print("Yaml configuration file not found!")
        return None

    raw_df = pd.read_csv(cfg[inp_data_section][file_name])  # ✅
    return raw_df


# -- EDA helpers ----------------------------------------------------------------

def summarise_dataframe(df):
    """Print shape, dtypes, null counts, and basic stats."""
    print(f"Shape: {df.shape}")
    print("\n--- Null counts ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\n--- Dtypes ---")
    print(df.dtypes)
    print("\n--- Describe ---")
    return df.describe()


# -- Modelling preprocessing ----------------------------------------------------

def features_target_extract(df, drop_cols, target_col):
    df_model = df.drop(columns=drop_cols, errors='ignore')
    if target_col in df_model.columns:
        features = df_model.drop(columns=[target_col])
        target = df_model[target_col]
        return df_model, features, target
    else:
        # test set: no target — return features only
        print('No target column; returning features only')
        return df_model, df_model, None


def dummies_catcols(X_train,X_test,nonnum_cols):
    #dummies for non-numerical columns, for feature train data
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, drop='first') # To avoid having an sparse_matrix as output
    
    ohe.fit(X_train[nonnum_cols]) # The .fit() method determines the unique values of each column
    X_train_dummified_np = ohe.transform(X_train[nonnum_cols])
    X_train_dummified_df = pd.DataFrame(X_train_dummified_np, columns = ohe.get_feature_names_out(nonnum_cols), index=X_train.index)
    
    #Now for the test set
    X_test_dummified_np = ohe.transform(X_test[nonnum_cols])
    X_test_dummified_df=pd.DataFrame(X_test_dummified_np, columns=ohe.get_feature_names_out(nonnum_cols), index=X_test.index)    
    
    return X_train_dummified_df, X_test_dummified_df


def df_full_standarized(X_train_dumm,X_test_dumm,X_train_num,X_test_num):
    dfX_train_full = pd.concat([X_train_dumm, X_train_num], axis=1)
    dfX_test_full  = pd.concat([X_test_dumm, X_test_num], axis=1)
    
    from sklearn.preprocessing import StandardScaler
    
    std_scaler = StandardScaler()
    std_scaler.fit(dfX_train_full)
    
    dfX_train_full_np = std_scaler.transform(dfX_train_full)
    dfX_test_full_np = std_scaler.transform(dfX_test_full)
    
    dfX_train_full_np_df = pd.DataFrame(dfX_train_full_np, columns=dfX_train_full.columns, index=dfX_train_full.index)
    dfX_test_full_np_df  = pd.DataFrame(dfX_test_full_np, columns=dfX_test_full.columns, index=dfX_test_full.index)

    return dfX_train_full_np_df, dfX_test_full_np_df
