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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

# -- Data loading ---------------------------------------------------------------


def read_file(yaml_path, inp_data_section, file_name):
    """Load a CSV whose path is stored in a YAML config.
    Returns a DataFrame, or None on a handled error."""

    #Read the YAML config
    try:
        with open(yaml_path, "r") as file:
            cfg = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file not found: {yaml_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Could not parse YAML: {e}")
        return None

    #Look up the CSV path inside the config
    try:
        csv_path = cfg[inp_data_section][file_name]
    except KeyError as e:
        print(f"Missing key in config: {e}")
        return None

    #Load the CSV (sep=None handles Windows CRLF/BOM exports)
    return pd.read_csv(csv_path, sep=None, engine="python")


def out_csv(df,yaml_path,output_section_yaml,file_name):
    try:
        with open(yaml_path, "r") as file:
            cfg = yaml.safe_load(file)
    except:
        print("Yaml configuration file not found!")
        return None
        
    df.to_csv(cfg[output_section_yaml][file_name], index=False)
    print(f"File saved to: {cfg[output_section_yaml][file_name]}")

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

    # -- Models creation ----------------------------------------------------

def knn_modelling(train_df, test_df, y_train, y_test):
    knn = KNeighborsRegressor()
    knn.fit(train_df, y_train)
    return f"The R2 of the model is {knn.score(test_df, y_test):.2f}"

def linear_regmodelling(train_df, test_df, y_train, y_test):

    lin_reg = LinearRegression()
    lin_reg.fit(train_df, y_train) # Determines the b0 and b1's values
    y_pred_test_lr = lin_reg.predict(test_df)
    print(f"MAE {mean_absolute_error(y_pred_test_lr, y_test): .2f}") # mean(abs(error)) = mean(abs(y_test - y_pred_test))
    print(f"MSE,  {mean_squared_error(y_pred_test_lr, y_test): .2f}")  # mean( (y_test - y_pred_test)^2 )
    print(f"RMSE, {root_mean_squared_error(y_pred_test_lr, y_test): .2f}") # sqrt(MSE), same units as salary_in_usd
    print(f"R2 score, {lin_reg.score(test_df, y_test): .2f}")
    return y_pred_test_lr

def RanForeRegr_modelling(train_df, test_df, y_train, y_test):
    PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'random_state': 42}
    
    rf = RandomForestRegressor(**PARAMS).fit(train_df,y_train)
 
    y_pred = rf.predict(test_df)
    r2_rf = rf.score(test_df, y_test)

    print(f"  R2 Score: {r2_rf:.4f}")
    print(f"  MAE:      {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"  MSE:      {mean_squared_error(y_test, y_pred):.2f}")
    print(f"  RMSE:     {root_mean_squared_error(y_test, y_pred):.2f}")  

    return y_pred

def bagreg_RanForeRegr_modelling(train_df, test_df, y_train, y_test):
    PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'random_state': 42}
    
    bagrf = BaggingRegressor(RandomForestRegressor(**PARAMS)).fit(train_df,y_train)
 
    y_pred = bagrf.predict(test_df)
    r2_bagrf = bagrf.score(test_df, y_test)

    print(f"  R2 Score: {r2_bagrf:.4f}")
    print(f"  MAE:      {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"  MSE:      {mean_squared_error(y_test, y_pred):.2f}")
    print(f"  RMSE:     {root_mean_squared_error(y_test, y_pred):.2f}")  

    return y_pred


