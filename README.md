# AI/ML/Data Salaries — Prediction (ML Week 7)

## Project Description

Predict `salary_in_usd` for global AI / ML / Data roles using supervised regression.
Two complementary models are built on the same cleaned dataset:

- **Model 1 (Diana)** — company-side features: `job_title`, `company_size`, `company_region`, `work_year`.
- **Model 2 (Bibian)** — experience-side features, anchored on `experience_level`, optimised separately with Random Forest + Optuna.

Both halves use the same train/test split and the same Random Forest + Bayesian (Optuna) tuning approach, so results are directly comparable once Bibian's half is complete (see **Pending: Joint Model Comparison** below).

## Data Sources

- **aijobs.net Global AI/ML/Data Salaries** (~34k rows, CC0 licence)
- Direct CSV download: https://aijobs.net/salaries/download/
- Raw rows: 29,562. After the 80/20 train/test split: 23,649 rows in the training set.


## Techniques Used

- Feature engineering: country → region grouping (`company_region`), rare job-title collapsing (titles with fewer than 150 occurrences grouped into `'Other'`), derived `companyloc_residence` boolean flag
- Feature selection: boxplots by category + one-way ANOVA (`f_oneway`) — all three categorical features rejected H0 (p < 0.05) and were kept
- Encoding & scaling: `OneHotEncoder(drop='first')` on nominal columns + `StandardScaler` on the combined numeric + dummy matrix
- Regression models tried: KNeighbors, Linear Regression, Random Forest, Bagging Regressor (Random Forest base)
- Hyperparameter tuning: Bayesian optimisation (Optuna, TPE sampler) on Random Forest — 45 trials, 10-fold cross-validation, tuning `max_depth`, `min_samples_split`, `max_leaf_nodes`, `max_features`
- Evaluation: R², MAE, MSE, RMSE — reported on both a single 80/20 test split and 10-fold cross-validation (kept clearly separate; see note below)

## Key Result (Diana's half)

All four model families converge to a similar ceiling: **R² ≈ 0.20–0.25** depending on model and evaluation method. Plain Random Forest and the Optuna-tuned Random Forest land within noise of each other once both are evaluated under the same 10-fold CV procedure (~0.217 vs ~0.220). This convergence across structurally different algorithms (distance-based, linear, and tree-based) points to a **feature information ceiling**, not a model-choice or tuning problem — the next real lever is `experience_level`, which sits in Bibian's half of the feature set.


## Pending: Joint Model Comparison

TBC by Bibian

## Functions (`notebooks/functions.py`)

| Function | Signature | Purpose |
|---|---|---|
| `read_file` | `read_file(yaml_path, inp_data_section, file_name)` | Read the YAML config and load the CSV at the path it points to (`sep=None, engine='python'` handles CRLF/BOM from non-Mac exports). Returns `None` on a missing file or key. |
| `out_csv` | `out_csv(df, yaml_path, output_section_yaml, file_name)` | Write a cleaned DataFrame to the path configured in the YAML's output section. |
| `summarise_dataframe` | `summarise_dataframe(df)` | Print shape, null counts, and dtypes; return `df.describe()`. |
| `features_target_extract` | `features_target_extract(df, drop_cols, target_col)` | Drop unused columns, then split into `features` and `target`. Pass `drop_cols=[]` when nothing needs dropping. |
| `dummies_catcols` | `dummies_catcols(X_train, X_test, nonnum_cols)` | One-hot encode the nominal columns (`drop='first'`). Fits the encoder on train only, then transforms both train and test; preserves the original index. |
| `df_full_standarized` | `df_full_standarized(X_train_dumm, X_test_dumm, X_train_num, X_test_num)` | Concatenate dummified + numeric columns and apply `StandardScaler` (fit on train, transform both). Returns scaled train/test DataFrames. |
| `knn_modelling` | `knn_modelling(train_df, test_df, y_train, y_test)` | Fit a default `KNeighborsRegressor`, return a formatted R² string. |
| `linear_regmodelling` | `linear_regmodelling(train_df, test_df, y_train, y_test)` | Fit `LinearRegression`, print MAE/MSE/RMSE/R² on the test set, return predictions. |
| `RanForeRegr_modelling` | `RanForeRegr_modelling(train_df, test_df, y_train, y_test)` | Fit `RandomForestRegressor` (`n_estimators=100, max_depth=20, random_state=42`), print R²/MAE/MSE/RMSE, return predictions. |
| `bagreg_RanForeRegr_modelling` | `bagreg_RanForeRegr_modelling(train_df, test_df, y_train, y_test)` | Fit a `BaggingRegressor` wrapping the same Random Forest params, print R²/MAE/MSE/RMSE, return predictions. |


## How to Run

```bash
# 1. Navigate to project root
cd "DataAnalytics Ironhack/Week7/Project_ML_W7"

# 2. Sync the environment (project .venv has historically been incomplete —
#    verify `optuna` is installed before running model_building_diana.ipynb)
uv sync

# 3. Register the Jupyter kernel (first time only)
python -m ipykernel install --user --name=venv

# 4. Launch Jupyter
jupyter lab
```

Run notebooks in order:
1. `notebooks/explore_clean_modelingv2_diana.ipynb` — EDA, feature engineering, feature selection (ANOVA), preprocessing, base model comparison (KNN / Linear / Random Forest / Bagging-RF), exports the cleaned feature set to `data/clean/X_df_ModelDiana.csv`.
2. `notebooks/model_building_diana.ipynb` — loads the exported clean data, re-fits baseline Linear Regression and Random Forest, runs Optuna Bayesian tuning on Random Forest (10-fold CV, 45 trials), evaluates the tuned model on the held-out test set, and produces the final model comparison figures.

## Project Structure

```
.
├── README.md
├── config.yaml
├── pyproject.toml
├── uv.lock
├── .gitignore
├── data/
│   ├── raw/
│   │   └── salaries.csv              ← original unaltered dataset
│   └── clean/
│       └── X_df_ModelDiana.csv       ← cleaned/engineered feature set (Diana's half)
├── figures/
│   ├── boxplot_*.png                 ← feature selection boxplots
│   ├── Optuna_optHistory_RFModel.png
│   ├── HyperparameterImportance_RF.png
│   ├── Predictions_Modelscomparison.png
│   └── R2_Modelscomparisons.png
├── notebooks/
│   ├── functions.py                  ← shared helper functions (table above)
│   ├── explore_clean_modelingv2_diana.ipynb
│   └── model_building_diana.ipynb
└── slides/                           ← presentation (placeholders for Bibian's results)
```

## Contributors
- Diana Yule — company-feature model, EDA, feature engineering, base model comparison, Optuna tuning
- Bibian — experience-feature model (in progress)

## Links
- Presentation: https://docs.google.com/presentation/d/1gQCRVV1ZCu3K6sQIn_v6LvVm_OvJhK4pV4hXGplZq1o/edit?slide=id.p9#slide=id.p9
- Kanban board: 

