# AI/ML/Data Salaries — Prediction (ML Week 7)

## Project Description

As students of the DA bootcamp, we wanted to estimate the salary opportunities the job market could offer after this program. The project predicts `salary_in_usd` for global AI / ML / Data roles using supervised regression, built as two complementary models on the same cleaned dataset:

- **Model 1 (Diana)** — company-side features: `job_title`, `company_size`, `company_region`, `work_year`.
- **Model 2 (Bibian)** — experience-side features: `experience_level`, `employment_type`, `job_title`, `employee_residence`.

Both halves use the same train/test split and the same model family progression (KNN → Linear Regression → Random Forest → Bagging-RF → Optuna-tuned Random Forest), so results are directly comparable. See **Joint Model Comparison** below for the head-to-head result.

## Data Sources

- **aijobs.net Global AI/ML/Data Salaries** (~34k rows, CC0 licence)
- Direct CSV download: https://aijobs.net/salaries/download/
- Raw rows: 29,562. After the 80/20 train/test split: 23,649 rows in the training set.


## Techniques Used

- Feature engineering: country → region grouping (`company_region`), rare job-title collapsing (titles with fewer than 150 occurrences grouped into `'Other'`), derived `companyloc_residence` boolean flag (Model 1)
- Feature selection: boxplots by category + one-way ANOVA (`f_oneway`) — for both models, all candidate categorical features reject H0 (p < 0.05): Model 1 keeps `job_title`, `company_region`, `company_size`; Model 2 keeps `experience_level`, `employment_type`, `job_title`, `employee_residence`
- Encoding & scaling: `OneHotEncoder(drop='first')` on nominal columns + `StandardScaler` on the combined numeric + dummy matrix
- Regression models tried (both models): KNeighbors, Linear Regression, Random Forest, Bagging Regressor (Random Forest base)
- Hyperparameter tuning: Bayesian optimisation (Optuna, TPE sampler) on Random Forest for both models — Model 1: 45 trials / 10-fold CV; Model 2: 30 trials / 10-fold CV — tuning `max_depth`, `min_samples_split`, `max_leaf_nodes`, `max_features`
- Evaluation: R², MAE, MSE, RMSE — reported on both a single 80/20 test split and 10-fold cross-validation (kept clearly separate; see note below)

## Key Results

**Model 1 (company-side features)** — three model families converge to a similar ceiling: **R² ≈ 0.24–0.25** on the single test-split (Linear Regression 0.240, Random Forest baseline 0.241, Random Forest Optuna-tuned 0.246). KNeighbors lags at R² ≈ 0.185. Under 10-fold CV, the Optuna-tuned Random Forest's mean R² is **0.2171** (95% CI: 0.2068–0.2329) — within noise of the untuned baseline (~0.003 R² gain).

**Model 2 (experience-side features)** — converges to a higher ceiling: **R² ≈ 0.27–0.28** on the single test-split (KNN 0.221, Linear Regression 0.274, Random Forest baseline 0.280, Bagging-RF 0.281, Optuna-tuned RF 0.281). MAE drops from ~46,500 USD (KNN) to ~44,050–44,096 USD for the other four models. Optuna gain over baseline is negligible here too.

Both models show the same diagnostic pattern: structurally different algorithms (distance-based, linear, tree-based) converge to the same R² ceiling for their respective feature set, and Bayesian tuning adds at most ~0.005 R² in either case. This is a **feature-ceiling signal, not a model-choice or tuning problem.**

> Single test-split R² and cross-validated mean R² are deliberately reported separately, never mixed. A single 80/20 split can land 2–3 points higher or lower purely from sampling luck (e.g. Model 1's Random Forest reads 0.241 on the single split vs 0.217 under CV). The single-split model-comparison charts (`figures/R2_Modelscomparisons.png` and Bibian's equivalent) are single-split; the Optuna confidence-interval figures are cross-validated.

## Joint Model Comparison

Model 2 (experience-side features) outperforms Model 1 (company-side features) by roughly **0.03–0.04 R²** across every matched model family (e.g. Optuna-tuned RF: 0.281 vs 0.246). `experience_level` and `employment_type` carry more salary signal on their own than the company-side feature set — consistent with the column-importance chart on the Team Split slide, where `job_title` (0.58) and `experience_level` (0.21) dominate over `employee_residence` (0.15) and `employment_type` (~0).

Neither model individually reaches R² above ~0.28 — confirming that for this dataset, salary variance is only partially explained by either feature group alone. A combined model (both feature sets together) was identified as a natural next step but was **not built**, due to time constraints ahead of this presentation.


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
3. `notebooks/claening_data_Bibian.ipynb` — Model 2 pipeline: EDA, data cleaning, ANOVA feature selection on experience-side features (`experience_level`, `employment_type`, `job_title`, `employee_residence`), base model comparison (KNN / Linear / Random Forest / Bagging-RF), and Optuna Bayesian tuning on Random Forest (10-fold CV, 30 trials).

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
│   ├── model_building_diana.ipynb
│   └── claening_data_Bibian.ipynb    ← Model 2 (Bibian) — full pipeline
└── slides/                           ← final presentation
```

## Key Findings & Insights

- 🌍 **Geography matters** — grouping countries into regions trades some country-level granularity for model speed and a more tractable category space.
- 📉 **Feature ceiling is real** — both models converge to their respective ceilings (R² ≈ 0.24 for Model 1, ≈ 0.28 for Model 2) across structurally different algorithms. That's diagnostic of the feature set, not a model-choice problem — a better feature set matters more than better tuning.
- 🔧 **Tuning ≠ cure** — Bayesian optimisation (Optuna) on both models gained at most ~0.005 R² over baseline. At a feature ceiling, tuning delivers diminishing returns.
- 🧪 **R² reporting discipline** — single test-split and CV results are reported separately throughout; mixing them would inflate apparent performance by 2–3 percentage points.

## Challenges & Learnings

- ⚠️ **KNN classifier → regressor** — one-hot dummies plus mixed scales across categories added complexity that didn't pay off; KNN consistently underperforms the other model families on both feature sets.
- 🔄 **Feature engineering before split** — under time pressure, the company-side vs. experience-side split was made intuitively rather than through formal feature selection across the full column set. A more rigorous approach would allocate more time to feature selection per model before committing to the split.
- 🔬 **Convergence = diagnosis** — four structurally different algorithms landing on the same R² for a given feature set isn't a failure; it's a signal about how much salary variance that feature set can explain.

## Real-World Application & Impact

- **Salary benchmarking** — HR teams can use a tuned model to flag underpaid roles or benchmark new hires against market ranges, especially in remote-first AI orgs where geography no longer fully anchors pay.
- **Career navigation** — job-seekers can estimate the USD salary premium of a title, region, experience level, or company-size choice before accepting an offer.
- **Limitation — no time-forward prediction** — the dataset only spans `work_year` 2020–2024, so current and future salary projections are not supported by the ML techniques used here; this would require a different modelling approach (e.g. time-series methods) and is out of scope for this project.


## Contributors
- Diana Yule — Model 1 (company-side features): EDA, feature engineering, base model comparison, Optuna tuning, presentation
- Bibian — Model 2 (experience-side features): EDA, data cleaning, feature selection, base model comparison, Optuna tuning

## Links
- Presentation: https://docs.google.com/presentation/d/1gQCRVV1ZCu3K6sQIn_v6LvVm_OvJhK4pV4hXGplZq1o/edit?slide=id.p9#slide=id.p9
- Kanban board: https://trello.com/b/IVZJzCng/salary-project-w7

