# AI/ML/Data Salaries — Prediction (ML Week 7)

## Project Description

Predict `salary_in_usd` for global AI / ML / Data roles using supervised regression.
Two complementary models are built on the same dataset: **Model 1 (Diana)** focuses on
company-related features; **Model 2 (Bibian)** focuses on experience level.

## Data Sources

- **aijobs.net Global AI/ML/Data Salaries** (~34k rows, CC0 licence)
- Direct CSV download: https://aijobs.net/salaries/download/
- After cleaning: ~23,649 rows used for modelling.

## Techniques Used

- Feature engineering: region grouping, rare-category collapsing, derived same-region flag
- Feature selection: boxplots + one-way ANOVA (`f_oneway`)
- Encoding & scaling: `OneHotEncoder` (nominal columns) + `StandardScaler`
- Regression models: KNeighbors, Linear Regression, Random Forest
- Evaluation: R², MAE, RMSE on a held-out test set

## Functions


| Function | Signature | Purpose |
|---|---|---|
| `read_file` | `read_file(yaml_path, inp_data_section, file_name)` | Read the YAML config and load the CSV at the path it points to. Returns a DataFrame (or `None` if the YAML is missing). |
| `summarise_dataframe` | `summarise_dataframe(df)` | Print shape, null counts, and dtypes; return `df.describe()`. |
| `features_target_extract` | `features_target_extract(df, drop_cols, target_col)` | Drop unused columns, then split into `features` and `target`. Pass `drop_cols=[]` when nothing needs dropping. |
| `dummies_catcols` | `dummies_catcols(X_train, X_test, nonnum_cols)` | One-hot encode the nominal columns. Fits the encoder on train only, then transforms both train and test; preserves the original index. |
| `df_full_standarized` | `df_full_standarized(X_train_dumm, X_test_dumm, X_train_num, X_test_num)` | Concatenate dummified + numeric columns and apply `StandardScaler` (fit on train, transform both). Returns scaled train/test DataFrames. |

> 🤖 Known limitation to review: `features_target_extract` returns `None` in its `else` branch (no target column), which breaks tuple-unpacking on a test set. `df_full_standarized` scales every column including the one-hot dummies.

## How to Run

```bash
# 1. Navigate to project root
cd "DataAnalytics Ironhack/Week7/Project_ML_W7"

# 2. Sync the environment
uv sync

# 3. Register the Jupyter kernel (first time only)
python -m ipykernel install --user --name=venv

# 4. Launch Jupyter
jupyter lab
```

Run notebooks in order:
1. `notebooks/load_and_clean_data_diana.ipynb`
2. `notebooks/explore_clean_data_diana.ipynb`
3. `notebooks/model_building_diana.ipynb`

## Project Structure

```
.
├── README.md
├── config.yaml
├── pyproject.toml
├── .gitignore
├── data/
│   ├── raw/          ← original unaltered dataset
│   └── clean/        ← processed dataset
├── figures/          ← saved plots
├── notebooks/
│   ├── functions.py  ← shared helper functions
│   ├── load_and_clean_data_diana.ipynb
│   ├── explore_clean_data_diana.ipynb
│   └── model_building_diana.ipynb
└── slides/           ← link to online presentation
```

## Contributors
- Diana Yule
- Bibian
