# [Project Title] — ML Week 7

## Project Description
<!-- TBD: What are you predicting? Why does it matter? -->

## Data Sources
<!-- TBD: Where did the data come from? Link here. -->

## Techniques Used
- Data cleaning & feature engineering
- Supervised learning (classification / regression)
- Hyperparameter tuning: Grid Search + Cross-Validation
- Ensemble models (e.g. Random Forest, Gradient Boosting)

## Key Findings
<!-- TBD: Fill in after analysis. -->

## Next Steps
<!-- TBD: What would you do with more time? -->

## How to Run

```bash
# 1. Navigate to project root
cd "DataAnalytics Ironhack/Week7/Project_ML_W7"

# 2. Activate virtual environment
source .venv/bin/activate  # or: uv sync

# 3. Register Jupyter kernel (first time only)
python -m ipykernel install --user --name=venv

# 4. Launch Jupyter
jupyter lab
```

Run notebooks in this order:
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
