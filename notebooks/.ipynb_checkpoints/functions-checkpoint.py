"""
functions.py — Shared helper functions for ML Project Week 7
All reusable logic lives here. Import into notebooks with:
    from functions import *
"""

import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns



# ── Config loader ──────────────────────────────────────────────────────────────

def load_config(config_path="../config.yaml"):
    """Load project config from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_raw_data(path, **kwargs):
    """
    Load raw CSV. Uses sep=None + python engine to handle
    Windows CRLF/BOM exports safely.
    """
    return pd.read_csv(path, sep=None, engine="python", **kwargs)


# ── EDA helpers ────────────────────────────────────────────────────────────────

def summarise_dataframe(df):
    """Print shape, dtypes, null counts, and basic stats."""
    print(f"Shape: {df.shape}")
    print("\n--- Null counts ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\n--- Dtypes ---")
    print(df.dtypes)
    print("\n--- Describe ---")
    return df.describe()




