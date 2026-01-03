import pandas as pd


def load_recipes(path="data/RAW_recipes.csv"):
    return pd.read_csv(path, on_bad_lines="skip")
