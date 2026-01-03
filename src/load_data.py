import pandas as pd


def load_recipes():
    recipes_df = pd.read_csv('RAW_recipes.csv', on_bad_lines='skip')
