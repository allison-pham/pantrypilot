def avg_pantry_utilization(results, pantry, df):
    utilizations = []

    for idx, _ in results:
        ingredients = df.iloc[idx]["ingredients_clean"].split()
        overlap = set(pantry) & set(ingredients)
        utilizations.append(len(overlap) / max(len(pantry), 1))

    return sum(utilizations) / len(utilizations)
