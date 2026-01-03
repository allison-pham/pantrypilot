def explain_recommendation(
    pantry_items,
    recipe_ingredients
):

    overlap = set(pantry_items) & set(recipe_ingredients)
    utilization = len(overlap) / max(len(pantry_items), 1)

    return {
        "overlap": list(overlap),
        "pantry_utilization": utilization
    }
