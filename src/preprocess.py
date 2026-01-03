def normalize_ingredients(ingredients: str) -> str:
    ingredients = ingredients.lower()
    ingredients = ingredients.replace(",", " ")
    return ingredients
