def keyword_overlap_score(pantry, recipe_ingredients):
    pantry_set = set(pantry)
    recipe_set = set(recipe_ingredients)
    return len(pantry_set & recipe_set)


def baseline_rank(pantry, recipes_df, top_k=5):
    scores = []

    for idx, row in recipes_df.iterrows():
        ingredients = row["ingredients_clean"].lower().split()
        score = keyword_overlap_score(pantry, ingredients)
        scores.append((idx, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[:top_k]
