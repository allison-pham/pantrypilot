from src.load_data import load_recipes
from src.preprocess import normalize_ingredients
from src.vectorize import build_vectorizer, fit_transform
from src.recommend import recommend_recipes
from src.explain import explain_recommendation
from src.ranker import baseline_rank
from src.evaluate import avg_pantry_utilization


def main():
    df = load_recipes("data/RAW_recipes.csv")
    df["ingredients_clean"] = df["ingredients"].apply(normalize_ingredients)

    raw_pantry = ["chicken", "garlic", "onion"]
    pantry = normalize_ingredients(" ".join(raw_pantry)).split()

    corpus = df["ingredients_clean"].tolist() + [" ".join(pantry)]

    vectorizer = build_vectorizer()
    vectors = fit_transform(vectorizer, corpus)

    recipe_vectors = vectors[:-1]
    user_vector = vectors[-1]

    ml_results = recommend_recipes(
        user_vector,
        recipe_vectors,
        pantry,
        df,
    )

    # indices, scores = recommend_recipes(user_vector, recipe_vectors)
    # ml_results = list(zip(indices, scores))

    baseline_results = baseline_rank(pantry, df)

    print("Baseline vs. ML Comparison")
    print(
        "Baseline avg pantry utilization:",
        avg_pantry_utilization(baseline_results, pantry, df),
    )
    print(
        "ML avg pantry utilization:",
        avg_pantry_utilization(ml_results, pantry, df),
    )
    print()

    # ML results
    print("Top ML Recommendations")
    for idx, score in ml_results:
        explanation = explain_recommendation(
            pantry,
            df.iloc[idx]["ingredients_clean"].split(),
        )

        print(f"Recipe: {df.iloc[idx]['name']}")
        print(f"Similarity score: {round(float(score), 4)}")
        print("Explanation:", explanation)
        print("-" * 40)


if __name__ == "__main__":
    main()
