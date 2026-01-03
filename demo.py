from src.load_data import load_recipes
from src.preprocess import normalize_ingredients
from src.vectorize import build_vectorizer, fit_transform
from src.recommend import recommend_recipes
from src.explain import explain_recommendation


def main():
    df = load_recipes("data/RAW_recipes.csv")
    df["ingredients_clean"] = df["ingredients"].apply(normalize_ingredients)

    pantry = ["chicken", "garlic", "onion"]
    corpus = df["ingredients_clean"].tolist() + [" ".join(pantry)]

    vectorizer = build_vectorizer()
    vectors = fit_transform(vectorizer, corpus)

    recipe_vectors = vectors[:-1]
    user_vector = vectors[-1]

    indices, scores = recommend_recipes(user_vector, recipe_vectors)

    for idx, score in zip(indices, scores):
        explanation = explain_recommendation(
            pantry,
            df.iloc[idx]["ingredients"].split()
        )

        print(df.iloc[idx]["name"])
        print(explanation)
        print()


if __name__ == "__main__":
    main()
