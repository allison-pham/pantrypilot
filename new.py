import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_K = 5


# -----------------------------
# Data Loading
# -----------------------------
def load_recipes(path="recipes.csv"):
    """
    Expected columns:
    - title
    - ingredients (comma-separated string)
    """
    df = pd.read_csv(path)
    df["ingredients_clean"] = df["ingredients"].str.lower()
    return df


# -----------------------------
# Utility Functions
# -----------------------------
def pantry_utilization(recipe_ingredients, pantry):
    if not pantry:
        return 0.0
    overlap = set(recipe_ingredients).intersection(set(pantry))
    return len(overlap) / len(pantry)


def explain_recommendation(recipe_ingredients, pantry):
    overlap = list(set(recipe_ingredients).intersection(set(pantry)))
    return {
        "overlap": overlap,
        "pantry_utilization": pantry_utilization(recipe_ingredients, pantry)
    }


# -----------------------------
# ML Components
# -----------------------------
def build_vectorizer(corpus):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )
    vectors = vectorizer.fit_transform(corpus)
    return vectorizer, vectors


# -----------------------------
# Recommendation Pipeline
# -----------------------------
def recommend_recipes(df, pantry, vectorizer, recipe_vectors):
    pantry_text = " ".join(pantry)
    pantry_vector = vectorizer.transform([pantry_text])

    similarities = cosine_similarity(pantry_vector, recipe_vectors).flatten()

    candidates = []

    for idx, row in df.iterrows():
        ingredients = [i.strip() for i in row["ingredients_clean"].split(",")]
        utilization = pantry_utilization(ingredients, pantry)

        # 🔑 HARD FILTER: must use at least one pantry item
        if utilization == 0:
            continue

        candidates.append({
            "title": row["title"],
            "similarity": similarities[idx],
            "explanation": explain_recommendation(ingredients, pantry)
        })

    # Rank ONLY eligible recipes by semantic similarity
    ranked = sorted(candidates, key=lambda x: x["similarity"], reverse=True)

    return ranked[:TOP_K]


# -----------------------------
# Baseline (for comparison)
# -----------------------------
def baseline_utilization(df, pantry):
    utilizations = []
    for _, row in df.iterrows():
        ingredients = [i.strip() for i in row["ingredients_clean"].split(",")]
        utilizations.append(pantry_utilization(ingredients, pantry))
    return np.mean(utilizations)


# -----------------------------
# Main
# -----------------------------
def main():
    pantry = ["chicken", "rice", "onion"]

    df = load_recipes()
    vectorizer, recipe_vectors = build_vectorizer(df["ingredients_clean"])

    baseline_avg = baseline_utilization(df, pantry)
    recommendations = recommend_recipes(df, pantry, vectorizer, recipe_vectors)

    ml_avg = np.mean([
        r["explanation"]["pantry_utilization"] for r in recommendations
    ])

    print("Baseline vs. ML Comparison")
    print(f"Baseline avg pantry utilization: {baseline_avg:.2f}")
    print(f"ML avg pantry utilization: {ml_avg:.2f}\n")

    print("Top ML Recommendations\n")
    for r in recommendations:
        print(f"Recipe: {r['title']}")
        print(f"Similarity score: {r['similarity']:.4f}")
        print(f"Explanation: {r['explanation']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
