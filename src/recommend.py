from sklearn.metrics.pairwise import cosine_similarity


def recommend_recipes(user_vector, recipe_vectors, pantry, df, top_k=5):
    similarities = cosine_similarity(user_vector, recipe_vectors).flatten()

    candidates = []

    for idx, score in enumerate(similarities):
        ingredients = df.iloc[idx]["ingredients_clean"].split()

        if not set(ingredients).intersection(set(pantry)):
            continue

        candidates.append((idx, score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    return candidates[:top_k]

# def recommend_recipes(
#     user_vector,
#     recipe_vectors,
#     top_k=5
# ):
#     similarities = cosine_similarity(user_vector, recipe_vectors).flatten()
#     top_indices = similarities.argsort()[::-1][:top_k]
#     return top_indices, similarities[top_indices]
