from sklearn.metrics.pairwise import cosine_similarity


def recommend_recipes(
    user_vector,
    recipe_vectors,
    top_k=5
):
    similarities = cosine_similarity(user_vector, recipe_vectors).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return top_indices, similarities[top_indices]
