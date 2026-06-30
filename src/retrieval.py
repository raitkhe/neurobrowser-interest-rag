import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .models import SearchResult


class PersonalizedRetriever:
    """Hybrid retrieval: query relevance + user profile + category affinity."""

    def __init__(
        self,
        pages,
        interest_model,
        query_weight=0.68,
        profile_weight=0.22,
        category_weight=0.10,
    ):
        total = query_weight + profile_weight + category_weight
        if not np.isclose(total, 1.0):
            raise ValueError("Retriever weights must sum to 1")
        self.pages = pages
        self.interest_model = interest_model
        self.query_weight = query_weight
        self.profile_weight = profile_weight
        self.category_weight = category_weight

    def search(
        self,
        query,
        profile,
        top_k=5,
        personalized=True,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self.interest_model.vectorizer.transform([query])
        query_scores = cosine_similarity(
            query_vector, self.interest_model.page_matrix
        )[0]

        if personalized and profile.vector.nnz > 0:
            profile_scores = cosine_similarity(
                profile.vector, self.interest_model.page_matrix
            )[0]
            category_scores = np.asarray(
                [profile.category_scores.get(page.category, 0.0) for page in self.pages]
            )

            # Explicit queries should be driven mostly by the query itself.
            # Personalization becomes stronger only when the wording is vague.
            max_query_score = float(query_scores.max(initial=0.0))
            specificity = min(max_query_score / 0.30, 1.0)
            dynamic_query_weight = 0.55 + 0.35 * specificity
            remaining_weight = 1.0 - dynamic_query_weight
            dynamic_profile_weight = remaining_weight * 0.70
            dynamic_category_weight = remaining_weight * 0.30

            final_scores = (
                dynamic_query_weight * query_scores
                + dynamic_profile_weight * profile_scores
                + dynamic_category_weight * category_scores
            )
        else:
            profile_scores = np.zeros(len(self.pages), dtype=float)
            category_scores = np.zeros(len(self.pages), dtype=float)
            final_scores = query_scores

        order = [
            int(index)
            for index in np.argsort(-final_scores)
            if float(final_scores[index]) > 0.0
        ][:top_k]
        return [
            SearchResult(
                page=self.pages[index],
                score=float(final_scores[index]),
                query_score=float(query_scores[index]),
                profile_score=float(profile_scores[index]),
                category_score=float(category_scores[index]),
            )
            for index in order
        ]
