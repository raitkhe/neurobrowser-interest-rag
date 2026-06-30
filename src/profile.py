import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

class UserProfile:
    def __init__(self, vector, category_scores, event_weights):
        self.vector = vector
        self.category_scores = category_scores
        self.event_weights = event_weights

    def top_categories(self, limit=5):
        return sorted(
            self.category_scores.items(), key=lambda item: item[1], reverse=True
        )[:limit]


class InterestModel:
    """Builds a long-term profile from page semantics, dwell time and recency."""

    def __init__(self, pages, half_life_days=14.0):
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.pages = pages
        self.half_life_days = half_life_days
        self.categories = sorted({page.category for page in pages})

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=True,
        )
        page_texts = [page.full_text for page in pages]
        self.page_matrix = normalize(self.vectorizer.fit_transform(page_texts))

        self.category_model = LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        )
        self.category_model.fit(self.page_matrix, [page.category for page in pages])

    def build(self, history):
        if not history:
            empty = csr_matrix((1, self.page_matrix.shape[1]), dtype=float)
            uniform = 1.0 / len(self.categories)
            return UserProfile(
                vector=empty,
                category_scores={category: uniform for category in self.categories},
                event_weights=[],
            )

        latest_time = max(event.timestamp for event in history)
        history_matrix = self.vectorizer.transform(
            [event.full_text for event in history]
        )

        weights = []
        for event in history:
            age_days = max((latest_time - event.timestamp).total_seconds() / 86400, 0.0)
            recency = 0.5 ** (age_days / self.half_life_days)
            engagement = np.log1p(max(event.dwell_seconds, 0))
            weights.append(float(recency * engagement))

        weight_array = np.asarray(weights, dtype=float)
        if float(weight_array.sum()) == 0.0:
            weight_array = np.ones_like(weight_array)

        weighted_vector = history_matrix.multiply(weight_array[:, None]).sum(axis=0)
        profile_vector = normalize(csr_matrix(weighted_vector))

        probabilities = self.category_model.predict_proba(history_matrix)
        weighted_probabilities = np.average(probabilities, axis=0, weights=weight_array)
        category_scores = {
            str(category): float(score)
            for category, score in zip(
                self.category_model.classes_, weighted_probabilities
            )
        }

        return UserProfile(
            vector=profile_vector,
            category_scores=category_scores,
            event_weights=weights,
        )
