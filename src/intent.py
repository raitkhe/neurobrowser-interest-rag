from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class IntentPrediction:
    def __init__(self, label, confidence):
        self.label = label
        self.confidence = confidence


class IntentRouter:
    """A tiny trainable router instead of a brittle set of if-statements."""

    def __init__(self):
        samples = {
            "recommend": [
                "что мне почитать",
                "порекомендуй статьи",
                "подбери интересный контент",
                "что посмотреть дальше",
                "дай персональные рекомендации",
                "найди материалы под мои интересы",
                "посоветуй полезную статью",
                "что нового изучить",
                "покажи похожие материалы",
                "рекомендации на сегодня",
            ],
            "question": [
                "как работает rag",
                "объясни что такое sft",
                "ответь на вопрос по теме",
                "в чем отличие reranker от retriever",
                "зачем нужен reward model",
                "как оценивать поиск",
                "расскажи про агента",
                "что написано в материалах",
                "найди ответ на вопрос",
                "почему модель ошибается",
            ],
            "profile": [
                "какие у меня интересы",
                "покажи мой профиль",
                "что ты знаешь о моих интересах",
                "из чего состоит профиль пользователя",
                "обнови мои интересы",
                "почему ты это рекомендуешь",
                "какие темы мне нравятся",
                "покажи веса интересов",
                "как сформирован мой профиль",
                "моя история интересов",
            ],
        }
        texts = []
        labels = []
        for label, examples in samples.items():
            texts.extend(examples)
            labels.extend([label] * len(examples))

        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=1,
                        lowercase=True,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                ),
            ]
        )
        self.pipeline.fit(texts, labels)

    def predict(self, text):
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        best_index = int(probabilities.argmax())
        return IntentPrediction(
            label=str(classes[best_index]),
            confidence=float(probabilities[best_index]),
        )
