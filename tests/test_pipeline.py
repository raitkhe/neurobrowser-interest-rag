from src.pipeline import NeuroBrowserPipeline


def build_pipeline():
    return NeuroBrowserPipeline()


def test_profile_has_ml_as_top_interest():
    pipeline = build_pipeline()
    top_category, _ = pipeline.profile.top_categories(limit=1)[0]
    assert top_category == "machine_learning"


def test_recommendation_returns_ranked_documents():
    pipeline = build_pipeline()
    response = pipeline.run("Что мне почитать про обучение моделей?", top_k=3)
    assert response.intent.label in {"recommend", "question"}
    assert len(response.results) == 3
    assert response.results[0].score >= response.results[1].score


def test_profile_intent_does_not_call_retrieval():
    pipeline = build_pipeline()
    response = pipeline.run("Покажи мой профиль интересов")
    assert response.intent.label == "profile"
    assert response.results == []
    assert "machine_learning" in response.answer


def test_local_answer_contains_valid_citation():
    pipeline = build_pipeline()
    response = pipeline.run("Как оценивать качество RAG?", top_k=3)
    assert any(f"[{index}]" in response.answer for index in range(1, 4))


def test_training_data_builders_create_preference_pair():
    from src.feedback import build_preference_pairs, build_sft_examples

    events = [
        {"query_id": "q", "prompt": "p", "answer": "good", "rating": 5, "accepted": True},
        {"query_id": "q", "prompt": "p", "answer": "bad", "rating": 1, "accepted": False},
    ]
    assert len(build_sft_examples(events)) == 1
    pairs = build_preference_pairs(events)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "good"
