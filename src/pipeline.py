from .data import load_history, load_pages
from .generation import create_generator
from .intent import IntentRouter
from .profile import InterestModel
from .retrieval import PersonalizedRetriever


class PipelineResponse:
    def __init__(self, intent, answer, results):
        self.intent = intent
        self.answer = answer
        self.results = results


class NeuroBrowserPipeline:
    def __init__(
        self,
        pages_path="data/pages.jsonl",
        history_path="data/history.jsonl",
        generator=None,
    ):
        self.pages = load_pages(pages_path)
        self.history = load_history(history_path)
        self.interest_model = InterestModel(self.pages)
        self.profile = self.interest_model.build(self.history)
        self.retriever = PersonalizedRetriever(self.pages, self.interest_model)
        self.router = IntentRouter()
        self.generator = generator or create_generator()

    def describe_profile(self):
        interests = self.profile.top_categories(limit=5)
        if not interests:
            return "Профиль пока пуст."
        lines = ["Текущий профиль интересов:"]
        for category, score in interests:
            lines.append(f"- {category}: {score:.1%}")
        lines.append(
            "Вес события зависит от времени просмотра и уменьшается по мере устаревания."
        )
        return "\n".join(lines)

    def run(
        self,
        query,
        top_k=5,
        personalized=True,
    ):
        intent = self.router.predict(query)
        if intent.label == "profile":
            return PipelineResponse(
                intent=intent,
                answer=self.describe_profile(),
                results=[],
            )

        results = self.retriever.search(
            query=query,
            profile=self.profile,
            top_k=top_k,
            personalized=personalized,
        )
        answer = self.generator.generate(query, results)
        return PipelineResponse(intent=intent, answer=answer, results=results)
