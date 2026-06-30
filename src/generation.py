import os
import re


class ExtractiveGenerator:
    """Offline fallback that never uses information outside retrieved pages."""

    def generate(self, query, results):
        if not results or results[0].score <= 0.0:
            return "В локальной базе недостаточно информации для уверенного ответа."

        lines = [f"По запросу «{query}» наиболее полезны следующие материалы:"]
        for index, result in enumerate(results[:3], start=1):
            sentence = re.split(r"(?<=[.!?])\s+", result.page.text.strip())[0]
            lines.append(f"[{index}] **{result.page.title}** — {sentence}")
        lines.append("Ответ сформирован только по найденным фрагментам.")
        return "\n\n".join(lines)


class YandexGenerator:
    """Optional generation through the OpenAI-compatible Yandex AI Studio API."""

    def __init__(self):
        from openai import OpenAI

        api_key = os.environ["YANDEX_API_KEY"]
        folder_id = os.environ["YANDEX_FOLDER_ID"]
        model_name = os.getenv("YANDEX_MODEL", "yandexgpt-lite")

        self.folder_id = folder_id
        self.model_uri = f"gpt://{folder_id}/{model_name}"
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://ai.api.cloud.yandex.net/v1",
            project=folder_id,
        )

    def generate(self, query, results):
        contexts = []
        for index, result in enumerate(results, start=1):
            contexts.append(
                f"[{index}] Заголовок: {result.page.title}\n"
                f"URL: {result.page.url}\n"
                f"Текст: {result.page.text}"
            )

        prompt = (
            "Ты — ассистент в браузере. Ответь на вопрос только по контексту ниже. "
            "Не придумывай факты. После каждого содержательного утверждения ставь "
            "ссылку на источник в формате [1]. Если контекста недостаточно, прямо скажи об этом.\n\n"
            f"Вопрос: {query}\n\nКонтекст:\n" + "\n\n".join(contexts)
        )

        response = self.client.responses.create(
            model=self.model_uri,
            input=prompt,
            temperature=0.2,
            max_output_tokens=700,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)
        return str(response.output[0].content[0].text)


def create_generator():
    if os.getenv("YANDEX_API_KEY") and os.getenv("YANDEX_FOLDER_ID"):
        try:
            return YandexGenerator()
        except Exception:
            return ExtractiveGenerator()
    return ExtractiveGenerator()
