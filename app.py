import streamlit as st

from src.pipeline import NeuroBrowserPipeline


@st.cache_resource
def load_pipeline():
    return NeuroBrowserPipeline()


pipeline = load_pipeline()

st.set_page_config(page_title="NeuroBrowser Interest RAG", layout="wide")
st.title("NeuroBrowser Interest RAG")
st.caption("Персональный поиск и ответы по истории интересов пользователя")

with st.sidebar:
    st.subheader("Профиль интересов")
    for category, score in pipeline.profile.top_categories(limit=6):
        st.write(f"**{category}** — {score:.1%}")
        st.progress(min(max(score, 0.0), 1.0))
    personalized = st.toggle("Использовать персонализацию", value=True)
    top_k = st.slider("Количество документов", 1, 8, 5)

query = st.text_input(
    "Запрос",
    value="Что мне почитать про обучение и оценку ML-моделей?",
)

if st.button("Запустить пайплайн", type="primary"):
    response = pipeline.run(query, top_k=top_k, personalized=personalized)
    st.write(
        f"Маршрут: **{response.intent.label}**, уверенность: "
        f"**{response.intent.confidence:.2f}**"
    )
    st.markdown(response.answer)

    if response.results:
        st.subheader("Найденные документы")
        for index, result in enumerate(response.results, start=1):
            with st.expander(
                f"{index}. {result.page.title} — score {result.score:.3f}"
            ):
                st.write(result.page.text)
                st.write(
                    {
                        "query_score": round(result.query_score, 4),
                        "profile_score": round(result.profile_score, 4),
                        "category_score": round(result.category_score, 4),
                        "category": result.page.category,
                        "url": result.page.url,
                    }
                )
