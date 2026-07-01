import argparse

from src.pipeline import NeuroBrowserPipeline


def main():
    parser = argparse.ArgumentParser(description="Personalized browser RAG demo")
    parser.add_argument("query", nargs="?", default="Что мне почитать про обучение моделей?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-personalization", action="store_true")
    args = parser.parse_args()

    pipeline = NeuroBrowserPipeline()
    response = pipeline.run(
        query=args.query,
        top_k=args.top_k,
        personalized=not args.no_personalization,
    )

    print(f"Intent: {response.intent.label} ({response.intent.confidence:.2f})")
    print(response.answer)

    if response.results:
        print("\nRanking:")
        for index, result in enumerate(response.results, start=1):
            print(
                f"{index}. {result.page.id}: {result.page.title} | "
                f"score={result.score:.3f}, query={result.query_score:.3f}, "
                f"profile={result.profile_score:.3f}, category={result.category_score:.3f}"
            )


if __name__ == "__main__":
    main()
