import math

import numpy as np

from src.data import load_eval_cases
from src.pipeline import NeuroBrowserPipeline


def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    found = len(set(ranked_ids[:k]) & relevant_ids)
    return found / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids, k):
    for rank, page_id in enumerate(ranked_ids[:k], start=1):
        if page_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k):
    dcg = 0.0
    for rank, page_id in enumerate(ranked_ids[:k], start=1):
        if page_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg:
        return dcg / idcg
    return 0.0


def evaluate_mode(pipeline, cases, personalized, k=3):
    recalls = []
    mrrs = []
    ndcgs = []
    citation_validity = []

    for case in cases:
        response = pipeline.run(
            case["query"],
            top_k=k,
            personalized=personalized,
        )
        ranked_ids = [result.page.id for result in response.results]
        relevant_ids = set(case["relevant_ids"])

        recalls.append(recall_at_k(ranked_ids, relevant_ids, k))
        mrrs.append(reciprocal_rank(ranked_ids, relevant_ids, k))
        ndcgs.append(ndcg_at_k(ranked_ids, relevant_ids, k))

        allowed = {f"[{index}]" for index in range(1, len(response.results) + 1)}
        used = {token for token in allowed if token in response.answer}
        citation_validity.append(float(bool(used) or not response.results))

    return {
        f"Recall@{k}": float(np.mean(recalls)),
        f"MRR@{k}": float(np.mean(mrrs)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "CitationCoverage": float(np.mean(citation_validity)),
    }


def main():
    pipeline = NeuroBrowserPipeline()
    cases = load_eval_cases("data/eval.jsonl")

    baseline = evaluate_mode(pipeline, cases, personalized=False)
    personalized = evaluate_mode(pipeline, cases, personalized=True)

    print("Metric                 Baseline   Personalized   Lift")
    print("-" * 57)
    for metric in baseline:
        base = baseline[metric]
        personal = personalized[metric]
        print(f"{metric:<22} {base:>8.3f}   {personal:>12.3f}   {personal-base:>+.3f}")


if __name__ == "__main__":
    main()
