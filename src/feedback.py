from collections import defaultdict


def build_sft_examples(events):
    """Keep accepted or highly rated answers as supervised examples."""
    examples = []
    for event in events:
        rating = int(event.get("rating", 0))
        accepted = bool(event.get("accepted", False))
        if accepted or rating >= 4:
            examples.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Отвечай только по переданному контексту и указывай источники.",
                        },
                        {
                            "role": "user",
                            "content": event["prompt"],
                        },
                        {
                            "role": "assistant",
                            "content": event["answer"],
                        },
                    ],
                    "metadata": {
                        "query_id": event["query_id"],
                        "rating": rating,
                    },
                }
            )
    return examples


def build_preference_pairs(events):
    """Create chosen/rejected pairs for DPO-like preference optimization."""
    grouped = defaultdict(list)
    for event in events:
        grouped[str(event["query_id"])].append(event)

    pairs = []
    for query_id, variants in grouped.items():
        if len(variants) < 2:
            continue
        ordered = sorted(
            variants,
            key=lambda item: (int(item.get("rating", 0)), bool(item.get("accepted", False))),
            reverse=True,
        )
        chosen = ordered[0]
        rejected = ordered[-1]
        if int(chosen.get("rating", 0)) <= int(rejected.get("rating", 0)):
            continue
        pairs.append(
            {
                "prompt": chosen["prompt"],
                "chosen": chosen["answer"],
                "rejected": rejected["answer"],
                "metadata": {
                    "query_id": query_id,
                    "chosen_rating": int(chosen.get("rating", 0)),
                    "rejected_rating": int(rejected.get("rating", 0)),
                },
            }
        )
    return pairs
