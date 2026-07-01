from pprint import pprint

from src.data import load_eval_cases
from src.feedback import build_preference_pairs, build_sft_examples


def main():
    events = load_eval_cases("data/feedback.jsonl")
    sft_examples = build_sft_examples(events)
    preference_pairs = build_preference_pairs(events)

    print(f"SFT examples: {len(sft_examples)}")
    print(f"Preference pairs: {len(preference_pairs)}")
    print()

    if sft_examples:
        print("First SFT example:")
        pprint(sft_examples[0], width=100)
        print()

    if preference_pairs:
        print("First preference pair:")
        pprint(preference_pairs[0], width=100)


if __name__ == "__main__":
    main()
