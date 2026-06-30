import json
from datetime import datetime

from .models import HistoryEvent, Page


def _read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}") from error
    return items


def load_pages(path):
    pages = []
    for item in _read_jsonl(path):
        pages.append(
            Page(
                id=item["id"],
                title=item["title"],
                text=item["text"],
                category=item["category"],
                url=item["url"],
                published_at=datetime.fromisoformat(item["published_at"]),
            )
        )
    if not pages:
        raise ValueError("Pages dataset is empty")
    return pages


def load_history(path):
    events = []
    for item in _read_jsonl(path):
        events.append(
            HistoryEvent(
                page_id=item["page_id"],
                title=item["title"],
                text=item["text"],
                dwell_seconds=int(item["dwell_seconds"]),
                timestamp=datetime.fromisoformat(item["timestamp"]),
            )
        )
    return events


def load_eval_cases(path):
    return _read_jsonl(path)
