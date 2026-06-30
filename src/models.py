class Page:
    def __init__(self, id, title, text, category, url, published_at):
        self.id = id
        self.title = title
        self.text = text
        self.category = category
        self.url = url
        self.published_at = published_at

    @property
    def full_text(self):
        return f"{self.title}. {self.text}"


class HistoryEvent:
    def __init__(self, page_id, title, text, dwell_seconds, timestamp):
        self.page_id = page_id
        self.title = title
        self.text = text
        self.dwell_seconds = dwell_seconds
        self.timestamp = timestamp

    @property
    def full_text(self):
        return f"{self.title}. {self.text}"


class SearchResult:
    def __init__(self, page, score, query_score, profile_score, category_score):
        self.page = page
        self.score = score
        self.query_score = query_score
        self.profile_score = profile_score
        self.category_score = category_score
