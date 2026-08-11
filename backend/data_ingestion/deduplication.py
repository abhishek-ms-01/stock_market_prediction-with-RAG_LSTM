import re

def clean_text_snippet(text: str) -> str:
    """Sanitizes text for exact and fuzzy deduplication."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return ' '.join(text.split())

def deduplicate_articles(articles: list, sim_threshold: float = 0.85) -> list:
    """
    Deduplicates financial news articles based on title similarity and exact match.
    """
    unique_articles = []
    seen_titles = set()

    for item in articles:
        title = item.get("title", "")
        clean_t = clean_text_snippet(title)
        if not clean_t or clean_t in seen_titles:
            continue

        # Simple Jaccard similarity check against existing titles
        tokens = set(clean_t.split())
        is_duplicate = False
        for seen in seen_titles:
            s_tokens = set(seen.split())
            union_len = len(tokens.union(s_tokens))
            if union_len > 0:
                jaccard = len(tokens.intersection(s_tokens)) / union_len
                if jaccard >= sim_threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            seen_titles.add(clean_t)
            unique_articles.append(item)

    return unique_articles
