import re
from sklearn.feature_extraction import text as sk_text

# Keep negation words; they were also kept during training.
NEGATIONS = {
    "no", "not", "nor", "none", "never", "n't", "cannot", "can't", "won't",
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    "hasn't", "haven't", "hadn't", "shouldn't", "wouldn't", "couldn't",
    "but", "however", "although",
}
STOPWORDS = sk_text.ENGLISH_STOP_WORDS - NEGATIONS


def clean_text(text: str) -> str:
    """Match train.py cleaning so inference uses the same features."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-z\s'!?]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)
