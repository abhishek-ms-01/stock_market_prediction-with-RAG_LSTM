import re
import ssl

# Bypass SSL certificate verification for NLTK downloads (common macOS issue)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

_nltk_tokenize = None

try:
    import nltk
    for dataset in ['punkt', 'punkt_tab']:
        try:
            nltk.download(dataset, quiet=True)
        except Exception:
            pass
    from nltk.tokenize import word_tokenize as _nltk_tokenize
except Exception:
    _nltk_tokenize = None

def tokenize_text(text):
    if not isinstance(text, str):
        return []
    
    if _nltk_tokenize is not None:
        try:
            return _nltk_tokenize(text)
        except Exception:
            pass

    # Pure regex fallback tokenizer if NLTK or punkt resource is missing
    return re.findall(r'\w+|[^\w\s]', text)