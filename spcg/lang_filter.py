"""
lang_filter.py
==============
English-language detection used to keep only English posts in the content
(TF-IDF cosine) path -- content similarity is only meaningful when comparing the
same language.

Uses ``langid`` (its model is bundled in the package, so this works on offline
compute nodes -- no download, unlike fastText/HF detectors). The identifier is
lazily constructed and memoized per process (so it is built once per worker in a
ProcessPool fork), and is thread/loop-cheap thereafter.

Public API:
  is_english(text, min_prob=0.5, min_chars=1) -> bool
  english_mask(texts, ...) -> np.ndarray[bool]
"""

_IDENTIFIER = None


def _identifier():
    """Lazily build and memoize the langid identifier (norm_probs -> calibrated
    0..1 probabilities so a probability threshold is meaningful)."""
    global _IDENTIFIER
    if _IDENTIFIER is None:
        from langid.langid import LanguageIdentifier, model

        _IDENTIFIER = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    return _IDENTIFIER


def is_english(text, min_prob=0.5, min_chars=1):
    """True iff ``text`` is detected as English with probability >= ``min_prob``.

    Empty/whitespace or shorter-than ``min_chars`` text is treated as NOT English
    (dropped): such fragments are unreliable to classify, and the filter is
    deliberately conservative (English-only).
    """
    if not text:
        return False
    t = text.strip()
    if len(t) < min_chars:
        return False
    lang, prob = _identifier().classify(t)
    return lang == "en" and prob >= min_prob


def english_mask(texts, min_prob=0.5, min_chars=1):
    """Boolean mask over an iterable of texts (True = keep)."""
    import numpy as np

    return np.fromiter(
        (is_english(t, min_prob=min_prob, min_chars=min_chars) for t in texts),
        dtype=bool,
        count=len(texts) if hasattr(texts, "__len__") else -1,
    )
