"""
Text normalisation used by catalog search.

Both helpers are called thousands of times per search, so results are memoised
with an LRU cache. The catalog has a few thousand distinct titles, which fits
comfortably inside the cache and makes repeat lookups free.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Dropped from search tokens: they appear in hundreds of titles and carry no
# discriminating power, so indexing them would only widen the candidate set.
STOP_WORDS = frozenset({"a", "an", "the", "of", "and", "or", "in", "on", "to"})


@lru_cache(maxsize=8192)
def normalise(value: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    'Amélie (Le Fabuleux Destin...)' and 'amelie le fabuleux destin' both
    reduce to the same comparable string.
    """
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value))
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub(" ", ascii_only.lower()).strip()


@lru_cache(maxsize=8192)
def tokenise(value: str) -> frozenset[str]:
    """Split into meaningful search tokens, dropping stop words.

    Single characters are kept only when they are digits ('7' in 'Se7en' or
    'Ocean's 11'), since a lone letter is almost always noise.
    """
    words = normalise(value).split()
    return frozenset(
        word for word in words
        if word not in STOP_WORDS and (len(word) > 1 or word.isdigit())
    )
