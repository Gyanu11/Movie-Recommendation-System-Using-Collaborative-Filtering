"""Trailer links.

Nothing here is required for the app to work: if YouTube is unreachable the
functions degrade to a plain search URL rather than raising, so a missing
network connection never breaks a page.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_VIDEO_ID_RE = re.compile(r'"videoId":"([a-zA-Z0-9_-]{11})"')
_DEFAULT_TIMEOUT = 6


def _search_query(title: str, year) -> str:
    return " ".join(part for part in (str(title).strip(), str(year or "").strip(), "official trailer") if part)


def trailer_search_url(title: str, year="") -> str:
    """A YouTube search-results link. No network call, so it always works."""
    return f"https://www.youtube.com/results?search_query={quote_plus(_search_query(title, year))}"


def resolve_youtube_trailer(title: str, year="", timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Best-effort embeddable URL for a movie's trailer.

    Scrapes the first video id off YouTube's search page. On any failure -
    no network, a timeout, a markup change - it returns an embeddable search
    playlist instead, which still plays something relevant.
    """
    query = _search_query(title, year)
    encoded = quote_plus(query)

    try:
        request = Request(
            f"https://www.youtube.com/results?search_query={encoded}",
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        )
        with urlopen(request, timeout=timeout) as response:
            markup = response.read().decode("utf-8", errors="ignore")
        match = _VIDEO_ID_RE.search(markup)
        if match:
            return f"https://www.youtube.com/embed/{match.group(1)}?autoplay=1&rel=0&modestbranding=1"
    except Exception:  # noqa: BLE001 - a broken trailer must never break the page
        log.debug("Trailer lookup failed for %r", query, exc_info=True)

    return f"https://www.youtube.com/embed?listType=search&list={encoded}&autoplay=1"
