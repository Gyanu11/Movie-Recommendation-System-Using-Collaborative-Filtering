"""
Poster artwork.

MovieLens ships ids and ratings, not images. Roughly 13% of the catalog has a
real poster URL carried over in `movies.csv`; `scripts/fetch_posters.py` can
fill in the rest from TMDB if you supply an API key.

Everything else gets a generated placeholder: a deterministic gradient card
built from the movie's own id, with its title and year set in the site's
typeface. It renders instantly, needs no network, and looks like part of the
design rather than a broken image icon.
"""
from __future__ import annotations

import colorsys
from html import escape

# Two poster-shaped gradients per movie, derived from a hash of its id so the
# same film always gets the same artwork across page loads and machines.
_POSTER_WIDTH = 400
_POSTER_HEIGHT = 600


def enlarge_poster(url: str | None) -> str:
    """Swap an IMDb thumbnail URL for its 500px-wide rendition."""
    if not isinstance(url, str) or "._V1_" not in url:
        return url or ""
    return url.split("._V1_")[0] + "._V1_SX500.jpg"


def _palette(seed: str) -> tuple[str, str]:
    """Two harmonious hex colours derived deterministically from `seed`."""
    digest = 0
    for character in seed:
        digest = (digest * 31 + ord(character)) & 0xFFFFFFFF

    hue = (digest % 360) / 360.0
    complement = (hue + 0.12) % 1.0

    def to_hex(h: float, s: float, v: float) -> str:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    return to_hex(hue, 0.55, 0.32), to_hex(complement, 0.72, 0.13)


def _wrap(title: str, width: int = 16, max_lines: int = 4) -> list[str]:
    """Greedy word wrap, because SVG text does not reflow on its own."""
    lines: list[str] = []
    current = ""
    for word in title.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if not lines:
        lines = [title[:width] or "Untitled"]
    if len(lines) == max_lines and len(" ".join(lines)) < len(title):
        lines[-1] = lines[-1][: width - 1] + "\u2026"
    return lines


def placeholder_svg(movie_id: str, title: str, year: str = "", genres: str = "") -> str:
    """Render a self-contained poster placeholder as an SVG document."""
    start, end = _palette(str(movie_id))
    lines = _wrap(title)

    # Vertically centre the title block, then lay lines out from there.
    line_height = 44
    block_top = (_POSTER_HEIGHT / 2) - ((len(lines) - 1) * line_height / 2) - 10
    text_rows = "".join(
        f'<text x="50%" y="{block_top + index * line_height:.0f}" '
        f'text-anchor="middle" fill="#ffffff" font-size="38" font-weight="700" '
        f'font-family="Outfit, Segoe UI, sans-serif">{escape(line)}</text>'
        for index, line in enumerate(lines)
    )

    caption_parts = [part for part in (str(year).strip(), genres.split("|")[0].strip()) if part]
    caption = escape(" \u00b7 ".join(caption_parts))

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_POSTER_WIDTH} {_POSTER_HEIGHT}" \
width="{_POSTER_WIDTH}" height="{_POSTER_HEIGHT}" role="img" aria-label="{escape(title)} poster">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{start}"/>
      <stop offset="100%" stop-color="{end}"/>
    </linearGradient>
  </defs>
  <rect width="{_POSTER_WIDTH}" height="{_POSTER_HEIGHT}" fill="url(#g)"/>
  <circle cx="{_POSTER_WIDTH * 0.82:.0f}" cy="{_POSTER_HEIGHT * 0.16:.0f}" r="130" fill="#ffffff" opacity="0.05"/>
  <circle cx="{_POSTER_WIDTH * 0.12:.0f}" cy="{_POSTER_HEIGHT * 0.88:.0f}" r="160" fill="#000000" opacity="0.12"/>
  <rect x="34" y="34" width="{_POSTER_WIDTH - 68}" height="{_POSTER_HEIGHT - 68}" fill="none"
        stroke="#ffffff" stroke-opacity="0.16" stroke-width="2" rx="18"/>
  {text_rows}
  <text x="50%" y="{_POSTER_HEIGHT - 66}" text-anchor="middle" fill="#fbbf24" font-size="21"
        font-weight="600" letter-spacing="2" font-family="Outfit, Segoe UI, sans-serif">{caption}</text>
</svg>"""
