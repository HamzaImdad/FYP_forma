"""Ingest FORMA's public React pages into a knowledge base for the public
chatbot.

Reads the .tsx source files under app/web/src/pages/ for the logged-out public
routes (Home, About, HowItWorks, Features, Exercises), extracts human-readable
text (JSX text nodes, string-literal props like title/body/subtitle/eyebrow),
chunks it by heading, and writes the result to data/website_kb.json.

This is a SOURCE-BASED ingestion pipeline — we parse the JSX source instead of
running a headless browser on the rendered page, which keeps the script simple,
offline-friendly, and deterministic. It gives the widget's retrieval layer a
stable corpus of ~30-60 chunks per run.

Usage:
    "C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" scripts/ingest_website.py

Output:
    data/website_kb.json — list of {id, route, heading, text}

The KB has no vector embeddings — the public chat module does keyword/TF-IDF
retrieval at runtime. Embeddings can be added later once OPENAI_API_KEY is
set, without requiring a re-ingest.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = PROJECT_ROOT / "app" / "web" / "src" / "pages"
OUTPUT = PROJECT_ROOT / "data" / "website_kb.json"

# Map of page file → route → display label
PAGES = [
    ("Home.tsx", "/", "Home"),
    ("AboutPage.tsx", "/about", "About"),
    ("HowItWorksPage.tsx", "/how-it-works", "How It Works"),
    ("FeaturesPage.tsx", "/features", "Features"),
    ("ExercisesPage.tsx", "/exercises", "Exercises"),
]

# Token-sized chunks (target char length, not strict)
MIN_CHUNK_CHARS = 120
MAX_CHUNK_CHARS = 900


@dataclass
class KBChunk:
    id: str
    route: str
    page: str
    heading: str
    text: str


# ── Text extraction helpers ─────────────────────────────────────────────

# Props that commonly contain natural-language copy on our pages
COPY_PROP_NAMES = {
    "title",
    "italic",
    "subtitle",
    "eyebrow",
    "body",
    "label",
    "kicker",
    "tagline",
    "caption",
    "description",
    "tag",
    "heading",
    "detail",
    "placeholder",
}

# Common JSX text patterns
_UNICODE_ESCAPES = {
    "\\u2014": "—",
    "\\u2192": "→",
    "\\u2191": "↑",
    "\\u2193": "↓",
    "\\u00a0": " ",
    "\\u00A0": " ",
    "\\u2018": "'",
    "\\u2019": "'",
    "\\u201c": '"',
    "\\u201d": '"',
    "\\u003e": ">",
    "\\u003E": ">",
    "\\u003c": "<",
    "\\u003C": "<",
    "\\u2026": "…",
}


def _unescape(text: str) -> str:
    for k, v in _UNICODE_ESCAPES.items():
        text = text.replace(k, v)
    # Generic \uXXXX
    def _sub(m: re.Match) -> str:
        try:
            return chr(int(m.group(1), 16))
        except ValueError:
            return m.group(0)

    text = re.sub(r"\\u([0-9a-fA-F]{4})", _sub, text)
    # Escaped quotes and backslashes inside string literals
    text = text.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
    return text


def _clean(text: str) -> str:
    # Strip leading/trailing whitespace, collapse runs of whitespace, drop
    # punctuation leftovers from JSX curly braces.
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("{ ", "").replace(" }", "")
    # Drop common JSX noise tokens
    text = re.sub(r"\{`[^`]*`\}", "", text)
    return text


def _extract_string_prop_values(source: str) -> List[str]:
    """Find `prop="..."` or `prop={"..."}` for copy-carrying props.

    Catches both double-quote and backtick-delimited string literals. Ignores
    JSX expressions (anything starting with `{` that isn't a plain string).
    """
    results: List[str] = []

    # prop="literal" and prop='literal'
    pattern_quoted = re.compile(
        r"\b(" + "|".join(sorted(COPY_PROP_NAMES)) + r")\s*=\s*([\"'])((?:(?!\2).|\\\2)*?)\2",
        re.DOTALL,
    )
    for m in pattern_quoted.finditer(source):
        val = _unescape(m.group(3))
        val = _clean(val)
        if len(val) >= 4:
            results.append(val)

    # prop={"literal"} / prop={'literal'}
    pattern_brace_quoted = re.compile(
        r"\b(" + "|".join(sorted(COPY_PROP_NAMES)) + r")\s*=\s*\{\s*([\"'])((?:(?!\2).|\\\2)*?)\2\s*\}",
        re.DOTALL,
    )
    for m in pattern_brace_quoted.finditer(source):
        val = _unescape(m.group(3))
        val = _clean(val)
        if len(val) >= 4:
            results.append(val)

    return results


def _extract_standalone_strings(source: str, min_len: int = 30) -> List[str]:
    """Find top-level object-literal string fields like `title: "..."`, `body: "..."`.

    These show up in `const FEATURES = [{ title: "...", body: "..." }]` arrays
    which define most of our page content.
    """
    results: List[str] = []
    # field: "literal" (same props as COPY_PROP_NAMES)
    pattern = re.compile(
        r"\b(" + "|".join(sorted(COPY_PROP_NAMES)) + r")\s*:\s*([\"'])((?:(?!\2).|\\\2)*?)\2",
        re.DOTALL,
    )
    for m in pattern.finditer(source):
        val = _unescape(m.group(3))
        val = _clean(val)
        if len(val) >= min_len or m.group(1) in ("title", "tag", "heading", "kicker"):
            results.append(val)
    return results


def _extract_jsx_text(source: str) -> List[str]:
    """Pull human-readable text out of `>...<` nodes."""
    results: List[str] = []
    # Match >text< where text has at least one letter
    pattern = re.compile(r">\s*([^<>\{\}]+?)\s*<", re.DOTALL)
    for m in pattern.finditer(source):
        raw = m.group(1)
        if not re.search(r"[A-Za-z]{3}", raw):
            continue
        if raw.strip().startswith(("//", "/*", "*", "import ", "const ", "return")):
            continue
        val = _unescape(raw)
        val = _clean(val)
        if len(val) >= 20:
            results.append(val)
    return results


def _extract_headings_and_bodies(source: str) -> List[tuple[str, str]]:
    """Pair up title-like strings with the paragraph-like strings that follow.

    We collect `title: "..."` fields and nearby `body: "..."` fields as
    heading+body pairs to give the KB a semantic grouping that retrieval
    can latch onto.
    """
    pairs: List[tuple[str, str]] = []
    pattern = re.compile(
        r"\btitle\s*:\s*([\"'])((?:(?!\1).|\\\1)*?)\1"
        r"(.{0,1200}?)"
        r"\bbody\s*:\s*([\"'])((?:(?!\4).|\\\4)*?)\4",
        re.DOTALL,
    )
    for m in pattern.finditer(source):
        title = _clean(_unescape(m.group(2)))
        body = _clean(_unescape(m.group(5)))
        if title and body:
            pairs.append((title, body))
    return pairs


# ── Chunking ────────────────────────────────────────────────────────────


def _chunk_text(text: str) -> List[str]:
    """Split `text` into <= MAX_CHUNK_CHARS chunks on sentence boundaries."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text] if len(text) >= MIN_CHUNK_CHARS // 4 else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 > MAX_CHUNK_CHARS and buf:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = (buf + " " + s).strip() if buf else s
    if buf:
        chunks.append(buf.strip())
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS // 4]


# ── Main ────────────────────────────────────────────────────────────────


def ingest_page(filename: str, route: str, page_label: str) -> List[KBChunk]:
    path = PAGES_DIR / filename
    if not path.exists():
        print(f"[skip] {filename} not found at {path}")
        return []

    source = path.read_text(encoding="utf-8")
    out: List[KBChunk] = []

    # 1. Heading+body pairs (best semantic units — feature/principle cards)
    pairs = _extract_headings_and_bodies(source)
    seen_texts: set[str] = set()
    for i, (heading, body) in enumerate(pairs):
        combined = f"{heading}. {body}"
        for chunk_text in _chunk_text(combined):
            if chunk_text in seen_texts:
                continue
            seen_texts.add(chunk_text)
            out.append(
                KBChunk(
                    id=f"{page_label}#pair-{i}-{len(out)}",
                    route=route,
                    page=page_label,
                    heading=heading,
                    text=chunk_text,
                )
            )

    # 2. Standalone high-signal strings (paragraph body props, subtitles,
    #    descriptions). These capture section-level copy that isn't in a
    #    title/body pair.
    standalone = _extract_standalone_strings(source)
    for i, text in enumerate(standalone):
        if len(text) < 40:
            continue
        if any(text in seen for seen in seen_texts):
            continue
        for chunk_text in _chunk_text(text):
            if chunk_text in seen_texts:
                continue
            seen_texts.add(chunk_text)
            out.append(
                KBChunk(
                    id=f"{page_label}#prop-{i}-{len(out)}",
                    route=route,
                    page=page_label,
                    heading=page_label,
                    text=chunk_text,
                )
            )

    # 3. JSX text nodes (e.g. top-level subtitle/paragraph elements). Lower
    #    precision, so we dedupe aggressively against what we already have.
    jsx_text = _extract_jsx_text(source)
    for i, text in enumerate(jsx_text):
        if len(text) < 60:
            continue
        if text in seen_texts:
            continue
        if any(text[:60] in seen[:200] for seen in seen_texts):
            continue
        seen_texts.add(text)
        for chunk_text in _chunk_text(text):
            out.append(
                KBChunk(
                    id=f"{page_label}#jsx-{i}-{len(out)}",
                    route=route,
                    page=page_label,
                    heading=page_label,
                    text=chunk_text,
                )
            )

    return out


def main() -> int:
    all_chunks: List[KBChunk] = []
    for filename, route, label in PAGES:
        chunks = ingest_page(filename, route, label)
        print(f"[ok]   {label:14} ({route}) -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    # Add a small static "meta" card so basic questions always have a hit
    static_meta = [
        KBChunk(
            id="meta#what-is-forma",
            route="/",
            page="Meta",
            heading="What is FORMA",
            text=(
                "FORMA is a real-time computer vision form coach for ten "
                "foundational exercises: squat, deadlift, bench press, overhead "
                "press, lunge, pull-up, push-up, plank, bicep curl, tricep dip. "
                "It runs in your browser with just a webcam — no wearables, no "
                "installs, no uploads. Pose estimation is done on-device via "
                "MediaPipe BlazePose, and each exercise has its own dedicated "
                "detector that scores your form in real time."
            ),
        ),
        KBChunk(
            id="meta#what-you-need",
            route="/",
            page="Meta",
            heading="What you need",
            text=(
                "You need a laptop or desktop with a webcam and a modern "
                "browser. No app install, no special hardware, no fitness "
                "tracker. FORMA works with your laptop's built-in camera."
            ),
        ),
        KBChunk(
            id="meta#privacy",
            route="/how-it-works",
            page="Meta",
            heading="Privacy and data",
            text=(
                "FORMA processes your webcam frames locally in the browser. "
                "Nothing is uploaded — no video, no frames, no biometric "
                "templates. The only things that ever reach the backend are "
                "the numeric session summaries you choose to save, and only "
                "when you are signed in."
            ),
        ),
        KBChunk(
            id="meta#how-to-start",
            route="/",
            page="Meta",
            heading="How to start",
            text=(
                "Click Sign In in the top-right, then switch to the Create "
                "Account tab. After signing up you land on the dashboard, where "
                "you can pick an exercise and start a session immediately."
            ),
        ),
        KBChunk(
            id="meta#features-list",
            route="/features",
            page="Meta",
            heading="Main features",
            text=(
                "FORMA ships with nine main features: real-time pose detection, "
                "ten dedicated per-exercise detectors, form score with live "
                "feedback cues, drill-down dashboard and session history, "
                "adaptive plans and goals, milestones and badges, an AI coach "
                "that knows your training history, a floating website guide "
                "for visitors, and on-device processing for privacy."
            ),
        ),
    ]
    all_chunks.extend(static_meta)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2, ensure_ascii=False)

    print(f"\n[done] {len(all_chunks)} chunks written to {OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
