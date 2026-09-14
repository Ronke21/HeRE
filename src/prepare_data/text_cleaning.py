"""Shared Wikipedia text cleaning for the HeRE pipeline.

Single source of truth for `preprocess_text`, imported by both
`prepare_dataset.py` (gold) and `prepare_silver.py` (silver).

Historically these two scripts each carried their own copy of the cleaning
logic. They drifted: the whitespace-tolerant header fix of 2026-07-10 landed
in `prepare_silver.py` only, leaving the gold path on the old exact-string
matcher. Keeping the logic here means gold and silver cannot diverge again.

Cleaning levels
---------------
``clean(text, level="v3")`` — current default, see rules below.
``clean(text, level="v2")`` — the 2026-07-10 fixed silver logic, kept so
    older artifacts can be reproduced exactly.
``clean(text, level="v1")`` — the original buggy exact-string logic that
    produced `prepared_gold_500.csv`. Kept for reproducing the paper's
    published numbers; do not use for new work.

v3 rules, in order
------------------
0. Drop zero-width / bidi control characters and convert nbsp to a plain space.
   This runs *first* on purpose: dumps contain headers such as
   ``== קישורים\xa0חיצוניים ==`` whose words are separated by a non-breaking
   space. Normalising later lets those slip past the footer cut in step 1 and
   leaves the entire footer sitting in the passage.
1. Truncate at the first footer section header (tolerant of whitespace both
   around and *inside* the section name), or at a line-initial ``קטגוריה:``
   tag, whichever comes first.
2. Drop image lines entirely, caption included, along with a caption that
   spills onto the following line.
3. Unwrap wiki links: ``[[a|b]]`` -> ``b``, ``[[a]]`` -> ``a``.
4. Unwrap surviving section headers, keeping the words:
   ``== ביוגרפיה ==`` -> ``ביוגרפיה``, ``=== ניסוי ALPHA ===`` -> ``ניסוי ALPHA``.
5. Strip list bullets (``*``, ``#``), keeping the item text.
6. Collapse runs of spaces.
7. Drop blank lines and strip.

Note that *footer* sections never reach step 4 — step 1 truncates the passage at
the first one, so ``== קישורים חיצוניים ==`` and everything after it is gone
rather than unwrapped.
"""

from __future__ import annotations

import re

__all__ = ["clean", "clean_entity", "preprocess_text", "FOOTER_SECTIONS"]


# --------------------------------------------------------------------------
# Footer sections. Everything from the first of these onward is boilerplate
# (link lists, footnotes, image galleries, further-reading lists) and is cut.
# --------------------------------------------------------------------------
FOOTER_SECTIONS = (
    "קישורים חיצוניים",   # external links
    "ראו גם",             # see also
    "הערות שוליים",       # footnotes
    "לקריאה נוספת",       # further reading
    "גלריה",              # gallery
    "גלריית תמונות",      # image gallery
    "עיינו גם",           # see also (variant)
    "מקורות",             # sources
    "ביבליוגרפיה",        # bibliography
)

_V1_HEADERS = ["==קישורים חיצוניים==", "==ראו גם==", "==הערות שוליים=="]
_V2_SECTIONS = ("קישורים חיצוניים", "ראו גם", "הערות שוליים")

def _flex(section: str) -> str:
    """Allow any whitespace run between the words of a section name.

    Real dumps contain "קישורים\xa0חיצוניים" (non-breaking space) as well as the
    ordinary spaced form; a literal space in the pattern misses those.
    """
    return r"\s+".join(re.escape(w) for w in section.split())


_P_FOOTER_V2 = re.compile(r"==\s*(?:" + "|".join(_flex(s) for s in _V2_SECTIONS) + r")\s*==")
_P_FOOTER_V3 = re.compile(r"==\s*(?:" + "|".join(_flex(s) for s in FOOTER_SECTIONS) + r")\s*==")

# Line-anchored on purpose: an unanchored match truncates legitimate prose such
# as "...הכללים של הקטגוריה: סרטים זכאים...".
_P_CATEGORY_V2 = re.compile(r"\nקטגוריה:")
_P_CATEGORY_V3 = re.compile(r"(?m)^\s*קטגוריה:")

# Wiki links. Applied repeatedly to unwrap nested forms.
_P_LINK_PIPED = re.compile(r"\[\[[^\[\]\|]*\|([^\[\]]*)\]\]")
_P_LINK_PLAIN = re.compile(r"\[\[([^\[\]\|]*)\]\]")
_P_LINK_LEFTOVER = re.compile(r"\[\[|\]\]")

# Image / media syntax. The whole line is dropped, caption included: these are
# image captions, not article prose, and they carry the subject/object exclusively
# in ~0.3% of the rows that have them (measured on 20k silver rows).
#
# The variants below were all found in the corpus and each needed its own case:
#   שמאל|ממוזער|230px|caption          placement + thumb + size
#   קישור=[[..|ממוזער|298x29px|cap]]   inside a link
#   שגרירות סין...|273x273 פיקסלים     Hebrew "pixels" rather than "px"
#   שמאל|מסגרת|caption                 מסגרת = frame
#   קישור=דנמרק|טקסט=דנמרק|גבול|20px   flag-icon template remnant
#   right                              bare placement word on its own line
_IMG_SIZE = r"\d+\s*[xX×]?\s*\d*\s*(?:px|פיקסלים)"
_IMG_CORE = r"ממוזער|thumb|מסגרת|frameless|frame|ללא[_ ]מסגרת|upright"
_IMG_PLACE = r"right|left|center|שמאל|ימין|מרכז"

_P_IMAGE_CORE = re.compile(r"(?:" + _IMG_CORE + r")\s*\|")
# The same directives also appear as a *suffix*: "caption|ממוזער", "caption|ימין".
_P_IMAGE_CORE_SUFFIX = re.compile(
    r"\|\s*(?:" + _IMG_CORE + r"|" + _IMG_PLACE + r")\s*$")
# Infobox parameter lines: "|מסה= 2.9", "|גיל=", "|כוכבים נלווים=מוליפיין A/B".
_P_INFOBOX_PARAM = re.compile(r"^\s*\|\s*[^=|]{0,40}=")
# Attribute assignment after a pipe: "...|טקסט=", "...|קישור=".
_P_ATTR_AFTER_PIPE = re.compile(r"\|\s*(?:קישור|טקסט|link|alt)\s*=")
_P_IMAGE_SIZE = re.compile(_IMG_SIZE)
_P_IMAGE_ATTR = re.compile(r"^\s*(?:קישור|טקסט|link|alt)\s*=")
_P_IMAGE_PLACE_PREFIX = re.compile(r"^\s*(?:" + _IMG_PLACE + r")\s*\|")
_P_IMAGE_PLACE_BARE = re.compile(r"^\s*(?:" + _IMG_PLACE + r")\s*$", re.IGNORECASE)
# Table markup: cell-attribute lines and "||" row separators.
_P_TABLE = re.compile(r"\|\||^\s*(?:width|align|style|colspan|rowspan)\s*=")


def _is_media_line(line: str) -> bool:
    """True for image/table markup lines."""
    s = line.strip()
    if not s:
        return False
    if _P_IMAGE_PLACE_BARE.match(s):
        return True
    if "|" not in s:
        return False
    return bool(
        _P_IMAGE_CORE.search(s)
        or _P_IMAGE_CORE_SUFFIX.search(s)
        or _P_IMAGE_SIZE.search(s)
        or _P_IMAGE_ATTR.match(s)
        or _P_IMAGE_PLACE_PREFIX.match(s)
        or _P_INFOBOX_PARAM.match(s)
        or _P_ATTR_AFTER_PIPE.search(s)
        or _P_TABLE.search(s)
    )


# An article body is sometimes glued straight onto an image caption with no
# newline between them:
#
#   ממוזער|סרטוט כללי של שסתום בטיחותשסתום בטיחות הוא שסתום המשחרר...
#   ^--- directive ---^^--- caption ---^^--- the entire article ---^
#
# Dropping that line whole deletes the article. When substantial prose follows
# the last directive pipe, keep it: a caption prefix left on the front is a far
# smaller problem than losing the passage. Below the threshold the remainder is
# just the caption ("אות שמע", "273x273 פיקסלים"), so the line still goes.
# Set at 12 rather than something higher: a 24-word article body glued to a
# caption was being deleted wholesale by a 25-word threshold. Real captions run
# well under 12 words ("אות שמע", "273x273 פיקסלים"), so the cost of the lower
# bar is that an occasional long caption survives — much cheaper than losing a
# passage.
_MEDIA_KEEP_MIN_WORDS = 12

# A size token left at the start of retained text ("271x271 פיקסליםאלנה...") or
# on its own from a flag-icon template ("30px טאיוואן", "20px - כרמית בוריאן").
# Anchored at line start on purpose: a genuine mention mid-sentence, such as
# "ברזולוציה של 1920x1080 פיקסלים", is real content and must survive.
_P_LEADING_SIZE = re.compile(
    r"(?m)^\s*\d+\s*[xX×]?\s*\d*\s*(?:px|פיקסלים)\s*[-–—]?\s*")


def _media_remainder(line: str) -> str | None:
    """What to keep from a media line, or None to drop it entirely."""
    tail = line.rsplit("|", 1)[-1].strip()
    tail = _P_LEADING_SIZE.sub("", tail)
    if len(tail.split()) >= _MEDIA_KEEP_MIN_WORDS:
        return tail
    return None


# Leftover HTML tags ("</small>") appear in a couple of rows per 20k.
_P_HTML_TAG = re.compile(r"</?[a-zA-Z][^>\n]{0,60}>")

# A caption sometimes spills onto the line after the image directive
# ("[[פסקול]] בסרט", "ננסי קרוליין על [[אופנוע]]", "הלגולנד"). Such a line is
# short, has no sentence-ending punctuation, and is not a section header.
# Deliberately conservative: it fires on 0.74% of rows (measured on 20k silver).
_P_SECTION_LINE = re.compile(r"^\s*={2,}.*={2,}\s*$")

# Section headers that survive the footer cut keep their words but lose the
# "==" markup: "== ביוגרפיה ==" -> "ביוגרפיה". Applied to any depth of header
# ("=== ניסוי ALPHA ===" -> "ניסוי ALPHA").
_P_HEADER_ANY = re.compile(r"(?m)^\s*={2,}\s*(.*?)\s*={2,}\s*$")
_SENTENCE_END = (".", "!", "?", '"', ":")
_CONTINUATION_MAX_WORDS = 6


def _is_caption_continuation(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _P_SECTION_LINE.match(stripped):
        return False
    if len(stripped.split()) > _CONTINUATION_MAX_WORDS:
        return False
    return not stripped.endswith(_SENTENCE_END)

_P_BULLET = re.compile(r"(?m)^\s*[\*\#]+\s*")
_P_INVISIBLE = re.compile(r"[​-‏‪-‮⁠﻿­]")
_P_NBSP = re.compile(r"[   ]")
_P_SPACES = re.compile(r"[ \t]{2,}")


def _drop_blank_lines(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def clean(text: str, level: str = "v3") -> str:
    """Clean one Wikipedia passage. See module docstring for levels."""
    if not isinstance(text, str):
        return ""

    if level == "v1":
        cut = len(text)
        for header in _V1_HEADERS:
            idx = text.find(header)
            if idx != -1 and idx < cut:
                cut = idx
        return _drop_blank_lines(text[:cut])

    if level == "v2":
        cut = len(text)
        m = _P_FOOTER_V2.search(text)
        if m:
            cut = min(cut, m.start())
        m2 = _P_CATEGORY_V2.search(text)
        if m2:
            cut = min(cut, m2.start())
        return _drop_blank_lines(text[:cut])

    if level != "v3":
        raise ValueError(f"unknown cleaning level: {level!r} (expected v1, v2 or v3)")

    # 0. normalise invisible characters and nbsp FIRST. Doing this later lets a
    #    header like "== קישורים\xa0חיצוניים ==" slip past the footer cut and only
    #    become matchable afterwards, leaving the whole footer in the passage.
    text = _P_INVISIBLE.sub("", text)
    text = _P_NBSP.sub(" ", text)

    # 1. truncate at footer boilerplate
    cut = len(text)
    m = _P_FOOTER_V3.search(text)
    if m:
        cut = min(cut, m.start())
    m2 = _P_CATEGORY_V3.search(text)
    if m2:
        cut = min(cut, m2.start())
    text = text[:cut]

    # 2. drop image lines (caption included) and any caption continuation line
    #    that trails them. Runs before link unwrapping so section headers are
    #    still recognisable and are never mistaken for a continuation.
    kept = []
    after_image = False
    for line in text.splitlines():
        if not line.strip():
            continue                      # blanks go anyway; keep adjacency intact
        if _is_media_line(line):
            kept_tail = _media_remainder(line)
            if kept_tail is not None:
                after_image = False
                kept.append(kept_tail)
                continue
            after_image = True
            continue
        if after_image and _is_caption_continuation(line):
            continue
        after_image = False
        kept.append(line)
    text = "\n".join(kept)

    # 2b. strip any leftover HTML tags, keeping the text around them
    text = _P_HTML_TAG.sub("", text)

    # 3. unwrap wiki links (loop: handles nesting)
    for _ in range(3):
        new = _P_LINK_PIPED.sub(r"\1", text)
        new = _P_LINK_PLAIN.sub(r"\1", new)
        if new == text:
            break
        text = new
    text = _P_LINK_LEFTOVER.sub("", text)

    # 3b. catch-all, AFTER link unwrapping (which needs the pipe in "[[a|b]]"):
    #     a pipe surviving every structural rule above is markup residue —
    #     citation separators, stray template debris. Hebrew prose effectively
    #     never uses "|", so turning it into a space is safe.
    text = text.replace("|", " ")

    # 4. unwrap surviving section headers, keeping the heading words
    text = _P_HEADER_ANY.sub(r"\1", text)

    # 5. list bullets, then orphan size tokens left at line start by flag-icon
    #    templates ("[[קובץ:Flag.svg|30px]] טאיוואן" unwraps to "30px טאיוואן").
    #    Bullets first, so "* 20px טאיוואן" is reachable by the size rule.
    text = _P_BULLET.sub("", text)
    text = _P_LEADING_SIZE.sub("", text)

    # 6. collapse space runs (invisible chars already handled in step 0)
    text = _P_SPACES.sub(" ", text)

    # 7. blank lines
    return _drop_blank_lines(text)


def clean_entity(value: str) -> str:
    """Normalise a subject / predicate / object value.

    The entity columns carry the same markup the passages do, just far more
    rarely (measured over 200k silver rows: 496 subjects and 73 objects contain
    ``[[...]]``, 139 subjects and 48 objects contain a pipe, 81 subjects contain
    a non-breaking space). It matters out of proportion to its frequency because
    these values are (a) substring-matched against the passage and (b) embedded
    verbatim into `basic_relation` / `template_relation`, so an uncleaned value
    puts wiki markup straight into the hypothesis fed to the model.

    Not applied automatically to existing datasets — changing entity values
    changes the data, so it is opt-in per script.
    """
    if not isinstance(value, str):
        return ""
    value = _P_INVISIBLE.sub("", value)
    value = _P_NBSP.sub(" ", value)
    for _ in range(3):
        new = _P_LINK_PIPED.sub(r"\1", value)
        new = _P_LINK_PLAIN.sub(r"\1", new)
        if new == value:
            break
        value = new
    value = _P_LINK_LEFTOVER.sub("", value)
    value = _P_HTML_TAG.sub("", value)
    value = value.replace("|", " ")
    return _P_SPACES.sub(" ", value).strip()


def preprocess_text(text: str) -> str:
    """Backwards-compatible alias used by the prepare_* scripts."""
    return clean(text, level="v3")
