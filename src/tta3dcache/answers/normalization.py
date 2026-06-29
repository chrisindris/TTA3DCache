from __future__ import annotations

import re
import unicodedata


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s']+")
_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def _canonicalize_spatial_terms(text: str) -> str:
    replacements = [
        (r"\bon the left side of the (.+)", r"left of \1"),
        (r"\bto the ([^']+?)'s left\b", r"left of \1"),
        (r"\bto the left of the (.+)", r"left of \1"),
        (r"\bleft of the (.+)", r"left of \1"),
        (r"\bleft of (.+)", r"left of \1"),
        (r"\bon the right side of the (.+)", r"right of \1"),
        (r"\bto the ([^']+?)'s right\b", r"right of \1"),
        (r"\bto the right of the (.+)", r"right of \1"),
        (r"\bright of the (.+)", r"right of \1"),
        (r"\bright of (.+)", r"right of \1"),
        (r"\bin front of the (.+)", r"in front of \1"),
        (r"\bin front of (.+)", r"in front of \1"),
        (r"\bnext to the (.+)", r"next to \1"),
        (r"\bnext to (.+)", r"next to \1"),
        (r"\bnear the (.+)", r"near \1"),
        (r"\bnear (.+)", r"near \1"),
        (r"\binside the (.+)", r"inside \1"),
        (r"\binside (.+)", r"inside \1"),
        (r"\bon top of the (.+)", r"on top of \1"),
        (r"\bon top of (.+)", r"on top of \1"),
        (r"\babove the (.+)", r"above \1"),
        (r"\babove (.+)", r"above \1"),
        (r"\bbelow the (.+)", r"below \1"),
        (r"\bbelow (.+)", r"below \1"),
    ]
    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def normalize_answer(text: str, *, canonicalize_spatial_terms: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    if canonicalize_spatial_terms:
        text = _canonicalize_spatial_terms(text)
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = _PUNCT_RE.sub(" ", text)
    text = _ARTICLE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    tokens = [_NUMBER_WORDS.get(token, token) for token in text.split()]
    return " ".join(tokens)


def normalize_open_answer(text: str, *, canonicalize_spatial_terms: bool = True) -> str:
    return normalize_answer(text, canonicalize_spatial_terms=canonicalize_spatial_terms)
