"""
Medical Metadata Extractor
Enriches documents with domain, publication year, and title inference.

Fixes:
- Title inference skips noisy lines (all-caps headers, page numbers,
  very short strings, lines starting with common noise patterns)
- Year regex and domain inference unchanged (already correct)
"""
from __future__ import annotations

import re
from typing import Optional
from langchain_core.documents import Document

# Medical domain keyword mapping
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "cardiology": [
        "heart", "cardiac", "myocardial", "cardiovascular", "hypertension",
        "arrhythmia", "ecg", "electrocardiogram", "coronary", "atrial",
        "ventricular", "angina", "stroke", "thrombosis",
    ],
    "oncology": [
        "cancer", "tumor", "tumour", "malignant", "chemotherapy", "radiation",
        "metastasis", "biopsy", "oncology", "carcinoma", "lymphoma", "leukemia",
    ],
    "neurology": [
        "brain", "neuron", "neurological", "epilepsy", "alzheimer", "parkinson",
        "dementia", "seizure", "migraine", "stroke", "cerebral", "spinal",
    ],
    "pulmonology": [
        "lung", "respiratory", "asthma", "copd", "pneumonia", "bronchitis",
        "pulmonary", "oxygen", "spirometry", "inhaler", "bronchial",
    ],
    "endocrinology": [
        "diabetes", "insulin", "thyroid", "hormone", "glucose", "pancreas",
        "endocrine", "cortisol", "adrenal", "pituitary",
    ],
    "gastroenterology": [
        "gastrointestinal", "stomach", "liver", "hepatic", "colon", "bowel",
        "intestine", "crohn", "ulcer", "gastric", "endoscopy",
    ],
    "infectious_disease": [
        "infection", "bacteria", "virus", "antibiotic", "antimicrobial",
        "pathogen", "sepsis", "hiv", "covid", "influenza", "vaccination",
    ],
    "pharmacology": [
        "drug", "medication", "dose", "dosage", "pharmacokinetics",
        "adverse effect", "contraindication", "interaction", "prescription",
    ],
    "general_medicine": [],  # fallback
}

# Patterns for lines that are clearly NOT titles
_RE_PURE_DIGITS   = re.compile(r"^[\d\s\.\-]+$")           # page numbers / numbering
_RE_ALL_CAPS      = re.compile(r"^[A-Z\s\d\W]{8,}$")       # ALL-CAPS banner headers
_RE_URL           = re.compile(r"https?://\S+")            # URLs
_RE_EMAIL         = re.compile(r"\S+@\S+\.\S+")            # email addresses
_RE_NOISE_PREFIX  = re.compile(                            # common PDF header noise
    r"^(abstract|keywords?|introduction|references?|doi:|issn:|isbn:)",
    re.IGNORECASE,
)


def infer_domain(text: str) -> str:
    """Infer medical domain from document content via keyword frequency."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if domain == "general_medicine":
            continue
        score = sum(text_lower.count(kw) for kw in keywords)
        if score > 0:
            scores[domain] = score
    if not scores:
        return "general_medicine"
    return max(scores, key=lambda d: scores[d])


def infer_year(text: str) -> Optional[int]:
    """Extract most prominent year (1980–2029) from document text."""
    matches = re.findall(r"\b(19[89]\d|20[0-2]\d)\b", text)
    if not matches:
        return None
    from collections import Counter
    counter = Counter(matches)
    return int(counter.most_common(1)[0][0])


def _is_noisy_line(line: str) -> bool:
    """Return True if a line is clearly not a document title."""
    if _RE_PURE_DIGITS.match(line):
        return True
    if _RE_ALL_CAPS.match(line) and len(line) < 60:
        # Short all-caps strings are usually section headers / journal names
        return True
    if _RE_URL.search(line) or _RE_EMAIL.search(line):
        return True
    if _RE_NOISE_PREFIX.match(line):
        return True
    return False


def infer_title(text: str, filename: str) -> str:
    """
    Infer document title from the first substantive non-noisy line.

    Heuristics (in order):
    1. Skip blank lines, pure-digit lines, all-caps banners, URLs, emails,
       and lines starting with section keywords (Abstract, Keywords, DOI …)
    2. Accept the first line that is 15–250 chars long and not noisy
    3. Fall back to a cleaned filename
    """
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) < 15 or len(line) > 250:
            continue
        if _is_noisy_line(line):
            continue
        return line

    # Filename fallback (strip extension, replace separators with spaces)
    stem = Path(filename).stem if filename else "Unknown Document"
    return stem.replace("_", " ").replace("-", " ").title()


def enrich_metadata(doc: Document) -> Document:
    """Enrich a single Document with inferred metadata fields."""
    text = doc.page_content
    meta = doc.metadata

    if not meta.get("domain"):
        meta["domain"] = infer_domain(text)

    if not meta.get("year"):
        year = infer_year(text)
        if year:
            meta["year"] = year

    if not meta.get("title"):
        filename = meta.get("source", "unknown.pdf")
        meta["title"] = infer_title(text, filename)

    doc.metadata = meta
    return doc


# Local import to avoid circular at module level
from pathlib import Path  # noqa: E402 (needed for infer_title fallback)
