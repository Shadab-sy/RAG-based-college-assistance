import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

ACADEMIC_YEAR_PATTERN = re.compile(
    r"(?<!\d)(20\d{2})\s*(?:[-\u2013\u2014]\s*){1,2}(\d{2}|20\d{2})(?!\d)"
)


def normalize_academic_year(value: str) -> str:
    """Normalize an academic-year range while preserving the standard two-digit end year."""
    match = ACADEMIC_YEAR_PATTERN.search(value or "")
    if not match:
        return ""
    return f"{match.group(1)}-{match.group(2)[-2:]}"


def _years_in_text(value: str) -> List[str]:
    years = []
    for match in ACADEMIC_YEAR_PATTERN.finditer(value or ""):
        year = f"{match.group(1)}-{match.group(2)[-2:]}"
        if year not in years:
            years.append(year)
    return years


def _explicit_document_metadata(chunks: List[Dict[str, Any]]) -> str:
    for chunk in chunks:
        value = chunk.get("document_academic_year")
        if value:
            years = _years_in_text(str(value))
            if years:
                return "; ".join(years)
        metadata = chunk.get("document_metadata")
        if isinstance(metadata, dict) and metadata.get("academic_year"):
            years = _years_in_text(str(metadata["academic_year"]))
            if years:
                return "; ".join(years)
    return ""


def _filename_academic_year(document: str) -> str:
    years = _years_in_text(document)
    return years[0] if years else ""


def _heading_academic_year(chunks: List[Dict[str, Any]], document: str) -> str:
    if "nirf" in document.lower():
        return ""

    title_text = "\n".join(
        str(chunk.get("text", ""))
        for chunk in sorted(chunks, key=lambda item: (int(item.get("page", 0)), int(item.get("chunk_number_on_page", 0))))
        if int(chunk.get("page", 0)) <= 2
    )

    effective_year = re.search(
        r"with\s+effect\s+from\s+(?:the\s+)?(?:academic\s+year|a\.?y\.?|ay)\s*:?\s*([^\n.;]{0,45})",
        title_text,
        re.IGNORECASE,
    )
    if effective_year:
        years = _years_in_text(effective_year.group(1))
        if years:
            return "; ".join(years)

    year_label = re.search(
        r"(?:academic\s+year|a\.?\s*y\.?|\bay\b)\s*:?\s*([^\n.;]{0,55})",
        title_text,
        re.IGNORECASE,
    )
    if year_label:
        years = _years_in_text(year_label.group(1))
        if years:
            return "; ".join(years)

    squad_titles = re.finditer(
        r"anti.{0,12}ragging.{0,12}squad[^\n]{0,70}",
        title_text,
        re.IGNORECASE,
    )
    for squad_title in squad_titles:
        years = _years_in_text(squad_title.group(0))
        if years:
            return "; ".join(years)

    timetable_title = re.search(
        r"(?:time\s*table|timetable)[^\n]{0,100}\bAY\s*:?\s*([^\n.;]{0,30})",
        title_text,
        re.IGNORECASE,
    )
    if timetable_title:
        years = _years_in_text(timetable_title.group(1))
        if years:
            return "; ".join(years)
    return ""


def resolve_document_academic_year(document: str, chunks: List[Dict[str, Any]]) -> str:
    """Resolve one document-level academic year without reading arbitrary chunk years."""
    return (
        _explicit_document_metadata(chunks)
        or _filename_academic_year(document)
        or _heading_academic_year(chunks, document)
    )


def normalize_document_academic_years(
    chunks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], int]:
    """Set each chunk's academic_year from its document-level metadata/title, never its prose."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk.get("document", ""))].append(chunk)

    document_years = {
        document: resolve_document_academic_year(document, document_chunks)
        for document, document_chunks in grouped.items()
    }
    normalized = []
    changed_count = 0
    for chunk in chunks:
        row = dict(chunk)
        year = document_years[str(row.get("document", ""))]
        if str(row.get("academic_year", "")) != year:
            changed_count += 1
        row["academic_year"] = year
        normalized.append(row)
    return normalized, document_years, changed_count
