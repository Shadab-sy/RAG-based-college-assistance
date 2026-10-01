import re
import logging
from datetime import date
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.config import settings
from app.document_metadata import ACADEMIC_YEAR_PATTERN, normalize_academic_year
from app.embeddings import get_embedding_manager
from app.reranker import get_cross_encoder_reranker
from app.vector_store import get_vector_store

logger = logging.getLogger(__name__)
_reranker_warning_emitted = False

# Standard non-informative stopwords
STOPWORDS = {
    "what", "is", "are", "the", "in", "for", "of", "to", "a", "an", "and", "or",
    "by", "on", "with", "who", "can", "how", "do", "does", "available", "about",
    "from", "at", "as", "this", "that", "these", "those", "their", "there"
}

INTENT_TERMS = {
    "FEE": ("fee", "fees", "tuition", "charges", "payment"),
    "SCHOLARSHIP": ("scholarship", "freeship", "pragati", "saksham", "maha-dbt", "dbt"),
    "GRIEVANCE": ("grievance", "redressal", "complaint", "sgrc"),
    "ANTI_RAGGING": ("ragging", "anti-ragging", "squad"),
    "ADMISSION": ("admission", "eligibility", "intake", "enrolment", "dform", "seat"),
    "INTERNSHIP": ("internship", "summer internship"),
    "SYLLABUS": ("syllabus", "curriculum", "course", "subject", "scheme"),
    "ACADEMIC_STRUCTURE": ("academic structure", "academic calendar", "semester", "credits", "department"),
    "EXAMINATION": ("exam", "examination", "timetable", "time table", "schedule", "ese", "end semester"),
}

INTENT_DOCUMENT_TYPES = {
    "FEE": {"fee_structure"},
    "SCHOLARSHIP": {"scholarship"},
    "GRIEVANCE": {"grievance"},
    "ANTI_RAGGING": {"anti_ragging"},
    "ADMISSION": {"admission_form", "prospectus"},
    "INTERNSHIP": {"internship"},
    "SYLLABUS": {"syllabus", "academic_document"},
    "ACADEMIC_STRUCTURE": {"academic_document", "syllabus", "prospectus"},
    "EXAMINATION": {"examination", "exam_timetable"},
    "PERSON_ROLE": {"prospectus", "grievance", "iqac", "anti_ragging"},
}

ROLE_TERMS = {
    "principal": r"principal",
    "associate_dean": r"associate\s+dean",
    "dean": r"(?<!associate )dean",
    "director": r"director",
    "registrar": r"registrar",
    "hod": r"(?:hod|head\s+of\s+(?:the\s+)?department)",
    "coordinator": r"coordinator",
}
DEPARTMENT_ALIASES = {
    "it": (r"\bIT\b", r"\binformation\s+technology\b"),
    "aiml": (
        r"\bAIML\b",
        r"\bAI\s*&\s*ML\b",
        r"\bAI\s+and\s+ML\b",
        r"\bartificial\s+intelligence\s+(?:and|&)\s+machine\s+learning\b",
    ),
    "computer": (r"\bcomputer\s+engineering\b", r"\bCO\b"),
    "extc": (r"\bEXTC\b", r"\belectronics\s+(?:and|&)\s+telecommunication\b"),
    "etrx": (r"\bETRX\b", r"\belectronics(?:\s+engineering)?\b"),
    "civil": (r"\bcivil\s+engineering\b", r"\bcivil\b"),
    "mech": (r"\bmechanical\s+engineering\b", r"\bmech\b"),
}

DEPARTMENT_QUERY_ALIASES = (
    ("aiml", (r"\bartificial\s+intelligence\s+(?:and|&)\s+machine\s+learning\b", r"\bAI\s*&\s*ML\b", r"\bAI\s+and\s+ML\b", r"\bAIML\b")),
    ("computer", (r"\bcomputer\s+engineering\b", r"\bCO\b")),
    ("extc", (r"\belectronics\s+(?:and|&)\s+telecommunication\b", r"\bEXTC\b")),
    ("etrx", (r"\belectronics(?:\s+engineering)?\b", r"\bETRX\b")),
    ("it", (r"\binformation\s+technology\b", r"\bIT\b")),
    ("civil", (r"\bcivil\s+engineering\b", r"\bcivil\b")),
    ("mech", (r"\bmechanical\s+engineering\b", r"\bmech\b")),
)

ROLE_EVIDENCE_PATTERN = re.compile(
    r"\b(?P<label>(?:(?:acting(?:\s+as(?:\s+an?)?)?|in[\s-]*charge|i\s*/\s*c|l\s*/\s*c)\s+)*"
    r"(?:principal|associate\s+dean|(?<!associate )dean|director|registrar|hod|head\s+of\s+(?:the\s+)?department)"
    r"(?:\s+(?:IT|AIML|CO|EXTC|ETRX))?\b"
    r"(?:\s*\(\s*(?:in[\s-]*charge|i\s*/\s*c|l\s*/\s*c|acting)\s*\))?)",
    re.IGNORECASE,
)

PERSON_NAME_PATTERN = re.compile(
    r"\b(?:Dr|Prof|Professor|Doctor)\.?\s+[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3}",
    re.MULTILINE,
)
EXAM_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*,?\s+20\d{2}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+20\d{2}|"
    r"\d{1,2}[/-]\d{1,2}[/-]20\d{2}|20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b",
    re.IGNORECASE,
)

class RetrievedChunk(BaseModel):
    chunk_id: str
    document: str
    document_type: str
    page: int
    academic_year: str
    chunk_number_on_page: int
    text: str
    distance: float
    semantic_score: float
    lexical_score: float
    entity_score: float = 0.0
    proximity_score: float = 0.0
    department_score: float = 0.0
    role_qualifier_score: float = 0.0
    temporal_score: float = 0.0
    source_authority_score: float = 0.0
    cross_encoder_score: Optional[float] = None
    matched_name: str = ""
    matched_role: str = ""
    metadata_score: float
    answerability_score: float = 0.0
    relevance_score: float = 0.0
    final_score: float
    similarity_score: float  # Alias for final_score to support existing downstream callers

class RetrievalResponse(BaseModel):
    query: str
    normalized_query: str = ""
    intent: str = "GENERAL"
    top_k: int
    results: List[RetrievedChunk]
    answerability_score: float = 0.0
    relevance_score: float = 0.0
    top_composite_score: float = 0.0
    temporal_score: float = 0.0
    cross_encoder_applied: bool = False
    reranker_note: Optional[str] = None
    status: str = "INSUFFICIENT_KNOWLEDGE"
    conflicting_evidence: bool = False
    role_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    requires_personal_context: bool = False
    has_sufficient_context: bool = True
    confidence_note: Optional[str] = None


def _is_person_role_query(query: str) -> bool:
    """Recognize person/role questions without treating conceptual principles as roles."""
    q_lower = query.lower()
    concept_context = re.search(r"\b(working|basic|fundamental|operating|scientific)\s+principle\b", q_lower)
    explicit_role = re.search(
        r"\b(principal|dean|director|registrar|hod|head\s+of\s+(?:the\s+)?department|coordinator)\b",
        q_lower,
    )
    person_cue = re.search(r"\b(who|name|dr|prof|professor)\b", q_lower)
    typo_role_cue = re.search(
        r"\bwho\b.{0,24}\b(?:is|was)\b.{0,12}\bprinciple\b|"
        r"\bprinciple\b.{0,30}\bof\s+(?:the\s+)?(?:college|institution|department)\b",
        q_lower,
    )
    return bool(explicit_role and not concept_context or person_cue and typo_role_cue)


def _requested_department_patterns(query: str) -> tuple[str, ...]:
    role_match = re.search(r"\b(?:hod|head\s+of\s+(?:the\s+)?department)\b", query, re.I)
    if not role_match:
        return ()
    role_context = query[role_match.end():]
    for _, aliases in DEPARTMENT_QUERY_ALIASES:
        if any(re.search(alias, role_context, re.I if alias.islower() else 0) for alias in aliases):
            return aliases
    return ()


def _department_pattern_matches(pattern: str, text: str) -> bool:
    flags = 0 if pattern in {r"\bIT\b", r"\bAIML\b", r"\bCO\b", r"\bEXTC\b", r"\bETRX\b"} else re.I
    return re.search(pattern, text, flags) is not None


def _role_matches_department(query: str, text: str, role_match: re.Match[str]) -> bool:
    patterns = _requested_department_patterns(query)
    if not patterns:
        return True
    department_matches = [
        match
        for pattern in patterns
        for match in re.finditer(pattern, text, 0 if pattern in {r"\bIT\b", r"\bAIML\b", r"\bCO\b", r"\bEXTC\b", r"\bETRX\b"} else re.I)
    ]
    return any(
        max(0, max(role_match.start(), department.start()) - min(role_match.end(), department.end())) <= 260
        for department in department_matches
    )


def _display_person_name(name: str) -> str:
    cleaned = re.sub(r"\s+(?:ph\.?d\.?|m\.?e\.?|m\.?tech\.?|b\.?e\.?|m\.?sc\.?)\b.*$", "", name, flags=re.I)
    cleaned = re.sub(
        r"\s+(?:(?:in[\s-]*charge|acting|i\s*/\s*c|l\s*/\s*c)\s+)*(?:associate\s+dean|principal|dean|director|registrar|hod|head\s+of\s+(?:the\s+)?department|coordinator)\b.*$",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+(?:in[\s-]*charge|acting|i\s*/\s*c|l\s*/\s*c)\s*$", "", cleaned, flags=re.I)
    return " ".join(cleaned.split())


def _clean_person_name(name: str) -> str:
    return _display_person_name(name).lower()


def _query_role_keys(query: str) -> set[str]:
    normalized = normalize_query(query).lower()
    return {
        key for key, pattern in ROLE_TERMS.items()
        if re.search(rf"\b(?:{pattern})\b", normalized)
    }


def _requested_role_qualifier(query: str) -> str:
    normalized = query.lower()
    if re.search(r"\bacting\b", normalized):
        return "acting"
    if re.search(r"\b(?:in[\s-]*charge|i\s*/\s*c|l\s*/\s*c)\b", normalized):
        return "in_charge"
    return ""


def _role_evidence_matches(query: str, text: str) -> List[Dict[str, Any]]:
    """Find nearby titled names and exact role labels, constrained to the requested department."""
    requested_roles = _query_role_keys(query)
    requested_department = bool(_requested_department_patterns(query))
    requested_qualifier = _requested_role_qualifier(query)
    role_matches = []
    for match in ROLE_EVIDENCE_PATTERN.finditer(text):
        label = match.group("label")
        label_lower = label.lower()
        role_key = "associate_dean" if re.search(r"\bassociate\s+dean\b", label_lower) else "hod" if re.search(r"\b(?:hod|head\s+of\s+(?:the\s+)?department)\b", label_lower) else next(
            (key for key in ("principal", "dean", "director", "registrar", "coordinator") if re.search(rf"\b{key}\b", label_lower)),
            "",
        )
        if requested_roles and role_key not in requested_roles:
            continue
        has_acting = bool(re.search(r"\bacting\b", label_lower))
        has_in_charge = bool(re.search(r"(?:in[\s-]*charge|i\s*/\s*c|l\s*/\s*c)", label_lower))
        if requested_qualifier == "acting" and not (has_acting or has_in_charge):
            continue
        if requested_qualifier == "in_charge" and not has_in_charge:
            continue
        if role_key == "dean" and "associate_dean" in requested_roles and not re.search(r"associate", label_lower):
            continue
        if requested_department and not _role_matches_department(query, text, match):
            continue
        if not requested_qualifier:
            qualifier_score = 1.0
        elif requested_qualifier == "acting":
            qualifier_score = 1.0 if has_acting else 0.45 if has_in_charge else 0.0
        elif re.search(r"\bin[\s-]*charge\b", label_lower):
            qualifier_score = 1.0
        elif re.search(r"\bi\s*/\s*c\b", label_lower):
            qualifier_score = 0.8
        elif re.search(r"\bl\s*/\s*c\b", label_lower):
            qualifier_score = 0.55
        else:
            qualifier_score = 0.0
        role_matches.append((match, role_key))

    names = list(PERSON_NAME_PATTERN.finditer(text))
    all_role_matches = list(ROLE_EVIDENCE_PATTERN.finditer(text))
    pairs = []
    for role_match, role_key in role_matches:
        for name_match in names:
            distance = min(abs(role_match.end() - name_match.start()), abs(name_match.end() - role_match.start()))
            if distance > 120:
                continue
            nearest_role_distance = min(
                min(abs(match.end() - name_match.start()), abs(name_match.end() - match.start()))
                for match in all_role_matches
            )
            if distance > nearest_role_distance:
                continue
            label = role_match.group("label").strip(" ,:-")
            display_name = _display_person_name(name_match.group(0))
            pairs.append({
                "name": display_name,
                "normalized_name": display_name.lower(),
                "role": label,
                "role_key": role_key,
                "distance": distance,
                "department_score": 1.0 if requested_department else 0.0,
                "qualifier_score": qualifier_score,
                "qualifier": "acting" if has_acting else "in_charge" if has_in_charge else "",
            })
    return sorted(pairs, key=lambda pair: pair["distance"])


def _document_academic_years(value: str) -> set[str]:
    years = set()
    for item in (value or "").split(";"):
        normalized = normalize_academic_year(item.strip())
        if normalized:
            years.add(normalized)
    if not years:
        normalized = normalize_academic_year(value or "")
        if normalized:
            years.add(normalized)
    return years


def compute_temporal_score(query: str, academic_year: str) -> float:
    """Score a document's explicit academic year against requested or current period intent."""
    query_years = _document_academic_years(query)
    document_years = _document_academic_years(academic_year)
    if query_years:
        return 1.0 if query_years.intersection(document_years) else 0.0

    if not re.search(r"\b(current|currently|now|latest|most recent)\b", query, re.I):
        return 0.0
    today = date.today()
    start_year = today.year if today.month >= 6 else today.year - 1
    current_year = f"{start_year}-{(start_year + 1) % 100:02d}"
    if current_year in document_years:
        return 1.0
    if not document_years:
        return 0.0
    latest_document_year = max(document_years, key=lambda value: int(value[:4]))
    return 0.25 if int(latest_document_year[:4]) < start_year else 0.0


def compute_source_authority_score(intent: str, doc_type: str, text: str) -> float:
    """Provide a small source-type signal, contingent on explicit role evidence."""
    if intent != "PERSON_ROLE" or not ROLE_EVIDENCE_PATTERN.search(text):
        return 0.0
    dtype = doc_type.lower()
    if re.search(r"\boffice\s+order\b", text, re.I):
        return 1.0
    return {
        "grievance": 0.85,
        "prospectus": 0.75,
        "anti_ragging": 0.65,
        "iqac": 0.25,
        "general": 0.35,
    }.get(dtype, 0.2)


def normalize_query(query: str) -> str:
    """Apply contextual role correction and common scholarship spelling corrections."""
    normalized = query.strip()
    if _is_person_role_query(normalized):
        normalized = re.sub(r"\bprinciple\b", "principal", normalized, flags=re.IGNORECASE)
    return re.sub(r"\b(?:scolarship|scholership|scholorship)\b", "scholarship", normalized, flags=re.IGNORECASE)


def detect_query_intent(query: str) -> str:
    """Return a deterministic intent label for retrieval ranking and answerability checks."""
    normalized = normalize_query(query).lower()
    if _is_person_role_query(normalized):
        return "PERSON_ROLE"
    if re.search(r"\b(fee|fees|tuition|charges|payment)\b", normalized):
        return "FEE"
    if re.search(r"\b(scholarship|freeship|pragati|saksham|maha-dbt|dbt)\b", normalized):
        return "SCHOLARSHIP"
    if re.search(r"\b(grievance|redressal|complaint|sgrc)\b", normalized):
        return "GRIEVANCE"
    if re.search(r"\b(ragging|anti-ragging|squad)\b", normalized):
        return "ANTI_RAGGING"
    if re.search(r"\b(admission|eligibility|intake|enrolment|dform|seats?)\b", normalized):
        return "ADMISSION"
    if re.search(r"\b(internship|internships)\b", normalized):
        return "INTERNSHIP"
    if re.search(r"\b(syllabus|curriculum|courses?|subjects?|scheme)\b", normalized):
        return "SYLLABUS"
    if re.search(r"\b(exam|examination|timetable|time\s+table|schedule|ese|end\s+semester)\b", normalized):
        return "EXAMINATION"
    if re.search(r"\b(academic structure|academic calendar|semester|credits?|departments?)\b", normalized):
        return "ACADEMIC_STRUCTURE"
    return "GENERAL"

def extract_meaningful_terms(query: str) -> List[str]:
    """
    Extract informative keywords from user query, preserving engineering acronyms like 'BE'.
    """
    query = normalize_query(query)
    tokens = re.findall(r'\b[A-Za-z0-9\-\_]+\b', query)
    meaningful = []
    has_be_acronym = False

    for t in tokens:
        # Check for uppercase BE (Bachelor of Engineering)
        if t == "BE":
            has_be_acronym = True
            meaningful.append("be")
        elif t.lower() not in STOPWORDS and len(t) > 1:
            meaningful.append(t.lower())

    if has_be_acronym and "be" not in meaningful:
        meaningful.append("be")

    return list(dict.fromkeys(meaningful))

def compute_lexical_score(query_terms: List[str], chunk_text: str, doc_name: str) -> float:
    """
    Calculate lexical relevance score based on query keywords matching within
    the chunk text and the document filename.
    Returns a normalized score in [0.0, 1.0].
    """
    if not query_terms:
        return 0.0

    text_lower = chunk_text.lower()
    doc_lower = doc_name.lower()

    text_matched = 0
    doc_matched = 0
    freq_total = 0

    for term in query_terms:
        pattern = r'\b' + re.escape(term) + r'\b'
        matches = len(re.findall(pattern, text_lower))
        if matches > 0:
            text_matched += 1
            freq_total += min(matches, 5)

        # Document filename matching is a strong relevance signal
        if re.search(pattern, doc_lower):
            doc_matched += 1

    term_count = len(query_terms)
    coverage = text_matched / term_count
    doc_signal = doc_matched / term_count
    freq_bonus = min(1.0, freq_total / (term_count * 3))

    lexical_score = (0.50 * coverage) + (0.30 * doc_signal) + (0.20 * freq_bonus)
    return min(1.0, max(0.0, lexical_score))

def compute_metadata_score(
    query: str,
    query_terms: List[str],
    doc_type: str,
    academic_year: str,
    doc_name: str,
    intent: Optional[str] = None,
    chunk_text: str = "",
) -> float:
    """
    Calculate metadata score based on:
    1. Query intent keywords matching chunk's document_type
    2. Direct token match with document name
    3. Academic year match between query and metadata
    Returns a normalized score in [0.0, 1.0].
    """
    q_lower = normalize_query(query).lower()
    doc_lower = doc_name.lower()
    score = 0.0
    intent = intent or detect_query_intent(query)

    if doc_type.lower() in INTENT_DOCUMENT_TYPES.get(intent, set()):
        score += 0.55
    elif intent == "SYLLABUS" and doc_type.lower() == "academic_document":
        score += 0.20

    name_hints = {
        "FEE": ("fee", "fees"),
        "SCHOLARSHIP": ("scholarship", "pragati", "saksham"),
        "GRIEVANCE": ("grievance", "redressal"),
        "ANTI_RAGGING": ("ragging", "squad"),
        "ADMISSION": ("admission", "prospectus", "dform"),
        "INTERNSHIP": ("internship",),
        "SYLLABUS": ("syllabus",),
        "EXAMINATION": ("exam", "timetable", "time table"),
    }
    if any(hint in doc_lower for hint in name_hints.get(intent, ())):
        score += 0.25
    if intent == "EXAMINATION" and re.search(r"\b(timetable|time\s+table|schedule)\b", chunk_text, re.I):
        score += 0.10

    # 3. Academic Year Match (e.g. 2026-27 or 2025-26)
    year_match = re.search(r'\b(20\d\d[-–]\d\d|\d\d\d\d)\b', query)
    if year_match:
        target_year = year_match.group(1).replace('–', '-')
        if academic_year and target_year in academic_year:
            score += 0.20

    return min(1.0, max(0.0, score))


def compute_entity_proximity_scores(query: str, intent: str, chunk_text: str) -> tuple[float, float]:
    """Score explicit role/name evidence and the textual distance between them."""
    if intent != "PERSON_ROLE":
        return 0.0, 0.0

    pairs = _role_evidence_matches(query, chunk_text)
    if not pairs:
        return 0.0, 0.0
    proximity = max(0.0, 1.0 - pairs[0]["distance"] / 120.0)
    entity_score = 0.75 + 0.25 * pairs[0]["qualifier_score"]
    return round(entity_score, 4), round(proximity, 4)


def _has_exam_date_evidence(text: str, doc_name: str) -> bool:
    """Require an exam-specific schedule and a date near that evidence."""
    combined = f"{doc_name}\n{text}"
    combined = re.sub(
        r"(?im)\bPage\s+\d+(?:\s*/\s*\d+)?\s*(?:\r?\n)?\s*\d{1,2}[/-]\d{1,2}[/-]20\d{2}(?:\s+\d{1,2}:\d{2}:\d{2})?",
        " ",
        combined,
    )
    schedule_matches = list(re.finditer(r"\b(timetable|time\s+table|exam\s+schedule|examination\s+schedule)\b", combined, re.I))
    exam_matches = list(re.finditer(r"\b(exam|examination|ese|end\s+semester)\b", combined, re.I))
    date_matches = list(EXAM_DATE_PATTERN.finditer(combined))
    for schedule in schedule_matches:
        for exam in exam_matches:
            for date in date_matches:
                if max(schedule.start(), exam.start(), date.start()) - min(schedule.start(), exam.start(), date.start()) <= 220:
                    return True
    return False


def _is_personal_eligibility_query(query: str) -> bool:
    return bool(re.search(
        r"\b(?:am\s+i|do\s+i|for\s+which\s+scholarship\s+am\s+i)\b.{0,40}\b(?:eligible|qualify|apply|get)\b",
        query,
        re.I,
    ))


def _has_personal_eligibility_details(query: str) -> bool:
    detail_patterns = (
        r"\b(girl|female|woman|boy|male)\b",
        r"\b(?:first|second|third|fourth)[- ]year\b",
        r"\b(?:income|family income|annual income|lakh|rupee|rs\.?|inr)\b",
        r"\b(disabilit\w*|specially[- ]abled|sc|st|obc|nt|sbc)\b",
        r"\b(?:admitted|enrolled|studying)\b",
    )
    return sum(bool(re.search(pattern, query, re.I)) for pattern in detail_patterns) >= 2


def _asks_for_admission_documents(query: str) -> bool:
    return bool(
        re.search(r"\b(document|documents|paperwork)\b", query, re.I)
        and re.search(r"\b(admission|required|need|submit)\b", query, re.I)
    )


def compute_answerability_score(
    intent: str,
    query_terms: List[str],
    text: str,
    doc_type: str,
    doc_name: str,
    lexical_score: float,
    entity_score: float = 0.0,
    proximity_score: float = 0.0,
    query: str = "",
    academic_year: str = "",
) -> float:
    """Estimate whether a result contains direct evidence, independently of relevance."""
    text_lower = text.lower()
    dtype = doc_type.lower()

    if intent == "PERSON_ROLE":
        return round(min(1.0, 0.6 * entity_score + 0.4 * proximity_score), 4)
    if intent == "EXAMINATION":
        return 0.95 if _has_exam_date_evidence(text, doc_name) else 0.10 if re.search(r"\b(exam|examination|ese)\b", text_lower) else 0.0
    if intent == "FEE":
        if dtype == "fee_structure" and re.search(r"\b(fee|fees|tuition)\b", text_lower):
            return 0.9 if re.search(r"\b(?:rs\.?|inr|\d[\d,]{3,})\b", text_lower) else 0.75
    elif intent == "SCHOLARSHIP":
        if _is_personal_eligibility_query(query):
            return 0.0
        if dtype == "scholarship" and re.search(r"\b(scholarship|eligible|eligibility|apply|application)\b", text_lower):
            return 0.9 if re.search(r"\b(eligible|eligibility|apply|application)\b", text_lower) else 0.75
    elif intent == "GRIEVANCE":
        if re.search(r"\b(grievance|redressal|complaint|sgrc)\b", text_lower) and re.search(r"\b(committee|procedure|process|submit|approach|complaint)\b", text_lower):
            return 0.85
    elif intent == "ANTI_RAGGING":
        if re.search(r"\b(ragging|anti-ragging)\b", text_lower) and re.search(r"\b(committee|squad|report|contact|prohibit)\b", text_lower):
            return 0.85
    elif intent == "ADMISSION":
        if _asks_for_admission_documents(query) and not re.search(
            r"\b(?:required admission documents|documents required for admission|submit the following documents)\b",
            text_lower,
        ):
            return 0.0
        if dtype in {"admission_form", "prospectus"} and re.search(r"\b(admission|eligibility|eligible|intake|seat)\b", text_lower):
            return 0.85
    elif intent == "INTERNSHIP":
        if re.search(r"\b(available|currently|current|now|latest)\b", query, re.I):
            source_years = _document_academic_years(academic_year)
            content_years = _document_academic_years(text)
            source_years.update(content_years)
            dated_opportunity = re.search(r"\b(?:summer|winter|spring|fall)\s+20\d{2}\b", text, re.I)
            if dated_opportunity:
                source_years.add(dated_opportunity.group(0)[-4:] + "-" + str((int(dated_opportunity.group(0)[-4:]) + 1) % 100).zfill(2))
            current_year = date.today().year
            if not source_years or max(int(year[:4]) for year in source_years) < current_year:
                return 0.0
        if (dtype == "internship" or "internship" in text_lower) and re.search(r"\b(internship|internships)\b", text_lower):
            return 0.85
    elif intent == "SYLLABUS":
        if dtype in {"syllabus", "academic_document"} and re.search(r"\b(syllabus|curriculum|course|subject|semester)\b", text_lower):
            return 0.85
    elif intent == "ACADEMIC_STRUCTURE":
        if re.search(r"\b(semester|credits?|department|academic calendar|academic structure)\b", text_lower):
            return 0.75

    if intent == "GENERAL" and query_terms:
        matches = sum(bool(re.search(rf"\b{re.escape(term)}\b", text_lower)) for term in query_terms)
        coverage = matches / len(query_terms)
        if coverage >= 0.75:
            return round(min(0.8, 0.45 + 0.35 * coverage), 4)
    return round(min(0.35, lexical_score * 0.35), 4)


def _role_name_evidence(query: str, text: str) -> set[str]:
    """Return names close to the queried role, for detecting conflicting dated records."""
    return {
        pair["normalized_name"]
        for pair in _role_evidence_matches(query, text)
        if pair["normalized_name"]
    }


def _select_diverse_chunks(chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
    selected: List[RetrievedChunk] = []
    document_counts: Dict[str, int] = {}
    for chunk in chunks:
        count = document_counts.get(chunk.document, 0)
        if count < 2 or chunk.answerability_score >= 0.75:
            selected.append(chunk)
            document_counts[chunk.document] = count + 1
        if len(selected) >= top_k:
            break
    return selected


def format_evidence_excerpt(query: str, text: str, max_chars: int = 300) -> str:
    """Center person-role previews on the closest role/name pair; otherwise show the chunk start."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars or detect_query_intent(query) != "PERSON_ROLE":
        return cleaned if len(cleaned) <= max_chars else cleaned[:max_chars] + "..."

    normalized = normalize_query(query).lower()
    selected_roles = [
        pattern for pattern in ROLE_TERMS.values()
        if re.search(rf"\b(?:{pattern})\b", normalized)
    ] or list(ROLE_TERMS.values())
    role_matches = [
        match
        for pattern in selected_roles
        for match in re.finditer(rf"\b(?:{pattern})\b", cleaned, re.I)
    ]
    names = list(PERSON_NAME_PATTERN.finditer(cleaned))
    pairs = [
        (role, name, min(abs(role.end() - name.start()), abs(name.end() - role.start())))
        for role in role_matches
        for name in names
    ]
    if not pairs:
        return cleaned[:max_chars] + "..."

    role, name, distance = min(pairs, key=lambda pair: pair[2])
    if distance > 140:
        return cleaned[:max_chars] + "..."
    evidence_start = min(role.start(), name.start())
    evidence_end = max(role.end(), name.end())
    start = max(0, evidence_start - max_chars // 3)
    if start + max_chars < evidence_end:
        start = evidence_end - max_chars
    end = min(len(cleaned), start + max_chars)
    excerpt = cleaned[start:end]
    if start:
        excerpt = "..." + excerpt
    if end < len(cleaned):
        excerpt += "..."
    return excerpt


def retrieve_documents(
    query: str,
    top_k: int = 5,
    candidate_k: Optional[int] = None,
    filter_dict: Optional[Dict[str, Any]] = None,
    w_semantic: Optional[float] = None,
    w_lexical: Optional[float] = None,
    w_metadata: Optional[float] = None,
    hybrid_evidence: Optional[List[RetrievedChunk]] = None,
) -> List[RetrievedChunk]:
    """
    Hybrid Retrieval Pipeline:
    1. Semantic search in ChromaDB to retrieve candidate chunks.
    2. Keyword (lexical) scoring against chunk text and document name.
    3. Intent-aware metadata and entity/proximity scoring.
    4. Relevance and direct-answer evidence are scored separately.
    5. Return diversified top-k chunks, allowing extra same-document hits for direct evidence.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    normalized_query = normalize_query(clean_query)
    intent = detect_query_intent(clean_query)
    pool_size = candidate_k or settings.CANDIDATE_POOL_SIZE
    w_sem = w_semantic if w_semantic is not None else settings.HYBRID_W_SEMANTIC
    w_lex = w_lexical if w_lexical is not None else settings.HYBRID_W_LEXICAL
    w_meta = w_metadata if w_metadata is not None else settings.HYBRID_W_METADATA

    embedding_mgr = get_embedding_manager()
    vector_store = get_vector_store()
    if candidate_k is None and intent == "PERSON_ROLE":
        pool_size = max(pool_size, min(vector_store.count(), 200))

    # Step 1: Semantic Candidate Retrieval
    query_vector = embedding_mgr.embed_query(normalized_query)
    raw_results = vector_store.query(
        query_embedding=query_vector,
        top_k=pool_size,
        where=filter_dict
    )

    if not raw_results or not raw_results.get("ids") or not raw_results["ids"][0]:
        return []

    ids = raw_results["ids"][0]
    docs = raw_results["documents"][0]
    metas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    # Step 2: Extract query keywords
    query_terms = extract_meaningful_terms(normalized_query)

    scored_chunks: List[RetrievedChunk] = []

    for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
        # ChromaDB cosine distance: dist = 1 - cos_sim
        sem_score = max(0.0, min(1.0, 1.0 - dist))
        doc_name = meta.get("document", "Unknown")
        doc_type = meta.get("document_type", "Unknown")
        ay = str(meta.get("academic_year", ""))

        # Step 3: Lexical & Metadata Scoring
        lex_score = compute_lexical_score(query_terms, text, doc_name)
        meta_score = compute_metadata_score(
            normalized_query, query_terms, doc_type, ay, doc_name, intent=intent, chunk_text=text
        )
        role_pairs = _role_evidence_matches(normalized_query, text) if intent == "PERSON_ROLE" else []
        entity_score, proximity_score = compute_entity_proximity_scores(normalized_query, intent, text)
        department_score = max((pair["department_score"] for pair in role_pairs), default=0.0)
        role_qualifier_score = max((pair["qualifier_score"] for pair in role_pairs), default=0.0)
        temporal_score = compute_temporal_score(normalized_query, ay)
        source_authority_score = compute_source_authority_score(intent, doc_type, text)

        base_relevance = (w_sem * sem_score) + (w_lex * lex_score) + (w_meta * meta_score)
        relevance_score = base_relevance
        if intent == "PERSON_ROLE":
            final_score = (
                settings.ROLE_W_BASE_RELEVANCE * base_relevance
                + settings.ROLE_W_ENTITY * entity_score
                + settings.ROLE_W_PROXIMITY * proximity_score
                + settings.ROLE_W_DEPARTMENT * department_score
                + settings.ROLE_W_QUALIFIER * role_qualifier_score
                + settings.ROLE_W_TEMPORAL * temporal_score
                + settings.ROLE_W_SOURCE_AUTHORITY * source_authority_score
            )
            relevance_score = final_score
        elif re.search(r"\b(current|currently|now|latest|most recent)\b", normalized_query, re.I) or _document_academic_years(normalized_query):
            final_score = (
                settings.TEMPORAL_W_BASE_RELEVANCE * base_relevance
                + settings.TEMPORAL_W_MATCH * temporal_score
            )
            relevance_score = final_score
        else:
            final_score = base_relevance
        answerability_score = compute_answerability_score(
            intent=intent,
            query_terms=query_terms,
            text=text,
            doc_type=doc_type,
            doc_name=doc_name,
            lexical_score=lex_score,
            entity_score=entity_score,
            proximity_score=proximity_score,
            query=normalized_query,
            academic_year=ay,
        )

        chunk = RetrievedChunk(
            chunk_id=chunk_id,
            document=doc_name,
            document_type=doc_type,
            page=int(meta.get("page", 0)),
            academic_year=ay,
            chunk_number_on_page=int(meta.get("chunk_number_on_page", 1)),
            text=text,
            distance=round(dist, 4),
            semantic_score=round(sem_score, 4),
            lexical_score=round(lex_score, 4),
            entity_score=round(entity_score, 4),
            proximity_score=round(proximity_score, 4),
            department_score=round(department_score, 4),
            role_qualifier_score=round(role_qualifier_score, 4),
            temporal_score=round(temporal_score, 4),
            source_authority_score=round(source_authority_score, 4),
            matched_name=role_pairs[0]["name"] if role_pairs else "",
            matched_role=role_pairs[0]["role"] if role_pairs else "",
            metadata_score=round(meta_score, 4),
            answerability_score=answerability_score,
            relevance_score=round(relevance_score, 4),
            final_score=round(final_score, 4),
            similarity_score=round(final_score, 4)
        )
        scored_chunks.append(chunk)

    # Step 5: Limit hybrid-ranked candidates before pairwise reranking.
    scored_chunks.sort(key=lambda c: c.final_score, reverse=True)
    if hybrid_evidence is not None:
        hybrid_evidence.extend(_select_diverse_chunks(scored_chunks, top_k))
    cross_candidate_k = min(settings.CANDIDATE_POOL_SIZE, len(scored_chunks))
    rerank_candidates = scored_chunks[:cross_candidate_k]
    if settings.CROSS_ENCODER_ENABLED and rerank_candidates:
        try:
            reranked_chunks = get_cross_encoder_reranker().rerank(
                normalized_query,
                rerank_candidates,
            )
        except Exception as error:
            global _reranker_warning_emitted
            if not settings.CROSS_ENCODER_ALLOW_HYBRID_FALLBACK:
                raise RuntimeError("Cross-Encoder reranking failed and hybrid fallback is disabled.") from error
            if not _reranker_warning_emitted:
                logger.warning("Cross-Encoder unavailable; retaining hybrid order: %s", error)
                _reranker_warning_emitted = True
            reranked_chunks = rerank_candidates
    else:
        reranked_chunks = rerank_candidates

    return _select_diverse_chunks(reranked_chunks, top_k)

def retrieve_with_evaluation(
    query: str,
    top_k: int = 5,
    filter_dict: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None
) -> RetrievalResponse:
    """
    Retrieves documents using hybrid scoring and evaluates whether context is sufficient.
    """
    cutoff = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
    normalized_query = normalize_query(query.strip())
    intent = detect_query_intent(query)
    conflict_review_chunks: List[RetrievedChunk] = []
    chunks = retrieve_documents(
        query=query,
        top_k=top_k,
        filter_dict=filter_dict,
        hybrid_evidence=conflict_review_chunks if intent == "PERSON_ROLE" else None,
    )
    if not conflict_review_chunks:
        conflict_review_chunks = chunks

    if not chunks:
        return RetrievalResponse(
            query=query,
            normalized_query=normalized_query,
            intent=intent,
            top_k=top_k,
            results=[],
            answerability_score=0.0,
            relevance_score=0.0,
            top_composite_score=0.0,
            temporal_score=0.0,
            cross_encoder_applied=False,
            reranker_note="No candidates available for reranking.",
            status="INSUFFICIENT_KNOWLEDGE",
            has_sufficient_context=False,
            confidence_note="No matching documents found in the MHSSCE knowledge base."
        )

    best_score = chunks[0].final_score
    best_relevance = max(chunk.relevance_score for chunk in chunks)
    answerability_score = max(chunk.answerability_score for chunk in chunks)
    requires_personal_context = intent == "SCHOLARSHIP" and _is_personal_eligibility_query(query) and not _has_personal_eligibility_details(query)
    is_sufficient = best_relevance >= cutoff and answerability_score >= 0.5 and not requires_personal_context
    role_evidence = []
    if intent == "PERSON_ROLE":
        seen_records = set()
        for chunk in [*chunks, *conflict_review_chunks]:
            if not chunk.matched_name or not chunk.matched_role:
                continue
            key = (chunk.matched_name.lower(), chunk.matched_role.lower(), chunk.document, chunk.page, chunk.academic_year)
            if key in seen_records:
                continue
            seen_records.add(key)
            role_evidence.append({
                "name": chunk.matched_name,
                "role": chunk.matched_role,
                "academic_year": chunk.academic_year,
                "document": chunk.document,
                "page": chunk.page,
            })
    role_names = {record["name"].lower() for record in role_evidence}
    has_conflicting_records = len(role_names) > 1
    status = (
        "CONFLICTING_EVIDENCE" if has_conflicting_records
        else "INSUFFICIENT_KNOWLEDGE" if not is_sufficient
        else "SUFFICIENT"
    )

    if has_conflicting_records:
        note = "Retrieved sources contain distinct explicit role/name records. Review the dated sources; no person has been selected as the winner."
    elif requires_personal_context:
        note = "Personal scholarship eligibility cannot be assessed without the relevant student eligibility details."
    elif is_sufficient:
        note = f"Answerability {answerability_score:.4f}; relevance {best_relevance:.4f}. Direct supporting evidence was retrieved."
    else:
        note = f"Answerability {answerability_score:.4f}; relevance {best_relevance:.4f}. Low confidence / insufficient knowledge base coverage."

    return RetrievalResponse(
        query=query,
        normalized_query=normalized_query,
        intent=intent,
        top_k=top_k,
        results=chunks,
        answerability_score=round(answerability_score, 4),
        relevance_score=round(best_relevance, 4),
        top_composite_score=round(best_score, 4),
        temporal_score=round(max(chunk.temporal_score for chunk in chunks), 4),
        cross_encoder_applied=any(chunk.cross_encoder_score is not None for chunk in chunks),
        reranker_note=(
            "Cross-Encoder reranking applied."
            if any(chunk.cross_encoder_score is not None for chunk in chunks)
            else "Cross-Encoder unavailable; hybrid fallback applied."
            if settings.CROSS_ENCODER_ENABLED
            else "Cross-Encoder disabled; hybrid ranking used."
        ),
        status=status,
        conflicting_evidence=has_conflicting_records,
        role_evidence=role_evidence,
        requires_personal_context=requires_personal_context,
        has_sufficient_context=is_sufficient,
        confidence_note=note
    )
