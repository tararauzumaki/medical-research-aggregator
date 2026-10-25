import html
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

HISTORY_FILE = Path("sent_papers.json")
OPENALEX_URL = "https://api.openalex.org/works"

LOOKBACK_DAYS = 30
PER_PAGE = 100
MAX_PAGES = 3
MAX_HISTORY = 1000
MESSAGE_DELAY = 1

# STRICT MEDICAL FIELDS ONLY (Excludes biology, agriculture, physics, engineering, etc.)
MEDICAL_FIELDS = {
    "Medicine",
    "Nursing",
    "Health Professions",
    "Dentistry",
}

VALIDATION_TERMS = [
    "bangladesh",
    "bangladeshi",
    "dhaka",
    "chattogram",
    "chittagong",
    "sundarbans",
    "cox's bazar",
    "padma river",
    "jamuna river",
    "meghna river",
    "sylhet",
    "rajshahi",
    "khulna",
    "barisal",
    "rangpur",
    "mymensingh",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC %(levelname)s: %(message)s",
)

session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
    respect_retry_after_header=True,
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)


def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Unable to read {HISTORY_FILE}: {exc}") from exc

    if not isinstance(data, list):
        raise RuntimeError(f"{HISTORY_FILE} must contain a JSON list.")

    return list(dict.fromkeys(item for item in data if isinstance(item, str)))[-MAX_HISTORY:]


def save_history(sent_ids):
    temporary_file = HISTORY_FILE.with_suffix(".tmp")
    recent_ids = list(dict.fromkeys(sent_ids))[-MAX_HISTORY:]
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(recent_ids, file, ensure_ascii=False, indent=2)
    temporary_file.replace(HISTORY_FILE)


def get_abstract_text(paper):
    inverted_index = paper.get("abstract_inverted_index") or {}
    positioned_words = []
    for word, positions in inverted_index.items():
        for position in positions:
            positioned_words.append((position, word))
    positioned_words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned_words)


def get_searchable_text(paper):
    title = paper.get("title") or ""
    abstract = get_abstract_text(paper)
    keywords = paper.get("keywords") or []
    keyword_text = " ".join(
        item.get("display_name", "") for item in keywords if isinstance(item, dict)
    )
    return f"{title} {abstract} {keyword_text}".casefold()


def is_locally_relevant(paper):
    """Checks if the paper content mentions Bangladesh or regional keywords."""
    text = get_searchable_text(paper)
    for term in VALIDATION_TERMS:
        pattern = r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)"
        if re.search(pattern, text):
            return True
    return False


def has_bangladeshi_author(paper):
    """Checks if any author is affiliated with an institution in Bangladesh."""
    authorships = paper.get("authorships") or []
    for authorship in authorships:
        institutions = authorship.get("institutions") or []
        for inst in institutions:
            if inst.get("country_code") == "BD":
                return True
            if "bangladesh" in (inst.get("display_name") or "").casefold():
                return True
    return False


def is_medical_paper(paper):
    """
    STRICT GATEKEEPER: Ensures the paper belongs exclusively to medical/health fields.
    Blocks computer science, physics, pure biology, agriculture, etc., even if 
    they originate from Bangladesh or mention Bangladesh.
    """
    topics = paper.get("topics") or []
    if not topics:
        return False
    for topic in topics:
        field = topic.get("field") or {}
        field_name = field.get("display_name", "")
        if field_name in MEDICAL_FIELDS:
            return True
    return False


def fetch_openalex_papers():
    start_date = (
        datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    ).date().isoformat()

    unique_papers = {}

    # ------------------------------------------------------------------
    # STRATEGIC STREAM 1: Medical Research ABOUT Bangladesh (Keyword Search)
    # ------------------------------------------------------------------
    for term in ["Bangladesh"]:
        logging.info("Querying OpenAlex keyword search for: %s", term)
        for page in range(1, MAX_PAGES + 1):
            params = {
                "search": term,
                "filter": f"from_publication_date:{start_date},type:article|report",
                "per_page": PER_PAGE,
                "page": page,
                "sort": "publication_date:desc",
            }

            try:
                response = session.get(
                    OPENALEX_URL,
                    params=params,
                    headers={"User-Agent": "MedicalResearchAggregator/1.0 (mailto:admin@medical-research-aggregator.org)"},
                    timeout=25,
                )
                response.raise_for_status()
                payload = response.json()
                batch = payload.get("results", [])

                if not batch:
                    break

                for paper in batch:
                    paper_id = paper.get("id")
                    if not paper_id or paper_id in unique_papers:
                        continue
                    # Must be strictly medical
                    if not is_medical_paper(paper):
                        continue
                    # Must be locally relevant (about Bangladesh or local regions)
                    if not (is_locally_relevant(paper) or has_bangladeshi_author(paper)):
                        continue
                    unique_papers[paper_id] = paper

                if len(batch) < PER_PAGE:
                    break

                time.sleep(0.2)

            except Exception as e:
                logging.error("Error fetching OpenAlex search data on page %d: %s", page, e)
                break

    # ------------------------------------------------------------------
    # STRATEGIC STREAM 2: Medical Research BY Bangladeshi Researchers
    # (Affiliation filter country_code:bd + Strict Medical Gatekeeper)
    # ------------------------------------------------------------------
    logging.info("Querying OpenAlex by Bangladeshi author affiliation (country_code:bd)")
    for page in range(1, MAX_PAGES + 1):
        params = {
            "filter": f"from_publication_date:{start_date},type:article|report,authorships.institutions.country_code:bd",
            "per_page": PER_PAGE,
            "page": page,
            "sort": "publication_date:desc",
        }

        try:
            response = session.get(
                OPENALEX_URL,
                params=params,
                headers={"User-Agent": "MedicalResearchAggregator/1.0 (mailto:admin@medical-research-aggregator.org)"},
                timeout=25,
            )
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("results", [])

            if not batch:
                break

            for paper in batch:
                paper_id = paper.get("id")
                if not paper_id or paper_id in unique_papers:
                    continue
                # STRICT FILTER: Discards non-medical papers (e.g., CS/Physics) even from BD authors
                if not is_medical_paper(paper):
                    continue
                unique_papers[paper_id] = paper

            if len(batch) < PER_PAGE:
                break

            time.sleep(0.2)

        except Exception as e:
            logging.error("Error fetching OpenAlex affiliation data on page %d: %s", page, e)
            break

    return sorted(
        unique_papers.values(),
        key=lambda paper: (paper.get("publication_date") or "", paper.get("id") or ""),
        reverse=True,
    )


def resolve_paper_url(paper):
    primary_location = paper.get("primary_location") or {}
    landing_url = primary_location.get("landing_page_url")
    if landing_url and landing_url.startswith(("https://", "http://")):
        return landing_url
    doi = paper.get("doi")
    if doi:
        if doi.startswith(("https://", "http://")):
            return doi
        return "https://doi.org/" + doi.removeprefix("doi:")
    return paper.get("id", "")


def get_source_name(paper):
    location = paper.get("primary_location") or {}
    source = location.get("source") or {}
    return source.get("display_name") or "Publication source unavailable"


def format_message(paper):
    title = paper.get("title") or "Title unavailable"
    pub_date = paper.get("publication_date") or "Unknown"
    journal = get_source_name(paper)
    url = resolve_paper_url(paper)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    safe_title = html.escape(title)
    safe_journal = html.escape(journal)
    safe_url = html.escape(url, quote=True)
    linked_title = f'<a href="{safe_url}">{safe_title}</a>' if url else safe_title

    return (
        "<b>New Medical Research Alert!</b>\n"
        f"Timestamp: {timestamp}\n\n"
        f"<b>Title:</b> {linked_title}\n\n"
        f"<b>Journal:</b> {safe_journal}\n"
        f"<b>Publishing Date:</b> {html.escape(pub_date)}\n\n"
        f"<b>Link:</b> {safe_url}"
    )


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    response = session.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
        },
        timeout=15,
    )
    response.raise_for_status()


def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing Telegram credentials.")

    sent_ids = load_history()
    sent_set = set(sent_ids)
    papers = fetch_openalex_papers()

    logging.info("Found %d unique validated medical research papers.", len(papers))
    new_count = 0

    for paper in papers:
        paper_id = paper["id"]
        if paper_id in sent_set:
            continue

        message = format_message(paper)
        send_telegram_message(message)

        sent_ids.append(paper_id)
        sent_set.add(paper_id)
        save_history(sent_ids)
        new_count += 1

        logging.info("Delivered: %s", paper.get("title", "Untitled"))
        time.sleep(MESSAGE_DELAY)

    logging.info("Completed. Delivered %d new papers.", new_count)


if __name__ == "__main__":
    main()
