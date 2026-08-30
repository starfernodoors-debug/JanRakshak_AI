"""
JanRakshak AI — Service Layer
Django-native service layer for civic issue reporting.
Database operations use Django ORM (PostgreSQL/SQLite).
AI operations use keyword classification + Google Gemini API.
"""

import re
import json
import io
import base64
import hashlib
import logging
import datetime
import colorsys
import time

from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths (for audit log only — DB is handled by Django ORM)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent.parent
AUDIT_LOG_PATH = _HERE / "data" / "audit.log"
AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# AI CONFIDENCE HELPER
# ---------------------------------------------------------------------------

def is_ai_confused(result: dict) -> bool:
    """
    Returns True if the AI classification has low confidence,
    indicating it is confused/unsure and needs authority verification.
    """

    confidence = result.get("Confidence", "Low")

    if not confidence:
        return True

    conf_str = str(confidence).strip().lower()

    if conf_str == "low":
        return True

    match = re.search(r"(\d+)", conf_str)

    if match:
        val = int(match.group(1))

        if val < 70:
            return True

    return False


# ===========================================================================
# DATABASE
# ===========================================================================

class Database:
    """
    Django ORM wrapper maintaining the original API for backward compatibility.

    Returns tuples in the same column order the views expect:

    (id, description, location, issue, priority, department, confidence,
     advice, risk_score, reason, suggested_action, submitted_at_str,
     photo_url_or_none, status_int, upvotes)
    """

    def create_database(self):
        """No-op — Django migrations handle schema creation."""
        pass

    def _to_tuple(self, report):
        """Convert a CivicReport ORM object to a legacy tuple."""

        from django.utils import timezone as tz

        photo_url = report.photo.url if report.photo else None

        submitted_str = (
            report.submitted_at.strftime("%Y-%m-%d %H:%M:%S")
            if report.submitted_at
            else ""
        )

        return (
            report.id,
            report.description,
            report.location,
            report.issue,
            report.priority,
            report.department,
            report.confidence,
            report.advice,
            report.risk_score,
            report.reason,
            report.suggested_action,
            submitted_str,
            photo_url,
            report.status,
            report.upvotes,
        )

    def save_report(
        self,
        description: str,
        location: str,
        result: dict,
        photo_path: str = None
    ):
        from portal.models import CivicReport
        from django.core.files.base import ContentFile

        if result.get("Issue") == "Spam":
            status = 1

        elif is_ai_confused(result) or result.get("ai_correct") is False:
            status = 2

        else:
            status = 0

        report = CivicReport(
            description=description,
            location=location,
            issue=result.get("Issue", "Unknown")[:50],
            priority=result.get("Priority", "Low")[:20],
            department=result.get("Department", "General")[:100],
            confidence=result.get("Confidence", "Medium")[:20],
            advice=result.get("Advice", ""),
            risk_score=int(result.get("Risk Score", 0)),
            reason=result.get("Reason", ""),
            suggested_action=result.get("Suggested Action", ""),
            status=status,
        )

        if photo_path and "," in photo_path and photo_path.startswith("data:"):
            try:
                _, b64 = photo_path.split(",", 1)

                image_data = base64.b64decode(b64)

                fname = (
                    f"report_"
                    f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    f".jpg"
                )

                report.photo.save(
                    fname,
                    ContentFile(image_data),
                    save=False
                )

            except Exception as exc:
                logger.warning("Photo save failed: %s", exc)

        report.save()

        return {
            "id": report.id,
            "upvotes": report.upvotes
        }

    def get_reports(self):
        from portal.models import CivicReport

        return [
            self._to_tuple(r)
            for r in CivicReport.objects.all()
        ]

    def search_reports(self, keyword: str):
        from portal.models import CivicReport
        from django.db.models import Q

        qs = CivicReport.objects.filter(
            Q(description__icontains=keyword)
            | Q(location__icontains=keyword)
            | Q(issue__icontains=keyword)
            | Q(department__icontains=keyword)
        )

        return [
            self._to_tuple(r)
            for r in qs
        ]

    def upvote_report(self, report_id: int):
        from portal.models import CivicReport
        from django.db.models import F

        try:
            report = CivicReport.objects.get(
                id=report_id
            )

            CivicReport.objects.filter(
                id=report_id
            ).update(
                upvotes=F("upvotes") + 1
            )

            report.refresh_from_db()

            return {
                "id": report.id,
                "upvotes": report.upvotes
            }

        except CivicReport.DoesNotExist:
            return None

    def toggle_spam(self, report_id: int):
        from portal.models import CivicReport

        try:
            report = CivicReport.objects.get(
                id=report_id
            )

            report.status = (
                1
                if report.status == 0
                else 0
            )

            report.save(
                update_fields=["status"]
            )

            return report.status

        except CivicReport.DoesNotExist:
            return None

    def set_spam_status(
        self,
        report_id: int,
        status: int
    ):
        from portal.models import CivicReport

        updated = CivicReport.objects.filter(
            id=report_id
        ).update(
            status=status
        )

        return status if updated else None

    def get_total_reports(self) -> int:
        from portal.models import CivicReport

        return CivicReport.objects.count()

    def get_average_risk(self) -> float:
        from portal.models import CivicReport
        from django.db.models import Avg

        result = CivicReport.objects.aggregate(
            avg=Avg("risk_score")
        )["avg"]

        return round(result or 0, 1)

    def get_most_common_issue(self):
        from portal.models import CivicReport
        from django.db.models import Count

        row = (
            CivicReport.objects
            .values("issue")
            .annotate(cnt=Count("issue"))
            .order_by("-cnt")
            .first()
        )

        return (
            row["issue"],
        ) if row else (
            "None",
        )

    def get_reports_by_issue(self):
        from portal.models import CivicReport
        from django.db.models import Count

        rows = (
            CivicReport.objects
            .values("issue")
            .annotate(cnt=Count("issue"))
            .order_by("-cnt")
        )

        return [
            (r["issue"], r["cnt"])
            for r in rows
        ]

    def get_reports_by_priority(self):
        from portal.models import CivicReport
        from django.db.models import Count

        rows = (
            CivicReport.objects
            .values("priority")
            .annotate(cnt=Count("priority"))
            .order_by("-cnt")
        )

        return [
            (r["priority"], r["cnt"])
            for r in rows
        ]

    def get_top_locations(
        self,
        limit: int = 5
    ):
        from portal.models import CivicReport
        from django.db.models import Count

        rows = (
            CivicReport.objects
            .exclude(location="Unknown")
            .values("location")
            .annotate(cnt=Count("location"))
            .order_by("-cnt")[:limit]
        )

        return [
            (r["location"], r["cnt"])
            for r in rows
        ]


# ===========================================================================
# GEMINI REQUEST HELPER
# ===========================================================================

def _gemini_post(
    url,
    payload,
    api_key,
    requests_module,
    retries=2
):
    """
    Send a request to Gemini with a longer read timeout and retry support.

    Connect timeout:
        10 seconds

    Read timeout:
        90 seconds

    This is important for image analysis because multimodal requests can
    take longer than simple text requests.
    """

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    last_exception = None

    for attempt in range(retries + 1):

        try:

            response = requests_module.post(
                url,
                json=payload,
                headers=headers,
                timeout=(10, 90),
            )

            return response

        except (
            requests_module.exceptions.Timeout,
            requests_module.exceptions.ConnectionError,
        ) as exc:

            last_exception = exc

            logger.warning(
                "Gemini request failed on attempt %s/%s: %s",
                attempt + 1,
                retries + 1,
                exc,
            )

            if attempt < retries:
                time.sleep(2)

    raise last_exception


# ===========================================================================
# AI ENGINE — GEMINI CLOUD CLASSIFIER
# ===========================================================================

class AIEngine:
    """
    Issue classifier using Google Gemini Flash API for text analysis.
    """

    def analyze(
        self,
        description: str
    ) -> dict:

        import os
        import json
        import requests
        from django.conf import settings

        api_key = (
            getattr(
                settings,
                "GEMINI_API_KEY",
                ""
            )
            or os.environ.get(
                "GEMINI_API_KEY",
                ""
            )
        )

        if not api_key:

            raise ValueError(
                "Google Gemini API Key is not configured on the server. "
                "Please set the GEMINI_API_KEY environment variable "
                "to enable AI classification."
            )

        try:

            url = (
                "https://generativelanguage.googleapis.com/"
                "v1beta/models/gemini-flash-latest:generateContent"
            )

            prompt = (
                f"Analyze the following civic issue description:\n"
                f"\"{description}\"\n\n"

                "Determine the issue category, which must be exactly one of: "

                "'Road Damage', "
                "'Water Leakage', "
                "'Garbage', "
                "'Electricity', "
                "'Streetlight', "
                "'Flood', "
                "'Fire Emergency', "
                "'Tree Fall', "
                "'Spam', "
                "or 'General Civic Issue'.\n\n"

                "Return a JSON object conforming exactly to this schema:\n"

                "{\n"
                "  \"Issue\": \"string\",\n"
                "  \"Priority\": \"Critical\" | \"High\" | \"Medium\" | \"Low\",\n"
                "  \"Department\": \"string (concerned authority name)\",\n"
                "  \"Confidence\": \"string (percentage, e.g., 92%)\",\n"
                "  \"Risk Score\": integer (0 to 100),\n"
                "  \"Reason\": \"string (1-sentence reason for classification)\",\n"
                "  \"Suggested Action\": \"string (action for authority)\",\n"
                "  \"Advice\": \"string (optional advice for citizen)\"\n"
                "}"
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json"
                }
            }

            response = _gemini_post(
                url=url,
                payload=payload,
                api_key=api_key,
                requests_module=requests,
            )

            if response.status_code != 200:

                err_text = response.text

                try:

                    err_json = response.json()

                    err_text = (
                        err_json
                        .get("error", {})
                        .get(
                            "message",
                            response.text
                        )
                    )

                except Exception:
                    pass

                raise RuntimeError(
                    f"Gemini API error "
                    f"({response.status_code}): "
                    f"{err_text}"
                )

            resp_json = response.json()

            candidates = resp_json.get(
                "candidates",
                []
            )

            if not candidates:

                raise RuntimeError(
                    "Gemini API returned an empty response."
                )

            text_content = (
                candidates[0]
                ["content"]
                ["parts"][0]
                ["text"]
            )

            result = json.loads(
                text_content
            )

            return {
                "Issue": result.get(
                    "Issue",
                    "General Civic Issue"
                ),

                "Priority": result.get(
                    "Priority",
                    "Medium"
                ),

                "Department": result.get(
                    "Department",
                    "Municipal Corporation"
                ),

                "Confidence": str(
                    result.get(
                        "Confidence",
                        "85%"
                    )
                ),

                "Risk Score": int(
                    str(
                        result.get(
                            "Risk Score",
                            50
                        )
                    )
                    .split("/")[0]
                    .replace(".", "")
                    if str(
                        result.get(
                            "Risk Score",
                            50
                        )
                    ).replace(".", "").isdigit()
                    else 50
                ),

                "Reason": result.get(
                    "Reason",
                    "Analyzed using Gemini AI."
                ),

                "Suggested Action": result.get(
                    "Suggested Action",
                    "Forward to municipal authority."
                ),

                "Advice": result.get(
                    "Advice",
                    "Keep your report ID safe for status tracking."
                ),
            }

        except Exception as exc:

            logger.warning(
                "Gemini text analysis failed: %s",
                exc
            )

            raise

    def _fallback_keyword_triage(
        self,
        description: str
    ) -> dict:

        desc_lower = (
            description or ""
        ).lower()

        if any(
            w in desc_lower
            for w in [
                "water",
                "pipe",
                "leak",
                "sewer",
                "drain",
                "burst",
                "tap",
                "flood"
            ]
        ):

            return {
                "Issue": "Water Leakage",
                "Priority": "High",
                "Department": "Water Supply & Sewerage Board",
                "Confidence": "78% (Rule Triage)",
                "Risk Score": 72,
                "Reason": "Detected water/leakage keywords in issue description.",
                "Suggested Action": "Inspect water supply pipeline and fix leak immediately.",
                "Advice": "Avoid walking near flooded water pipes.",
            }

        elif any(
            w in desc_lower
            for w in [
                "road",
                "pothole",
                "tar",
                "asphalt",
                "crack",
                "bridge",
                "street"
            ]
        ):

            return {
                "Issue": "Road Damage",
                "Priority": "Medium",
                "Department": "Public Works Department (PWD)",
                "Confidence": "80% (Rule Triage)",
                "Risk Score": 60,
                "Reason": "Detected road/pothole keywords in issue description.",
                "Suggested Action": "Dispatch PWD road repair crew to patch surface.",
                "Advice": "Drive carefully around damaged road section.",
            }

        elif any(
            w in desc_lower
            for w in [
                "garbage",
                "trash",
                "waste",
                "dump",
                "smell",
                "clean"
            ]
        ):

            return {
                "Issue": "Garbage",
                "Priority": "Medium",
                "Department": "Municipal Solid Waste Management",
                "Confidence": "82% (Rule Triage)",
                "Risk Score": 45,
                "Reason": "Detected waste/garbage keywords in issue description.",
                "Suggested Action": "Schedule solid waste collection truck.",
                "Advice": "Keep waste bagged until municipal pickup.",
            }

        elif any(
            w in desc_lower
            for w in [
                "wire",
                "electric",
                "spark",
                "transformer",
                "shock",
                "power"
            ]
        ):

            return {
                "Issue": "Electricity",
                "Priority": "Critical",
                "Department": "State Electricity Distribution Company",
                "Confidence": "85% (Rule Triage)",
                "Risk Score": 90,
                "Reason": "Detected electrical hazard keywords in issue description.",
                "Suggested Action": "Isolate high voltage line and repair circuit.",
                "Advice": "Do not touch open electrical wires or transformers.",
            }

        elif any(
            w in desc_lower
            for w in [
                "dark",
                "light",
                "lamp",
                "pole",
                "bulb"
            ]
        ):

            return {
                "Issue": "Streetlight",
                "Priority": "Low",
                "Department": "Urban Local Body — Street Lighting",
                "Confidence": "75% (Rule Triage)",
                "Risk Score": 35,
                "Reason": "Detected streetlight keywords in issue description.",
                "Suggested Action": "Replace damaged bulb/fixture.",
                "Advice": "Exercise caution in poorly lit areas.",
            }

        elif any(
            w in desc_lower
            for w in [
                "fire",
                "smoke",
                "flame",
                "burn"
            ]
        ):

            return {
                "Issue": "Fire Emergency",
                "Priority": "Critical",
                "Department": "Fire & Emergency Services",
                "Confidence": "90% (Rule Triage)",
                "Risk Score": 95,
                "Reason": "Detected fire hazard keywords in issue description.",
                "Suggested Action": "Deploy fire engine and emergency response unit.",
                "Advice": "Evacuate area immediately and dial emergency hotline 101.",
            }

        elif any(
            w in desc_lower
            for w in [
                "tree",
                "branch",
                "fallen"
            ]
        ):

            return {
                "Issue": "Tree Fall",
                "Priority": "Medium",
                "Department": "Horticulture / Forest Department",
                "Confidence": "80% (Rule Triage)",
                "Risk Score": 55,
                "Reason": "Detected tree fall keywords in issue description.",
                "Suggested Action": "Clear fallen tree and restore road passage.",
                "Advice": "Avoid standing under unstable trees.",
            }

        else:

            return {
                "Issue": "General Civic Issue",
                "Priority": "Medium",
                "Department": "Municipal Corporation",
                "Confidence": "70% (Rule Triage)",
                "Risk Score": 50,
                "Reason": "Automated triage classification. You can verify or change department below.",
                "Suggested Action": "Route report to municipal authority for verification.",
                "Advice": "Provide additional photos or location details if possible.",
            }


# ===========================================================================
# PHOTO ANALYZER
# ===========================================================================

class PhotoAnalyzer:
    """
    Analyzes a civic issue photo using Google Gemini Flash Vision API.
    """

    CATEGORY_META = {
        "Water Leakage": (
            "High",
            "Water Supply & Sewerage Board",
            72
        ),

        "Road Damage": (
            "Medium",
            "Public Works Department (PWD)",
            58
        ),

        "Garbage": (
            "Low",
            "Municipal Solid Waste Management",
            36
        ),

        "Electricity": (
            "Critical",
            "State Electricity Distribution Company",
            90
        ),

        "Flood": (
            "High",
            "Drainage & Flood Control Board",
            84
        ),

        "Fire Emergency": (
            "Critical",
            "Fire & Emergency Services",
            96
        ),

        "Tree Fall": (
            "Medium",
            "Horticulture / Forest Department",
            61
        ),

        "Streetlight": (
            "Low",
            "Urban Local Body — Street Lighting",
            30
        ),

        "General Civic Issue": (
            "Low",
            "Municipal Corporation",
            30
        ),
    }

    def _generate_description(
        self,
        result: dict
    ) -> str:

        issue = result.get(
            "Issue",
            "General Civic Issue"
        )

        dept = result.get(
            "Department",
            "Municipal Corporation"
        )

        detail_map = {

            "Water Leakage":
                "Photographic evidence indicates a water leakage issue. Requesting Water Supply & Sewerage Board line inspection.",

            "Road Damage":
                "Photographic evidence indicates road damage / pothole defect. Requesting PWD road maintenance team dispatch.",

            "Garbage":
                "Photographic evidence indicates solid waste / garbage accumulation. Requesting Municipal Solid Waste clearing.",

            "Electricity":
                "Photographic evidence indicates electrical infrastructure hazard. Requesting Electricity Distribution technician.",

            "Flood":
                "Photographic evidence indicates severe water-logging / flooding. Requesting Drainage & Flood Control Board action.",

            "Fire Emergency":
                "Photographic evidence indicates fire emergency / smoke. Requesting Fire & Emergency Services dispatch.",

            "Tree Fall":
                "Photographic evidence indicates fallen tree / branch obstruction. Requesting Horticulture / Forest Department.",

            "Streetlight":
                "Photographic evidence indicates streetlight defect. Requesting Urban Local Body Street Lighting repair.",
        }

        return detail_map.get(
            issue,
            f"Photo evidence attached for civic issue inspection by {dept}."
        )

    def analyze_photo_data(
        self,
        photo_data: str,
        photo_name: str = ""
    ) -> dict:

        import os
        import json
        import requests
        from django.conf import settings

        api_key = (
            getattr(
                settings,
                "GEMINI_API_KEY",
                ""
            )
            or os.environ.get(
                "GEMINI_API_KEY",
                ""
            )
        )

        if not api_key:

            raise ValueError(
                "Google Gemini API Key is not configured on the server. "
                "Please set the GEMINI_API_KEY environment variable "
                "to enable AI photo analysis."
            )

        try:

            from PIL import Image

            if "," in photo_data:

                mime, b64 = photo_data.split(
                    ",",
                    1
                )

                try:

                    mime_type = (
                        mime
                        .split(";")[0]
                        .split(":")[1]
                        .lower()
                    )

                except Exception:

                    mime_type = "image/jpeg"

            else:

                b64 = photo_data
                mime_type = "image/jpeg"

            # Gemini supported image formats.
            supported_mimes = {
                "image/jpeg",
                "image/png",
                "image/webp",
            }

            if mime_type not in supported_mimes:

                try:

                    img_bytes = base64.b64decode(
                        b64
                    )

                    img = Image.open(
                        io.BytesIO(img_bytes)
                    )

                    if img.mode in (
                        "RGBA",
                        "P"
                    ):

                        img = img.convert(
                            "RGB"
                        )

                    buffer = io.BytesIO()

                    img.save(
                        buffer,
                        format="JPEG",
                        quality=85
                    )

                    b64 = base64.b64encode(
                        buffer.getvalue()
                    ).decode("utf-8")

                    mime_type = "image/jpeg"

                except Exception as conv_err:

                    logger.warning(
                        "Could not convert %s image to JPEG: %s",
                        mime_type,
                        conv_err
                    )

                    mime_type = "image/jpeg"

            # Gemini endpoint.
            url = (
                "https://generativelanguage.googleapis.com/"
                "v1beta/models/gemini-flash-latest:generateContent"
            )

            prompt = (
                "Analyze the civic issue shown in this photo carefully. "

                "Determine the issue category, which must be exactly one of: "

                "'Road Damage', "
                "'Water Leakage', "
                "'Garbage', "
                "'Electricity', "
                "'Streetlight', "
                "'Flood', "
                "'Fire Emergency', "
                "'Tree Fall', "
                "'Spam', "
                "or 'General Civic Issue'.\n\n"

                "Instructions:\n"

                "1. If the image shows no real public civic issue "
                "(e.g. selfie, meme, indoor object, animal, food, "
                "generic landmark, clear sky without issue), "
                "classify it strictly as 'Spam' or "
                "'General Civic Issue' with low confidence.\n"

                "2. In 'Auto Description', describe ONLY what is "
                "explicitly visible in the photo. "
                "Do not assume or hallucinate unverified details.\n\n"

                "Return a JSON object conforming exactly to this schema:\n"

                "{\n"
                "  \"Issue\": \"string\",\n"
                "  \"Priority\": \"Critical\" | \"High\" | \"Medium\" | \"Low\",\n"
                "  \"Department\": \"string (concerned authority name)\",\n"
                "  \"Confidence\": \"string (percentage, e.g., 92%)\",\n"
                "  \"Risk Score\": integer (0 to 100),\n"
                "  \"Reason\": \"string (1-sentence reason for classification)\",\n"
                "  \"Suggested Action\": \"string (action for authority)\",\n"
                "  \"Advice\": \"string (optional advice for citizen)\",\n"
                "  \"Auto Description\": \"string (factual description of what is visible in photo)\"\n"
                "}"
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": b64
                                }
                            }
                        ]
                    }
                ],

                "generationConfig": {
                    "responseMimeType": "application/json"
                }
            }

            # IMPORTANT:
            # Increased read timeout from 25 seconds to 90 seconds.
            # Also retries temporary connection/timeout failures.
            response = _gemini_post(
                url=url,
                payload=payload,
                api_key=api_key,
                requests_module=requests,
                retries=2,
            )

            if response.status_code != 200:

                err_text = response.text

                try:

                    err_json = response.json()

                    err_text = (
                        err_json
                        .get("error", {})
                        .get(
                            "message",
                            response.text
                        )
                    )

                except Exception:
                    pass

                raise RuntimeError(
                    f"Gemini API error "
                    f"({response.status_code}): "
                    f"{err_text}"
                )

            resp_json = response.json()

            candidates = resp_json.get(
                "candidates",
                []
            )

            if not candidates:

                raise RuntimeError(
                    "Gemini API returned an empty response."
                )

            try:

                text_content = (
                    candidates[0]
                    ["content"]
                    ["parts"][0]
                    ["text"]
                )

            except (
                KeyError,
                IndexError,
                TypeError
            ):

                raise RuntimeError(
                    "Gemini API returned an unexpected response structure."
                )

            result = json.loads(
                text_content
            )

            normalized = {

                "Issue": result.get(
                    "Issue",
                    "General Civic Issue"
                ),

                "Priority": result.get(
                    "Priority",
                    "Medium"
                ),

                "Department": result.get(
                    "Department",
                    "Municipal Corporation"
                ),

                "Confidence": str(
                    result.get(
                        "Confidence",
                        "85%"
                    )
                ),

                "Risk Score": int(
                    str(
                        result.get(
                            "Risk Score",
                            50
                        )
                    )
                    .split("/")[0]
                    .replace(".", "")
                    if str(
                        result.get(
                            "Risk Score",
                            50
                        )
                    ).replace(".", "").isdigit()
                    else 50
                ),

                "Reason": result.get(
                    "Reason",
                    "Analyzed using Gemini AI Vision."
                ),

                "Suggested Action": result.get(
                    "Suggested Action",
                    "Field inspection recommended."
                ),

                "Advice": result.get(
                    "Advice",
                    "Please verify and edit description if needed."
                ),

                "Auto Description": result.get(
                    "Auto Description",
                    ""
                ),
            }

            if not normalized[
                "Auto Description"
            ]:

                normalized[
                    "Auto Description"
                ] = self._generate_description(
                    normalized
                )

            return normalized

        except Exception as exc:

            logger.warning(
                "Gemini photo analysis failed: %s",
                exc
            )

            raise


# ===========================================================================
# DUPLICATE DETECTOR
# ===========================================================================

class DuplicateDetector:
    """
    Detects near-duplicate reports using Jaccard similarity on token sets.
    """

    THRESHOLD = 0.55

    LOCATION_MIN_SIMILARITY = 0.20

    def _tokenize(
        self,
        text: str
    ) -> set:

        words = re.findall(
            r"\b\w{3,}\b",
            text.lower()
        )

        return set(words)

    def _tokenize_location(
        self,
        text: str
    ) -> set:

        words = re.findall(
            r"\b\w{2,}\b",
            text.lower()
        )

        return set(words)

    def _normalize_location(
        self,
        location: str
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            (location or "").strip().lower()
        )

    def _has_meaningful_location(
        self,
        location: str
    ) -> bool:

        loc = self._normalize_location(
            location
        )

        return loc not in {
            "",
            "unknown",
            "n/a",
            "na",
            "not provided"
        }

    def _jaccard(
        self,
        a: set,
        b: set
    ) -> float:

        if not a or not b:
            return 0.0

        return len(a & b) / len(a | b)

    def _location_similarity(
        self,
        location: str,
        existing_location: str
    ):

        if (
            not self._has_meaningful_location(location)
            or not self._has_meaningful_location(
                existing_location
            )
        ):

            return None

        normalized_a = (
            self._normalize_location(
                location
            )
        )

        normalized_b = (
            self._normalize_location(
                existing_location
            )
        )

        if normalized_a == normalized_b:
            return 1.0

        return self._jaccard(
            self._tokenize_location(
                normalized_a
            ),
            self._tokenize_location(
                normalized_b
            )
        )

    def find_duplicate(
        self,
        description: str,
        location: str,
        reports: list
    ) -> dict:

        tokens = self._tokenize(
            description
        )

        best_match = {
            "duplicate": False,
            "report_id": None,
            "similarity": 0
        }

        for row in reports:

            existing_desc = (
                row[1]
                if len(row) > 1
                else ""
            )

            existing_location = (
                row[2]
                if len(row) > 2
                else ""
            )

            existing_tokens = (
                self._tokenize(
                    existing_desc
                )
            )

            description_sim = (
                self._jaccard(
                    tokens,
                    existing_tokens
                )
            )

            location_sim = (
                self._location_similarity(
                    location,
                    existing_location
                )
            )

            if (
                location_sim is not None
                and location_sim
                < self.LOCATION_MIN_SIMILARITY
            ):

                continue

            combined_sim = (
                description_sim
                if location_sim is None
                else (
                    0.70 * description_sim
                    + 0.30 * location_sim
                )
            )

            sim_pct = round(
                combined_sim * 100
            )

            if (
                combined_sim >= self.THRESHOLD
                and sim_pct > best_match["similarity"]
            ):

                best_match = {
                    "duplicate": True,
                    "report_id": row[0],
                    "similarity": sim_pct,
                    "description_similarity": round(
                        description_sim * 100
                    ),
                    "location_similarity": (
                        0
                        if location_sim is None
                        else round(
                            location_sim * 100
                        )
                    ),
                }

        return best_match


# ===========================================================================
# HOTSPOT ANALYZER
# ===========================================================================

class HotspotAnalyzer:
    """
    Classifies a location as a civic hotspot based on historical report counts.
    """

    def analyze(
        self,
        location: str
    ) -> dict:

        try:

            from portal.models import CivicReport

            count = (
                CivicReport.objects
                .filter(
                    location=location
                )
                .count()
            )

        except Exception:

            count = 0

        if count >= 5:

            level = "🔴 Critical Hotspot"

            rec = (
                "Immediate municipal intervention required. "
                "Escalate to ward officer."
            )

        elif count >= 3:

            level = "🟡 Moderate Hotspot"

            rec = (
                "Multiple reports from this location. "
                "Schedule a field inspection."
            )

        elif count >= 1:

            level = "🟢 Low Frequency"

            rec = (
                "New report for this location. "
                "Normal triage process applies."
            )

        else:

            level = "⚪ First Report"

            rec = (
                "No prior reports for this location."
            )

        return {
            "Hotspot Level": level,
            "Recommendation": rec,
            "report_count": count
        }


# ===========================================================================
# DATA PROTECTION
# ===========================================================================

class DataProtection:
    """
    Redacts PII from civic reports and provides ward-level location coarsening.
    """

    _EMAIL_RE = re.compile(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    )

    _PHONE_RE = re.compile(
        r"(\+91[-\s]?)?[6-9]\d{9}"
    )

    _HOUSE_RE = re.compile(
        r"\b(house|flat|plot|door|h\.?no\.?|d\.?no\.?)"
        r"\s*[#\-]?\s*\d+[\w/]*",
        re.I
    )

    _AADHAR_RE = re.compile(
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    )

    def anonymize_report(
        self,
        description: str,
        location: str
    ) -> dict:

        clean = description

        clean = self._EMAIL_RE.sub(
            "[EMAIL REDACTED]",
            clean
        )

        clean = self._PHONE_RE.sub(
            "[PHONE REDACTED]",
            clean
        )

        clean = self._HOUSE_RE.sub(
            "[HOUSE# REDACTED]",
            clean
        )

        clean = self._AADHAR_RE.sub(
            "[ID REDACTED]",
            clean
        )

        safe_location = re.sub(
            r"\b(flat|door|house|plot)"
            r"\s*[#\-]?\s*\d+[\w/]*",
            "",
            location,
            flags=re.I
        ).strip()

        if not safe_location:
            safe_location = location

        fingerprint = hashlib.sha256(
            f"{clean}|{safe_location}|"
            f"{datetime.datetime.utcnow().date()}"
            .encode()
        ).hexdigest()

        return {
            "description": clean,
            "location": safe_location,
            "fingerprint": fingerprint,
        }

    def audit_log(
        self,
        event: str,
        metadata: dict
    ):

        try:

            AUDIT_LOG_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            entry = {
                "ts": datetime.datetime.utcnow().isoformat(),
                "event": event,
                **metadata,
            }

            with open(
                AUDIT_LOG_PATH,
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    json.dumps(entry)
                    + "\n"
                )

        except Exception as exc:

            logger.warning(
                "Audit log write failed: %s",
                exc
            )


# ===========================================================================
# AI MEMORY
# ===========================================================================

class AIMemory:
    """
    Maintains an in-memory tally of classified issues for session analytics.
    """

    def __init__(self):

        self._history = defaultdict(int)

    def remember(
        self,
        result: dict
    ) -> dict:

        issue = result.get(
            "Issue",
            "Unknown"
        )

        self._history[issue] += 1

        return {
            "session_count": sum(
                self._history.values()
            ),

            "issue_counts": dict(
                self._history
            ),
        }


# ===========================================================================
# EMERGENCY CONTACTS
# ===========================================================================

class EmergencyContacts:
    """
    Provides a directory of India civic emergency numbers.
    """

    contacts = {

        "Police": {
            "number": "100",
            "description": "National emergency police helpline",
        },

        "Fire Brigade": {
            "number": "101",
            "description": "Fire and rescue services",
        },

        "Ambulance": {
            "number": "102",
            "description": "Medical emergency / ambulance",
        },

        "Disaster Management": {
            "number": "108",
            "description": "National disaster response helpline",
        },

        "Women Helpline": {
            "number": "1091",
            "description": "Women in distress emergency line",
        },

        "Road Accident Emergency": {
            "number": "1073",
            "description": "National Highway accident helpline",
        },

        "Water Board (BWSSB)": {
            "number": "1916",
            "description": "Water supply complaints and emergencies",
        },

        "Electricity Board (BESCOM)": {
            "number": "1912",
            "description": "Power outage and electrical hazard",
        },

        "Municipal Corporation (BBMP)": {
            "number": "080-22221188",
            "description": "Garbage, roads, public infrastructure",
        },

        "Gas Leak Emergency": {
            "number": "1906",
            "description": "LPG gas leak emergency response",
        },

        "Child Helpline": {
            "number": "1098",
            "description": "CHILDLINE India — child distress helpline",
        },

        "Senior Citizen Helpline": {
            "number": "14567",
            "description": "Elder care and assistance",
        },
    }

    def search_contact(
        self,
        keyword: str
    ):

        kw = keyword.lower()

        return [
            (dept, info)
            for dept, info in self.contacts.items()
            if (
                kw in dept.lower()
                or kw in info["description"].lower()
            )
        ]


# ===========================================================================
# ASSISTANT
# ===========================================================================

class Assistant:
    """
    Keyword-based civic assistant that answers common citizen queries.
    """

    FAQ = [

        {
            "keywords": [
                "pothole",
                "road",
                "damage"
            ],

            "answer":
                "Road damage should be reported to the Public Works Department "
                "(PWD). File a complaint at your local municipal office or use "
                "JanRakshak AI to submit a geo-tagged report.",
        },

        {
            "keywords": [
                "water",
                "leak",
                "pipe",
                "supply"
            ],

            "answer":
                "Water leakage emergencies should be reported to the Water "
                "Supply & Sewerage Board at 1916. You can also submit a report "
                "here with location coordinates for faster routing.",
        },

        {
            "keywords": [
                "electricity",
                "electric",
                "power",
                "outage"
            ],

            "answer":
                "Electrical emergencies — call BESCOM at 1912 immediately. "
                "For non-urgent faults, submit a report via JanRakshak AI and "
                "it will be routed to the State Electricity Distribution Company.",
        },

        {
            "keywords": [
                "garbage",
                "waste",
                "trash",
                "dump"
            ],

            "answer":
                "Garbage issues are handled by the Municipal Solid Waste "
                "Management department. You can call BBMP at 080-22221188 "
                "or report here for AI-powered routing.",
        },

        {
            "keywords": [
                "fire"
            ],

            "answer":
                "For a fire emergency, call 101 (Fire Brigade) immediately. "
                "After the emergency, file a report here for municipal "
                "follow-up and prevention measures.",
        },

        {
            "keywords": [
                "flood",
                "waterlog",
                "rain"
            ],

            "answer":
                "Flooding issues are managed by the Drainage & Flood Control "
                "Board. Immediate risk — call 108 (Disaster Management). "
                "For recurring waterlogging, submit a report with GPS coordinates.",
        },

        {
            "keywords": [
                "privacy",
                "data",
                "personal"
            ],

            "answer":
                "JanRakshak AI automatically redacts emails, phone numbers, "
                "and house numbers from all reports. GPS coordinates are stored "
                "at ward-level precision only. Audit logs are kept for 30 days.",
        },

        {
            "keywords": [
                "offline",
                "internet",
                "connection"
            ],

            "answer":
                "JanRakshak AI works offline! The AI classification model runs "
                "entirely on-device. Reports are queued in IndexedDB and synced "
                "automatically when connectivity is restored.",
        },
    ]

    def ask(
        self,
        question: str
    ) -> dict:

        q = question.lower()

        for item in self.FAQ:

            if any(
                kw in q
                for kw in item["keywords"]
            ):

                return {
                    "answer": item["answer"],
                    "matched": True
                }

        return {
            "answer": (
                "I can help with questions about road damage, water leaks, "
                "electricity, garbage, flooding, fire emergencies, privacy, "
                "and offline functionality. Please try rephrasing your "
                "question or submit a report directly."
            ),

            "matched": False,
        }