import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

from pydantic import BaseModel, Field


API_KEY = os.getenv("TASK2M_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
BASE_URL = os.getenv("TASK2M_AGENT_BASE_URL", "")
MODEL_NAME = os.getenv("TASK2M_AGENT_MODEL", "gpt-5.2")
ENABLE_LLM_REQUEST_PARSER = os.getenv("TASK2M_AGENT_ENABLE_REQUEST_PARSER", "1") == "1"
ENABLE_LLM_EXPLANATION = os.getenv("TASK2M_AGENT_ENABLE_LLM", "1") == "1"
LLM_TIMEOUT_SECONDS = float(os.getenv("TASK2M_AGENT_TIMEOUT", "30"))

DEFAULT_CSV_CANDIDATES = [
    os.getenv("TASK_MCP_CSV_PATH", ""),
    "output/task_mcp_top10_info.csv",
    str(Path(__file__).resolve().parent / "output" / "task_mcp_top10_info.csv"),
    str(Path.cwd() / "output" / "task_mcp_top10_info.csv"),
]

logger = logging.getLogger("t2magent")
logging.basicConfig(
    level=os.getenv("TASK2M_AGENT_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)



class RecommendRequest(BaseModel):
    task_id: int
    requirement: str = ""


class RecommendResponse(BaseModel):
    task_id: int
    chosen_mcp_ids: List[int]
    overall_reason: str
    per_mcp_reason: Dict[str, str]
    raw_response: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    task_id: int
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    task_id: int
    assistant_reply: str
    raw_response: str


@dataclass
class CandidateEvidence:
    index: int
    mcp_id: int
    rank: Optional[int]
    name: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class ParsedTaskRequest:
    latest_user_message: str
    all_user_messages: str
    normalized_intent: str
    explicit_constraints: Dict[str, Any]
    missing_constraints: List[str]
    request_type: str




def _find_existing_csv_path() -> str:
    for candidate in DEFAULT_CSV_CANDIDATES:
        if candidate and Path(candidate).exists():
            return candidate
    return DEFAULT_CSV_CANDIDATES[1]


TASK_MCP_CSV_PATH = _find_existing_csv_path()


def _load_dataframe() -> pd.DataFrame:
    if not Path(TASK_MCP_CSV_PATH).exists():
        logger.warning("CSV file not found at %s. API will start, but requests will fail until the file is provided.", TASK_MCP_CSV_PATH)
        return pd.DataFrame()

    df = pd.read_csv(TASK_MCP_CSV_PATH)
    if "task_id" not in df.columns:
        raise RuntimeError(f"CSV at {TASK_MCP_CSV_PATH} must contain a task_id column.")
    df["task_id"] = pd.to_numeric(df["task_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["task_id"]).copy()
    df["task_id"] = df["task_id"].astype(int)
    return df


DF = _load_dataframe()



KNOWN_LANGUAGES = [
    "python", "javascript", "typescript", "java", "c++", "cpp", "c#", "go", "rust",
    "php", "ruby", "swift", "kotlin", "scala", "r", "matlab", "shell", "bash",
    "sql", "html", "css", "dart", "lua", "perl", "objective-c",
]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()



def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None



def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and pd.isna(value):
            return default
        text = str(value).replace(",", "").strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default



def _truthy_official(value: str) -> bool:
    text = value.lower()
    return "official" in text and "non-official" not in text and "unofficial" not in text



def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in keywords)



def _extract_languages(text: str) -> List[str]:
    lower = text.lower()
    hits: List[str] = []
    for lang in KNOWN_LANGUAGES:
        if re.search(rf"(?<![a-z0-9]){re.escape(lang)}(?![a-z0-9])", lower):
            normalized = {"cpp": "c++"}.get(lang, lang)
            if normalized not in hits:
                hits.append(normalized)
    return hits



def _compact_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)



def _try_extract_json(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in the response.")



def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()




class TaskRepository:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_task_row(self, task_id: int) -> pd.Series:
        if self.df.empty:
            raise ValueError(
                f"Dataset is not loaded. Expected CSV at {TASK_MCP_CSV_PATH}."
            )
        filtered = self.df[self.df["task_id"] == task_id]
        if filtered.empty:
            raise ValueError(f"task_id={task_id} does not exist in task_mcp_top10_info.csv.")
        return filtered.iloc[0]

    def build_task_payload(self, task_row: pd.Series) -> Dict[str, str]:
        payload = {
            "task_id": str(task_row.get("task_id", "")),
            "task_name": _safe_str(task_row.get("task_name", "")),
            "task_description": _safe_str(task_row.get("task_description", "")),
            "task_programming_language": _safe_str(task_row.get("task_programming_language", "")),
            "task_category": _safe_str(task_row.get("task_category", "")),
            "task_theme": _safe_str(task_row.get("task_theme", "")),
        }
        return payload

    def build_candidate_evidence(self, task_row: pd.Series) -> List[CandidateEvidence]:
        candidates: List[CandidateEvidence] = []
        for i in range(1, 11):
            mcp_id = _safe_int(task_row.get(f"mcp{i}_num"))
            if mcp_id is None:
                continue

            metadata = {
                "rank": _safe_int(task_row.get(f"mcp{i}_rank")),
                "license": _safe_str(task_row.get(f"mcp{i}_license")),
                "language": _safe_str(task_row.get(f"mcp{i}_language")),
                "activity": _safe_str(task_row.get(f"mcp{i}_activity")),
                "category": _safe_str(task_row.get(f"mcp{i}_new_category")),
                "system": _safe_str(task_row.get(f"mcp{i}_system")),
                "official": _safe_str(task_row.get(f"mcp{i}_official")),
                "stars": _safe_str(task_row.get(f"mcp{i}_star")),
                "watching": _safe_str(task_row.get(f"mcp{i}_watching")),
            }

            evidence = {
                "repository_url": _safe_str(task_row.get(f"mcp{i}_url", "")),
                "directory_url": _safe_str(task_row.get(f"mcp{i}_web_url", "")),
                "tools": _safe_str(task_row.get(f"mcp{i}_tools", "")),
                "author": _safe_str(task_row.get(f"mcp{i}_author", "")),
                "origin": _safe_str(task_row.get(f"mcp{i}_origin", "")),
            }

            candidates.append(
                CandidateEvidence(
                    index=i,
                    mcp_id=mcp_id,
                    rank=metadata["rank"],
                    name=_safe_str(task_row.get(f"mcp{i}_name", "")),
                    description=_safe_str(task_row.get(f"mcp{i}_description", "")),
                    metadata=metadata,
                    evidence={k: v for k, v in evidence.items() if v},
                )
            )
        return candidates




class LLMAdapter:
    def __init__(self):
        self.client: Optional[OpenAI] = None
        if API_KEY and OpenAI is not None:
            self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=LLM_TIMEOUT_SECONDS)
        elif API_KEY and OpenAI is None:
            logger.warning("API key is configured but the OpenAI client package is unavailable. Falling back to deterministic behavior.")
        else:
            logger.warning("No API key configured. Falling back to deterministic behavior when LLM assistance is needed.")

    @property
    def enabled(self) -> bool:
        return self.client is not None and ENABLE_LLM_EXPLANATION

    @property
    def request_parser_enabled(self) -> bool:
        return self.client is not None and ENABLE_LLM_REQUEST_PARSER

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        if not self.client:
            raise RuntimeError("LLM client is not configured.")
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        return content




class TaskUnderstandingService:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    def parse(self, task_payload: Dict[str, str], messages: Sequence[ChatMessage]) -> ParsedTaskRequest:
        user_messages = [m.content.strip() for m in messages if m.role.lower() == "user" and m.content.strip()]
        latest_user = user_messages[-1] if user_messages else ""
        all_user = "\n".join(user_messages)

        heuristic = self._heuristic_parse(task_payload, latest_user, all_user)
        if not self.llm.request_parser_enabled:
            return heuristic

        try:
            llm_result = self._llm_parse(task_payload, latest_user, all_user)
            merged_constraints = {**heuristic.explicit_constraints, **llm_result.explicit_constraints}
            missing = list(dict.fromkeys(heuristic.missing_constraints + llm_result.missing_constraints))
            normalized_intent = llm_result.normalized_intent or heuristic.normalized_intent
            request_type = llm_result.request_type or heuristic.request_type
            return ParsedTaskRequest(
                latest_user_message=latest_user,
                all_user_messages=all_user,
                normalized_intent=normalized_intent,
                explicit_constraints=merged_constraints,
                missing_constraints=missing,
                request_type=request_type,
            )
        except Exception as exc:
            logger.warning("LLM request parsing failed; using heuristic parser instead: %s", exc)
            return heuristic

    def _heuristic_parse(self, task_payload: Dict[str, str], latest_user: str, all_user: str) -> ParsedTaskRequest:
        lower = all_user.lower()
        languages = _extract_languages(all_user)
        explicit_constraints: Dict[str, Any] = {
            "preferred_languages": languages,
            "prefer_official": _contains_any(lower, ["official", "verified"]) and not _contains_any(lower, ["not official", "non-official", "unofficial ok"]),
            "prefer_easy_to_use": _contains_any(lower, ["easy to use", "easy-to-use", "simple", "straightforward", "beginner-friendly", "minimal friction"]),
            "prefer_lightweight": _contains_any(lower, ["light", "lightweight", "lighter", "minimal", "lean"]),
            "prefer_active": _contains_any(lower, ["active", "well maintained", "maintained", "popular", "stable"]),
            "prefer_no_api_key": _contains_any(lower, ["no api key", "without api key", "without credentials"]),
            "prefer_privacy": _contains_any(lower, ["privacy", "private", "local only", "offline", "self-hosted", "on-prem"]),
            "needs_comparison": _contains_any(lower, ["compare", "trade-off", "pros and cons", "difference"]),
            "needs_adoption_guidance": _contains_any(lower, ["how to use", "usage", "instruction", "adoption", "setup", "integrate", "deployment"]),
        }

        if not explicit_constraints["preferred_languages"]:
            task_lang = _extract_languages(task_payload.get("task_programming_language", ""))
            if task_lang:
                explicit_constraints["preferred_languages"] = task_lang

        request_type = "recommendation"
        if latest_user:
            latest_lower = latest_user.lower()
            if _contains_any(latest_lower, ["why", "not choose", "why not", "compare", "difference", "trade-off"]):
                request_type = "follow_up_explanation"
            elif _contains_any(latest_lower, ["how to use", "usage", "setup", "integrate", "deployment", "instruction"]):
                request_type = "adoption_guidance"
            elif _contains_any(latest_lower, ["clarify", "missing", "constraint"]):
                request_type = "clarification"

        missing_constraints: List[str] = []
        if not explicit_constraints["preferred_languages"] and not task_payload.get("task_programming_language"):
            missing_constraints.append("programming language preference")
        if not latest_user:
            missing_constraints.append("user requirement")

        normalized_intent = _normalize_whitespace(
            f"Task: {task_payload.get('task_name', '')}. Base requirement: {task_payload.get('task_description', '')}. "
            f"Latest user request: {latest_user or 'No additional requirement provided.'}"
        )

        return ParsedTaskRequest(
            latest_user_message=latest_user,
            all_user_messages=all_user,
            normalized_intent=normalized_intent,
            explicit_constraints=explicit_constraints,
            missing_constraints=missing_constraints,
            request_type=request_type,
        )

    def _llm_parse(self, task_payload: Dict[str, str], latest_user: str, all_user: str) -> ParsedTaskRequest:
        system_prompt = """You are a request parser for a retrieval-centered MCP recommendation agent.
Extract the user's intent and constraints into strict JSON.
Return only JSON.
"""
        user_prompt = f"""
[Task]
{_compact_json(task_payload)}

[Conversation - user messages only]
{all_user or latest_user or 'No extra user message.'}

Return JSON in this schema:
{{
  "normalized_intent": "...",
  "request_type": "recommendation | follow_up_explanation | adoption_guidance | clarification",
  "explicit_constraints": {{
    "preferred_languages": ["python"],
    "prefer_official": false,
    "prefer_easy_to_use": false,
    "prefer_lightweight": false,
    "prefer_active": false,
    "prefer_no_api_key": false,
    "prefer_privacy": false,
    "needs_comparison": false,
    "needs_adoption_guidance": false
  }},
  "missing_constraints": ["..."]
}}
"""
        content = self.llm.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.0)
        data = _try_extract_json(content)
        constraints = data.get("explicit_constraints", {}) or {}
        return ParsedTaskRequest(
            latest_user_message=latest_user,
            all_user_messages=all_user,
            normalized_intent=_safe_str(data.get("normalized_intent", "")),
            explicit_constraints={
                "preferred_languages": list(constraints.get("preferred_languages", []) or []),
                "prefer_official": bool(constraints.get("prefer_official", False)),
                "prefer_easy_to_use": bool(constraints.get("prefer_easy_to_use", False)),
                "prefer_lightweight": bool(constraints.get("prefer_lightweight", False)),
                "prefer_active": bool(constraints.get("prefer_active", False)),
                "prefer_no_api_key": bool(constraints.get("prefer_no_api_key", False)),
                "prefer_privacy": bool(constraints.get("prefer_privacy", False)),
                "needs_comparison": bool(constraints.get("needs_comparison", False)),
                "needs_adoption_guidance": bool(constraints.get("needs_adoption_guidance", False)),
            },
            missing_constraints=list(data.get("missing_constraints", []) or []),
            request_type=_safe_str(data.get("request_type", "recommendation")) or "recommendation",
        )




class RecommendationCore:
    def rank_candidates(
        self,
        task_payload: Dict[str, str],
        request_spec: ParsedTaskRequest,
        candidates: Sequence[CandidateEvidence],
    ) -> List[CandidateEvidence]:
        ranked: List[CandidateEvidence] = []
        task_language = task_payload.get("task_programming_language", "")
        task_theme = task_payload.get("task_theme", "")

        for candidate in candidates:
            score, breakdown = self._score_candidate(task_language, task_theme, request_spec, candidate)
            candidate.score = score
            candidate.score_breakdown = breakdown
            ranked.append(candidate)

        ranked.sort(key=lambda c: (-c.score, c.rank if c.rank is not None else 9999, c.name.lower()))
        return ranked

    def _score_candidate(
        self,
        task_language: str,
        task_theme: str,
        request_spec: ParsedTaskRequest,
        candidate: CandidateEvidence,
    ) -> Tuple[float, Dict[str, float]]:
        breakdown: Dict[str, float] = {}
        metadata = candidate.metadata
        description_plus = " ".join([
            candidate.name,
            candidate.description,
            _safe_str(metadata.get("category")),
            _safe_str(metadata.get("language")),
            _safe_str(metadata.get("system")),
            _safe_str(candidate.evidence.get("tools", "")),
        ]).lower()

        if candidate.rank is None:
            rank_prior = 0.30
        else:
            rank_prior = max(0.0, (11 - min(max(candidate.rank, 1), 10)) / 10.0)
        breakdown["t2mrec_rank_prior"] = rank_prior * 0.45

        preferred_languages = request_spec.explicit_constraints.get("preferred_languages", []) or []
        candidate_lang_text = _safe_str(metadata.get("language", "")).lower()
        language_hit = 0.0
        for lang in preferred_languages:
            if lang.lower() in candidate_lang_text:
                language_hit = 1.0
                break
        if not language_hit and task_language and task_language.lower() in candidate_lang_text.lower():
            language_hit = 0.7
        breakdown["language_compatibility"] = language_hit * 0.15

        official_score = 0.0
        if request_spec.explicit_constraints.get("prefer_official"):
            official_score = 1.0 if _truthy_official(_safe_str(metadata.get("official", ""))) else 0.0
        breakdown["official_preference"] = official_score * 0.10

        usability_proxy = 0.0
        if request_spec.explicit_constraints.get("prefer_easy_to_use"):
            if _contains_any(description_plus, ["simple", "easy", "straightforward", "minimal", "caption", "summarizer"]):
                usability_proxy += 0.6
            if _contains_any(description_plus, ["without api key", "no api key"]):
                usability_proxy += 0.4
        if request_spec.explicit_constraints.get("prefer_lightweight"):
            if _contains_any(description_plus, ["light", "lightweight", "simple", "single", "focused", "caption"]):
                usability_proxy += 0.5
        if request_spec.explicit_constraints.get("prefer_no_api_key") and _contains_any(description_plus, ["without api key", "no api key"]):
            usability_proxy += 0.6
        breakdown["usability_proxy"] = min(usability_proxy, 1.0) * 0.10

        stars = _safe_float(metadata.get("stars", 0.0))
        watching = _safe_float(metadata.get("watching", 0.0))
        activity_value = _safe_float(metadata.get("activity", 0.0))
        popularity_proxy = min((stars / 1000.0) + (watching / 100.0) + (activity_value / 10.0), 1.0)
        if request_spec.explicit_constraints.get("prefer_active"):
            breakdown["activity_proxy"] = popularity_proxy * 0.08
        else:
            breakdown["activity_proxy"] = popularity_proxy * 0.03

        theme_terms = _extract_key_terms(task_theme + " " + request_spec.latest_user_message)
        overlap_hits = sum(1 for term in theme_terms if term and term in description_plus)
        overlap_proxy = min(overlap_hits / 3.0, 1.0)
        breakdown["theme_overlap"] = overlap_proxy * 0.10

        total = sum(breakdown.values())
        return round(total, 6), breakdown



def _extract_key_terms(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())
    stop = {
        "task", "please", "recommend", "mcp", "server", "tool", "need", "want", "with",
        "that", "this", "from", "into", "about", "than", "more", "less", "easy", "use",
        "official", "focused", "based", "provide", "details", "instructions",
    }
    result: List[str] = []
    for word in words:
        if word in stop:
            continue
        if word not in result:
            result.append(word)
    return result[:8]




class EvidenceGroundingService:
    def build_shortlist_package(
        self,
        ranked_candidates: Sequence[CandidateEvidence],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        package: List[Dict[str, Any]] = []
        for candidate in list(ranked_candidates)[:limit]:
            package.append(
                {
                    "mcp_id": candidate.mcp_id,
                    "rank": candidate.rank,
                    "name": candidate.name,
                    "description": candidate.description,
                    "metadata": candidate.metadata,
                    "evidence": candidate.evidence,
                    "deterministic_score": candidate.score,
                    "score_breakdown": candidate.score_breakdown,
                    "adoption_notes": self._derive_adoption_notes(candidate),
                    "tradeoff_notes": self._derive_tradeoff_notes(candidate),
                }
            )
        return package

    def build_full_candidate_package(self, ranked_candidates: Sequence[CandidateEvidence]) -> List[Dict[str, Any]]:
        full: List[Dict[str, Any]] = []
        for candidate in ranked_candidates:
            full.append(
                {
                    "mcp_id": candidate.mcp_id,
                    "rank": candidate.rank,
                    "name": candidate.name,
                    "description": candidate.description,
                    "metadata": candidate.metadata,
                    "evidence": candidate.evidence,
                }
            )
        return full

    def _derive_adoption_notes(self, candidate: CandidateEvidence) -> List[str]:
        notes: List[str] = []
        language = _safe_str(candidate.metadata.get("language", ""))
        system = _safe_str(candidate.metadata.get("system", ""))
        license_text = _safe_str(candidate.metadata.get("license", ""))
        description = (candidate.description + " " + _safe_str(candidate.evidence.get("tools", ""))).lower()

        if language:
            notes.append(f"Check runtime compatibility with {language} before integration.")
        if system:
            notes.append(f"Validate deployment assumptions for system/platform: {system}.")
        if license_text:
            notes.append(f"Review license/compliance requirements: {license_text}.")
        if "api key" in description:
            if "without api key" in description or "no api key" in description:
                notes.append("This candidate appears usable without API-key setup based on the available metadata.")
            else:
                notes.append("This candidate may require credential or API-key setup.")
        if not notes:
            notes.append("Review repository documentation and installation steps before production use.")
        return notes[:4]

    def _derive_tradeoff_notes(self, candidate: CandidateEvidence) -> List[str]:
        notes: List[str] = []
        description = candidate.description.lower()
        if _contains_any(description, ["caption", "extract"]):
            notes.append("More specialized and likely simpler, but potentially narrower in capability coverage.")
        if _contains_any(description, ["summar", "workflow", "multiple"]):
            notes.append("Richer end-to-end functionality, but integration complexity may be higher.")
        if _truthy_official(_safe_str(candidate.metadata.get("official", ""))):
            notes.append("Official status may reduce trust risk, though it does not guarantee task fit.")
        if not notes:
            notes.append("Trade-offs should be checked against workflow fit, maintenance status, and required setup.")
        return notes[:3]




class ReliabilityController:
    def __init__(self, allowed_ids: Iterable[int]):
        self.allowed_ids = set(allowed_ids)

    def validate_recommendation_json(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        ids = data.get("chosen_mcp_ids")
        if not isinstance(ids, list) or len(ids) != 2:
            return False, "chosen_mcp_ids must contain exactly two IDs."
        try:
            normalized_ids = [int(x) for x in ids]
        except Exception:
            return False, "chosen_mcp_ids contains non-integer values."
        if len(set(normalized_ids)) != 2:
            return False, "chosen_mcp_ids must contain two distinct IDs."
        invalid = [x for x in normalized_ids if x not in self.allowed_ids]
        if invalid:
            return False, f"chosen_mcp_ids contains IDs outside the candidate set: {invalid}"
        if not isinstance(data.get("overall_reason"), str):
            return False, "overall_reason must be a string."
        per_reason = data.get("per_mcp_reason")
        if not isinstance(per_reason, dict):
            return False, "per_mcp_reason must be a dictionary."
        for mcp_id in normalized_ids:
            if str(mcp_id) not in per_reason:
                return False, f"per_mcp_reason is missing key {mcp_id}."
        return True, "ok"

    def validate_chat_answer(self, text: str) -> Tuple[bool, str]:
        found = {int(x) for x in re.findall(r"\b(?:id|ID)\s*[:：]?\s*(\d+)\b", text)}
        invalid = [x for x in found if x not in self.allowed_ids]
        if invalid:
            return False, f"chat response mentioned IDs outside the candidate set: {invalid}"
        return True, "ok"




class ResponseGenerator:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    def generate_recommendation(
        self,
        task_payload: Dict[str, str],
        request_spec: ParsedTaskRequest,
        shortlist: List[Dict[str, Any]],
        full_candidates: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if not self.llm.enabled:
            return None, ""

        system_prompt = """You are the response layer of a retrieval-centered MCP recommendation agent.
You receive:
1) a task description,
2) a normalized task specification derived from the user conversation,
3) a top-5 shortlist grounded in T2MRec results and evidence.
Your job is to produce a final recommendation by selecting exactly two MCPs from the shortlist.
Do not use MCPs outside the provided shortlist.
Return only valid JSON.
"""

        user_prompt = f"""
[Task Payload]
{_compact_json(task_payload)}

[Normalized Task Specification]
{_compact_json({
    'normalized_intent': request_spec.normalized_intent,
    'request_type': request_spec.request_type,
    'explicit_constraints': request_spec.explicit_constraints,
    'missing_constraints': request_spec.missing_constraints,
})}

[Evidence-Grounded Shortlist (top-5)]
{_compact_json({'shortlist': shortlist})}

[Full Candidate Pool (top-10)]
{_compact_json({'candidates': full_candidates})}

Return JSON in exactly this schema:
{{
  "chosen_mcp_ids": [123, 456],
  "overall_reason": "A concise synthesis grounded in the shortlist evidence, noting fit, trade-offs, and uncertainty if needed.",
  "per_mcp_reason": {{
    "123": "Why MCP 123 is recommended, with evidence-grounded fit and adoption notes.",
    "456": "Why MCP 456 is recommended, with evidence-grounded fit and adoption notes."
  }}
}}
"""
        content = self.llm.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.1)
        data = _try_extract_json(content)
        return data, content

    def generate_chat_reply(
        self,
        task_payload: Dict[str, str],
        request_spec: ParsedTaskRequest,
        shortlist: List[Dict[str, Any]],
        full_candidates: List[Dict[str, Any]],
        history_messages: Sequence[ChatMessage],
    ) -> str:
        if not self.llm.enabled:
            raise RuntimeError("LLM disabled")

        conversation = [
            {"role": "system", "content": self._chat_system_prompt(task_payload, request_spec, shortlist, full_candidates)}
        ]
        for m in history_messages:
            role = m.role.lower()
            if role not in {"user", "assistant"}:
                role = "user"
            conversation.append({"role": role, "content": m.content})

        return self.llm.chat(conversation, temperature=0.2)

    def _chat_system_prompt(
        self,
        task_payload: Dict[str, str],
        request_spec: ParsedTaskRequest,
        shortlist: List[Dict[str, Any]],
        full_candidates: List[Dict[str, Any]],
    ) -> str:
        return f"""You are T2MAgent's interaction layer.
Work as a retrieval-centered, evidence-grounded MCP recommendation assistant.

You must follow these rules:
- Base your answer only on the provided task payload, normalized task specification, shortlist evidence, full candidate pool, and the conversation.
- Never invent MCPs outside the candidate pool.
- When recommending MCPs, recommend at most two and explicitly mention their MCP IDs.
- When the user asks follow-up questions, use trade-off analysis and adoption guidance grounded in the evidence package.
- If the user's requirement is ambiguous, say what is missing and proceed cautiously using the available evidence.
- Prefer concise, readable prose with clear sections when helpful.

[Task Payload]
{_compact_json(task_payload)}

[Normalized Task Specification]
{_compact_json({
    'normalized_intent': request_spec.normalized_intent,
    'request_type': request_spec.request_type,
    'explicit_constraints': request_spec.explicit_constraints,
    'missing_constraints': request_spec.missing_constraints,
})}

[Evidence-Grounded Shortlist (top-5)]
{_compact_json({'shortlist': shortlist})}

[Full Candidate Pool (top-10)]
{_compact_json({'candidates': full_candidates})}
"""




class FallbackBuilder:
    def build_recommendation(
        self,
        task_payload: Dict[str, str],
        request_spec: ParsedTaskRequest,
        ranked_candidates: Sequence[CandidateEvidence],
    ) -> Dict[str, Any]:
        top2 = list(ranked_candidates)[:2]
        ids = [c.mcp_id for c in top2]
        per_reason: Dict[str, str] = {}
        for candidate in top2:
            reasons = self._reason_fragments(task_payload, request_spec, candidate)
            per_reason[str(candidate.mcp_id)] = " ".join(reasons)

        overall = (
            f"These two MCPs are the strongest options from the T2MRec candidate set for task {task_payload.get('task_id')}. "
            f"The selection prioritizes the upstream T2MRec rank, then refines it using request-specific constraints such as "
            f"language fit, official status, ease of use, activity signals, and lightweight adoption cues."
        )
        if request_spec.missing_constraints:
            overall += f" Some constraints are still implicit: {', '.join(request_spec.missing_constraints)}."

        return {
            "chosen_mcp_ids": ids,
            "overall_reason": overall,
            "per_mcp_reason": per_reason,
        }

    def build_chat_reply(
        self,
        task_payload: Dict[str, str],
        request_spec: ParsedTaskRequest,
        ranked_candidates: Sequence[CandidateEvidence],
    ) -> str:
        top2 = list(ranked_candidates)[:2]
        lines = [
            f"Based on task {task_payload.get('task_id')} and the grounded T2MRec shortlist, my current top suggestions are:",
            "",
        ]
        for candidate in top2:
            lines.append(f"- {candidate.name} (ID {candidate.mcp_id})")
            for fragment in self._reason_fragments(task_payload, request_spec, candidate):
                lines.append(f"  - {fragment}")
        if request_spec.explicit_constraints.get("needs_comparison"):
            lines.extend([
                "",
                "Trade-offs:",
                f"- {top2[0].name} is better when you prioritize stronger overall fit and richer workflow coverage.",
                f"- {top2[1].name} is better when you prefer a narrower or simpler integration path.",
            ])
        if request_spec.explicit_constraints.get("needs_adoption_guidance"):
            lines.extend([
                "",
                "Basic adoption guidance:",
                "- Confirm runtime/language compatibility before wiring the MCP into your workflow.",
                "- Review repository setup, credentials, and licensing before production use.",
                "- Start from the higher-ranked MCP first, then use the second one as an alternative or complementary option.",
            ])
        if request_spec.missing_constraints:
            lines.extend([
                "",
                f"Potentially missing constraints: {', '.join(request_spec.missing_constraints)}.",
            ])
        return "\n".join(lines)

    def _reason_fragments(
        self,
        task_payload: Dict[str, str],
        request_spec: ParsedTaskRequest,
        candidate: CandidateEvidence,
    ) -> List[str]:
        fragments = [
            f"It remains highly ranked in the upstream T2MRec results (rank: {candidate.rank or 'unknown'}).",
            f"It matches the task through this capability summary: {candidate.description or 'No description available.'}",
        ]

        preferred_languages = request_spec.explicit_constraints.get("preferred_languages", []) or []
        if preferred_languages and _contains_any(_safe_str(candidate.metadata.get("language", "")), preferred_languages):
            fragments.append(f"Its declared language metadata aligns with your preference: {candidate.metadata.get('language')}.")
        elif candidate.metadata.get("language"):
            fragments.append(f"Its declared language metadata is: {candidate.metadata.get('language')}.")

        if request_spec.explicit_constraints.get("prefer_official"):
            fragments.append(
                "It satisfies the official-preference constraint."
                if _truthy_official(_safe_str(candidate.metadata.get("official", "")))
                else "It does not clearly satisfy the official-preference constraint, so there is some fit uncertainty."
            )

        if request_spec.explicit_constraints.get("prefer_easy_to_use"):
            fragments.append("The available metadata suggests a relatively straightforward adoption path." if candidate.score_breakdown.get("usability_proxy", 0.0) > 0 else "Ease of use is not strongly evidenced in the available metadata.")

        if request_spec.explicit_constraints.get("prefer_active"):
            stars = _safe_str(candidate.metadata.get("stars", ""))
            if stars:
                fragments.append(f"Activity/popularity evidence is available (stars: {stars}).")

        for note in EvidenceGroundingService()._derive_tradeoff_notes(candidate)[:1]:
            fragments.append(note)
        return fragments[:5]




class T2MAgentService:
    def __init__(self, repository: TaskRepository, llm: LLMAdapter):
        self.repository = repository
        self.llm = llm
        self.task_understanding = TaskUnderstandingService(llm)
        self.recommendation_core = RecommendationCore()
        self.evidence_grounding = EvidenceGroundingService()
        self.response_generator = ResponseGenerator(llm)
        self.fallback_builder = FallbackBuilder()

    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        task_row = self.repository.get_task_row(req.task_id)
        task_payload = self.repository.build_task_payload(task_row)
        messages = [ChatMessage(role="user", content=req.requirement or "No specific additional requirements.")]
        request_spec = self.task_understanding.parse(task_payload, messages)
        ranked_candidates = self._rank(task_row, task_payload, request_spec)
        shortlist = self.evidence_grounding.build_shortlist_package(ranked_candidates, limit=5)
        full_candidates = self.evidence_grounding.build_full_candidate_package(ranked_candidates)
        reliability = ReliabilityController({c.mcp_id for c in ranked_candidates})

        raw_response = ""
        try:
            data, raw_response = self.response_generator.generate_recommendation(
                task_payload, request_spec, shortlist, full_candidates
            )
            if data is None:
                raise RuntimeError("LLM output unavailable.")
            ok, message = reliability.validate_recommendation_json(data)
            if not ok:
                raise ValueError(message)
            chosen_ids = [int(x) for x in data["chosen_mcp_ids"]]
            return RecommendResponse(
                task_id=req.task_id,
                chosen_mcp_ids=chosen_ids,
                overall_reason=_safe_str(data.get("overall_reason", "")),
                per_mcp_reason={str(k): _safe_str(v) for k, v in dict(data.get("per_mcp_reason", {})).items()},
                raw_response=raw_response,
            )
        except Exception as exc:
            logger.warning("Falling back to deterministic recommendation for task %s: %s", req.task_id, exc)
            fallback = self.fallback_builder.build_recommendation(task_payload, request_spec, ranked_candidates)
            return RecommendResponse(
                task_id=req.task_id,
                chosen_mcp_ids=[int(x) for x in fallback["chosen_mcp_ids"]],
                overall_reason=fallback["overall_reason"],
                per_mcp_reason=fallback["per_mcp_reason"],
                raw_response=raw_response or _compact_json(fallback),
            )

    def chat(self, req: ChatRequest) -> ChatResponse:
        task_row = self.repository.get_task_row(req.task_id)
        task_payload = self.repository.build_task_payload(task_row)
        request_spec = self.task_understanding.parse(task_payload, req.messages)
        ranked_candidates = self._rank(task_row, task_payload, request_spec)
        shortlist = self.evidence_grounding.build_shortlist_package(ranked_candidates, limit=5)
        full_candidates = self.evidence_grounding.build_full_candidate_package(ranked_candidates)
        reliability = ReliabilityController({c.mcp_id for c in ranked_candidates})

        raw_reply = ""
        try:
            raw_reply = self.response_generator.generate_chat_reply(
                task_payload, request_spec, shortlist, full_candidates, req.messages
            )
            ok, message = reliability.validate_chat_answer(raw_reply)
            if not ok:
                raise ValueError(message)
            return ChatResponse(task_id=req.task_id, assistant_reply=raw_reply, raw_response=raw_reply)
        except Exception as exc:
            logger.warning("Falling back to deterministic chat reply for task %s: %s", req.task_id, exc)
            fallback_text = self.fallback_builder.build_chat_reply(task_payload, request_spec, ranked_candidates)
            return ChatResponse(task_id=req.task_id, assistant_reply=fallback_text, raw_response=raw_reply or fallback_text)

    def _rank(
        self,
        task_row: pd.Series,
        task_payload: Dict[str, str],
        request_spec: ParsedTaskRequest,
    ) -> List[CandidateEvidence]:
        candidates = self.repository.build_candidate_evidence(task_row)
        if not candidates:
            raise ValueError(f"No candidate MCPs found for task_id={task_payload.get('task_id')}.")
        return self.recommendation_core.rank_candidates(task_payload, request_spec, candidates)




repository = TaskRepository(DF)
llm_adapter = LLMAdapter()
service = T2MAgentService(repository, llm_adapter)

app = FastAPI(
    title="T2MAgent API",
    description="Retrieval-centered MCP interaction backend built on top of T2MRec results.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "dataset_loaded": not DF.empty,
        "csv_path": TASK_MCP_CSV_PATH,
        "llm_enabled": llm_adapter.enabled,
        "llm_request_parser_enabled": llm_adapter.request_parser_enabled,
    }


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend_mcp(req: RecommendRequest) -> RecommendResponse:
    try:
        return service.recommend(req)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        logger.exception("Internal error in /api/recommend")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        return service.chat(req)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        logger.exception("Internal error in /api/chat")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
