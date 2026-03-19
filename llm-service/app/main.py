import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from google import genai

app = FastAPI(title="LLM Service")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6IUu38n0YfKf3eRtcESsq9uipNZerF987Lm533-ZirfYA")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


class LLMRequest(BaseModel):
    action: Optional[str] = None
    question: Optional[str] = None
    records: List[Dict[str, Any]] = []


class DashboardRequest(BaseModel):
    record_date: str
    kpis: Dict[str, Any]
    region_counts: List[Dict[str, Any]]
    critical_patients: List[Dict[str, Any]]


SYSTEM_PROMPT = """
You are a healthcare operations assistant.

Rules:
- Use only the provided structured data.
- Do not diagnose.
- Do not prescribe treatment.
- Focus on triage operations, ICU allocation, monitoring, discharge planning, and resource allocation.
- Return valid JSON only.
- No markdown.
- No text outside JSON.
"""


def build_prompt(action: str, data: Dict[str, Any], schema: Dict[str, Any]) -> str:
    return f"""
{SYSTEM_PROMPT}

Action:
{action}

Input Data:
{json.dumps(data, indent=2)}

Return only valid JSON in this schema:
{json.dumps(schema, indent=2)}
"""


def call_gemini(action: str, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing")

    prompt = build_prompt(action, data, schema)

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    )

    text = response.text.strip()

    # remove markdown fences if model adds them
    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()
    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON from model: {text}")


@app.get("/health")
def health():
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY)}


@app.post("/summarize")
def summarize(payload: LLMRequest):
    schema = {
        "summary": "string",
        "recommendations": ["string"],
        "alerts": [
            {
                "patient_id": "string",
                "message": "string",
                "severity": "low | medium | high"
            }
        ]
    }

    data = {
        "action": payload.action,
        "question": payload.question,
        "records": payload.records
    }

    return call_gemini(payload.action or "summarize", data, schema)


@app.post("/recommend")
def recommend(payload: LLMRequest):
    schema = {
        "summary": "string",
        "recommendations": ["string"],
        "alerts": []
    }

    data = {
        "action": payload.action,
        "question": payload.question,
        "records": payload.records
    }

    return call_gemini(payload.action or "recommend", data, schema)


@app.post("/explain")
def explain(payload: LLMRequest):
    schema = {
        "summary": "string",
        "recommendations": ["string"],
        "alerts": [
            {
                "patient_id": "string",
                "message": "string",
                "severity": "low | medium | high"
            }
        ]
    }

    data = {
        "action": payload.action,
        "question": payload.question,
        "records": payload.records
    }

    return call_gemini(payload.action or "explain", data, schema)


@app.post("/dashboard-response")
def dashboard_response(payload: DashboardRequest):
    schema = {
        "page_title": "string",
        "summary": "string",
        "kpis": [
            {"label": "string", "value": 0}
        ],
        "alerts": [
            {"patient_id": "string", "message": "string", "severity": "low | medium | high"}
        ],
        "patients": [
            {
                "patient_id": "string",
                "condition": "string",
                "priority_group": "Critical | Moderate | Stable",
                "severity_score": 0,
                "icu_required": "Yes | No",
                "branch": "string",
                "region": "string"
            }
        ],
        "charts": [
            {
                "chart_type": "bar",
                "title": "string",
                "data": [
                    {"label": "string", "value": 0}
                ]
            }
        ],
        "recommendations": ["string"]
    }

    data = {
        "record_date": payload.record_date,
        "kpis": payload.kpis,
        "region_counts": payload.region_counts,
        "critical_patients": payload.critical_patients
    }

    return call_gemini("dashboard_response", data, schema)