import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

app = FastAPI(title="LLM Service")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

class LLMRequest(BaseModel):
    action: Optional[str] = None
    question: Optional[str] = None
    records: List[Dict[str, Any]] = []

class DashboardRequest(BaseModel):
    record_date: str
    kpis: Dict[str, Any]
    region_counts: List[Dict[str, Any]]
    critical_patients: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok"}

def fake_llm_response(action: str, records: List[Dict[str, Any]], question: Optional[str] = None):
    if action == "critical_patients":
        return {
            "summary": f"There are {len(records)} patients requiring immediate attention.",
            "recommendations": [
                "Prioritize ICU and bed allocation.",
                "Notify senior physicians for urgent review.",
                "Monitor critical patients continuously."
            ],
            "alerts": [
                {
                    "patient_id": r.get("Patient_ID"),
                    "message": f"{r.get('Primary_Condition')} requires urgent attention.",
                    "severity": "high"
                }
                for r in records[:5]
            ]
        }

    if action == "discharge_candidates":
        return {
            "summary": f"There are {len(records)} likely discharge candidates based on stable status and no pending lab or ICU needs.",
            "recommendations": [
                "Review discharge checklist.",
                "Confirm follow-up schedule.",
                "Prepare patient discharge summary."
            ],
            "alerts": []
        }

    if action == "manager_question":
        return {
            "summary": f"Answer for question: {question}",
            "recommendations": [
                "Use filtered data for operational decision-making."
            ],
            "alerts": []
        }

    return {
        "summary": "No summary available.",
        "recommendations": [],
        "alerts": []
    }

@app.post("/summarize")
def summarize(payload: LLMRequest):
    return fake_llm_response(payload.action or "unknown", payload.records, payload.question)

@app.post("/recommend")
def recommend(payload: LLMRequest):
    return fake_llm_response(payload.action or "unknown", payload.records, payload.question)

@app.post("/explain")
def explain(payload: LLMRequest):
    return fake_llm_response(payload.action or "unknown", payload.records, payload.question)

@app.post("/dashboard-response")
def dashboard_response(payload: DashboardRequest):
    return {
        "page_title": f"Patient Dashboard - {payload.record_date}",
        "summary": (
            f"Total patients: {payload.kpis['total_patients']}. "
            f"Critical: {payload.kpis['critical']}, "
            f"Moderate: {payload.kpis['moderate']}, "
            f"Stable: {payload.kpis['stable']}."
        ),
        "kpis": [
            {"label": "Total Patients", "value": payload.kpis["total_patients"]},
            {"label": "Critical", "value": payload.kpis["critical"]},
            {"label": "Moderate", "value": payload.kpis["moderate"]},
            {"label": "Stable", "value": payload.kpis["stable"]},
        ],
        "alerts": [
            {
                "patient_id": p["Patient_ID"],
                "message": f"{p['Primary_Condition']} patient in {p['Hospital_Branch']} needs urgent attention.",
                "severity": "high"
            }
            for p in payload.critical_patients[:5]
        ],
        "charts": [
            {
                "chart_type": "bar",
                "title": "Patients by Region",
                "data": [
                    {"label": x["Region"], "value": x["count"]}
                    for x in payload.region_counts
                ]
            }
        ],
        "recommendations": [
            "Review critical patient queue first.",
            "Verify ICU and bed capacity by branch.",
            "Track moderate cases for any worsening trend."
        ]
    }