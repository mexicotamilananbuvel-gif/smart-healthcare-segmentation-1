from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import httpx
import math
from pathlib import Path

app = FastAPI(title="Backend Service")

DATA_PATH = Path(__file__).resolve().parent / "patients.csv"
LLM_SERVICE_URL = "http://llm-service:8001"

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")

    # Normalize column names
    df.columns = [col.strip() for col in df.columns]

    # Optional mapping for alternate date column names
    if "Record_Date" not in df.columns:
        if "record_date" in df.columns:
            df.rename(columns={"record_date": "Record_Date"}, inplace=True)
        elif "Date" in df.columns:
            df.rename(columns={"Date": "Record_Date"}, inplace=True)
        elif "RecordDate" in df.columns:
            df.rename(columns={"RecordDate": "Record_Date"}, inplace=True)

    if "Record_Date" not in df.columns:
        raise ValueError(f"Record_Date column not found. Available columns: {list(df.columns)}")

    df["Record_Date"] = pd.to_datetime(df["Record_Date"], dayfirst=True)
    return df

def clean_records(df: pd.DataFrame):
    cleaned = df.copy()

    for col in cleaned.columns:
        if pd.api.types.is_datetime64_any_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].dt.strftime("%Y-%m-%d")

    records = cleaned.to_dict(orient="records")

    safe_records = []
    for row in records:
        safe_row = {}
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                safe_row[key] = None
            elif pd.isna(value):
                safe_row[key] = None
            else:
                safe_row[key] = value
        safe_records.append(safe_row)

    return safe_records

def classify_row(row: pd.Series) -> str:
    if str(row.get("ICU_Required", "No")) == "Yes":
        return "Critical"
    if row.get("Severity_Score", 0) >= 8:
        return "Critical"
    if row.get("Treatment_Urgency", 0) >= 8 and row.get("Vital_Risk_Score", 0) >= 7:
        return "Critical"
    if 5 <= row.get("Severity_Score", 0) <= 7:
        return "Moderate"
    return "Stable"

class AskRequest(BaseModel):
    question: str
    region: Optional[str] = None
    hospital_branch: Optional[str] = None

class TriageInput(BaseModel):
    patient_id: str
    severity_score: int
    treatment_urgency: int
    vital_risk_score: int
    icu_required: str = "No"
    bed_required: str = "No"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/patients")
def get_patients():
    df = load_data()
    return clean_records(df)

@app.get("/patients/latest")
def get_latest_patients():
    df = load_data()
    latest_date = df["Record_Date"].max()
    latest = df[df["Record_Date"] == latest_date].copy()
    return {
        "record_date": str(latest_date.date()),
        "patients": latest.to_dict(orient="records")
    }

@app.get("/patients/{patient_id}/history")
def get_patient_history(patient_id: str):
    df = load_data()
    patient_df = df[df["Patient_ID"] == patient_id].sort_values("Record_Date")
    return {
        "patient_id": patient_id,
        "history": patient_df.to_dict(orient="records")
    }

@app.post("/triage/classify")
def triage_classify(payload: TriageInput):
    row = {
        "Severity_Score": payload.severity_score,
        "Treatment_Urgency": payload.treatment_urgency,
        "Vital_Risk_Score": payload.vital_risk_score,
        "ICU_Required": payload.icu_required,
        "Bed_Required": payload.bed_required,
    }
    group = classify_row(pd.Series(row))
    return {
        "patient_id": payload.patient_id,
        "priority_group": group
    }

@app.get("/dashboard/summary")
async def dashboard_summary():
    df = load_data()
    latest_date = df["Record_Date"].max()
    latest = df[df["Record_Date"] == latest_date].copy()

    kpis = {
        "total_patients": int(latest["Patient_ID"].nunique()),
        "critical": int((latest["Priority_Group"] == "Critical").sum()),
        "moderate": int((latest["Priority_Group"] == "Moderate").sum()),
        "stable": int((latest["Priority_Group"] == "Stable").sum())
    }

    region_counts = (
        latest.groupby("Region")["Patient_ID"]
        .count()
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    critical_patients = latest[
        (latest["Priority_Group"] == "Critical") | (latest["ICU_Required"] == "Yes")
    ][[
        "Patient_ID", "Primary_Condition", "Priority_Group", "Severity_Score",
        "ICU_Required", "Hospital_Branch", "Region"
    ]].to_dict(orient="records")

    payload = {
        "record_date": str(latest_date.date()),
        "kpis": kpis,
        "region_counts": region_counts,
        "critical_patients": critical_patients[:10]
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{LLM_SERVICE_URL}/dashboard-response", json=payload)
        llm_data = response.json()

    return {
        "data": payload,
        "llm": llm_data
    }

@app.post("/actions/critical-patients")
async def critical_patients_action():
    df = load_data()
    latest_date = df["Record_Date"].max()
    latest = df[df["Record_Date"] == latest_date].copy()

    subset = latest[
        (latest["Priority_Group"] == "Critical") |
        (latest["ICU_Required"] == "Yes") |
        ((latest["Severity_Score"] >= 8) & (latest["Treatment_Urgency"] >= 8))
    ]

    records = subset.to_dict(orient="records")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{LLM_SERVICE_URL}/summarize",
            json={"action": "critical_patients", "records": records}
        )
        llm_data = response.json()

    return {
        "count": len(records),
        "patients": records,
        "llm": llm_data
    }

@app.post("/actions/discharge-candidates")
async def discharge_candidates_action():
    df = load_data()
    latest_date = df["Record_Date"].max()
    latest = df[df["Record_Date"] == latest_date].copy()

    subset = latest[
        (latest["Priority_Group"] == "Stable") &
        (latest["Discharge_Priority"] == "High") &
        (latest["Lab_Test_Pending"] == "No") &
        (latest["Bed_Required"] == "No") &
        (latest["ICU_Required"] == "No")
    ]

    records = subset.to_dict(orient="records")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{LLM_SERVICE_URL}/recommend",
            json={"action": "discharge_candidates", "records": records}
        )
        llm_data = response.json()

    return {
        "count": len(records),
        "patients": records,
        "llm": llm_data
    }

@app.post("/ask")
async def ask_assistant(payload: AskRequest):
    df = load_data()
    latest_date = df["Record_Date"].max()
    latest = df[df["Record_Date"] == latest_date].copy()

    if payload.region:
        latest = latest[latest["Region"] == payload.region]
    if payload.hospital_branch:
        latest = latest[latest["Hospital_Branch"] == payload.hospital_branch]

    records = latest.to_dict(orient="records")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{LLM_SERVICE_URL}/summarize",
            json={
                "action": "manager_question",
                "question": payload.question,
                "records": records[:50]
            }
        )
        llm_data = response.json()

    return {
        "question": payload.question,
        "record_count": len(records),
        "llm": llm_data
    }