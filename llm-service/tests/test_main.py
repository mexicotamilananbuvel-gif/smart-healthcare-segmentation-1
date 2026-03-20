import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import (
    app,
    build_prompt,
    call_gemini,
    SYSTEM_PROMPT,
    LLMRequest,
    DashboardRequest,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestLLMRequest:
    def test_defaults(self):
        req = LLMRequest()
        assert req.action is None
        assert req.question is None
        assert req.records == []

    def test_with_values(self):
        req = LLMRequest(
            action="summarize",
            question="How many critical?",
            records=[{"Patient_ID": "P001"}],
        )
        assert req.action == "summarize"
        assert req.question == "How many critical?"
        assert len(req.records) == 1


class TestDashboardRequest:
    def test_required_fields(self):
        req = DashboardRequest(
            record_date="2026-03-18",
            kpis={"total_patients": 10},
            region_counts=[{"Region": "North", "count": 5}],
            critical_patients=[{"Patient_ID": "P001"}],
        )
        assert req.record_date == "2026-03-18"
        assert req.kpis["total_patients"] == 10
        assert len(req.region_counts) == 1
        assert len(req.critical_patients) == 1


# ---------------------------------------------------------------------------
# build_prompt tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_contains_system_prompt(self):
        prompt = build_prompt("test_action", {"key": "value"}, {"result": "string"})
        assert SYSTEM_PROMPT in prompt

    def test_contains_action(self):
        prompt = build_prompt("summarize", {"key": "value"}, {"result": "string"})
        assert "summarize" in prompt

    def test_contains_data(self):
        data = {"patients": [{"id": "P001"}]}
        prompt = build_prompt("action", data, {"result": "string"})
        assert '"P001"' in prompt

    def test_contains_schema(self):
        schema = {"summary": "string", "alerts": []}
        prompt = build_prompt("action", {}, schema)
        assert '"summary"' in prompt
        assert '"alerts"' in prompt


# ---------------------------------------------------------------------------
# call_gemini tests
# ---------------------------------------------------------------------------


class TestCallGemini:
    @patch("app.main.client")
    def test_successful_json_response(self, mock_client):
        mock_response = MagicMock()
        mock_response.text = '{"summary": "All good", "recommendations": []}'
        mock_client.models.generate_content.return_value = mock_response

        result = call_gemini("summarize", {"key": "val"}, {"summary": "string"})
        assert result == {"summary": "All good", "recommendations": []}

    @patch("app.main.client")
    def test_strips_markdown_json_fences(self, mock_client):
        mock_response = MagicMock()
        mock_response.text = '```json\n{"summary": "test"}\n```'
        mock_client.models.generate_content.return_value = mock_response

        result = call_gemini("summarize", {}, {})
        assert result == {"summary": "test"}

    @patch("app.main.client")
    def test_strips_plain_markdown_fences(self, mock_client):
        mock_response = MagicMock()
        mock_response.text = '```\n{"summary": "test"}\n```'
        mock_client.models.generate_content.return_value = mock_response

        result = call_gemini("summarize", {}, {})
        assert result == {"summary": "test"}

    @patch("app.main.client")
    def test_invalid_json_raises_500(self, mock_client):
        mock_response = MagicMock()
        mock_response.text = "this is not json at all"
        mock_client.models.generate_content.return_value = mock_response

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            call_gemini("summarize", {}, {})
        assert exc_info.value.status_code == 500
        assert "Invalid JSON from model" in exc_info.value.detail

    @patch("app.main.client", None)
    def test_missing_client_raises_500(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            call_gemini("summarize", {}, {})
        assert exc_info.value.status_code == 500
        assert "GEMINI_API_KEY is missing" in exc_info.value.detail


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "gemini_configured" in data


class TestSummarizeEndpoint:
    @patch("app.main.call_gemini")
    def test_summarize_success(self, mock_gemini):
        mock_gemini.return_value = {
            "summary": "Test summary",
            "recommendations": ["rec1"],
            "alerts": [],
        }

        response = client.post(
            "/summarize",
            json={
                "action": "critical_patients",
                "records": [{"Patient_ID": "P001"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary"] == "Test summary"
        assert len(data["recommendations"]) == 1

    @patch("app.main.call_gemini")
    def test_summarize_default_action(self, mock_gemini):
        mock_gemini.return_value = {"summary": "ok", "recommendations": [], "alerts": []}

        client.post("/summarize", json={"records": []})
        call_args = mock_gemini.call_args
        assert call_args[0][0] == "summarize"

    @patch("app.main.call_gemini")
    def test_summarize_with_question(self, mock_gemini):
        mock_gemini.return_value = {"summary": "answer", "recommendations": [], "alerts": []}

        response = client.post(
            "/summarize",
            json={
                "action": "manager_question",
                "question": "How many critical patients?",
                "records": [],
            },
        )
        assert response.status_code == 200
        call_args = mock_gemini.call_args
        assert call_args[0][1]["question"] == "How many critical patients?"


class TestRecommendEndpoint:
    @patch("app.main.call_gemini")
    def test_recommend_success(self, mock_gemini):
        mock_gemini.return_value = {
            "summary": "Discharge recommendations",
            "recommendations": ["Discharge P003"],
            "alerts": [],
        }

        response = client.post(
            "/recommend",
            json={
                "action": "discharge_candidates",
                "records": [{"Patient_ID": "P003"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary"] == "Discharge recommendations"

    @patch("app.main.call_gemini")
    def test_recommend_default_action(self, mock_gemini):
        mock_gemini.return_value = {"summary": "ok", "recommendations": [], "alerts": []}

        client.post("/recommend", json={"records": []})
        call_args = mock_gemini.call_args
        assert call_args[0][0] == "recommend"


class TestExplainEndpoint:
    @patch("app.main.call_gemini")
    def test_explain_success(self, mock_gemini):
        mock_gemini.return_value = {
            "summary": "Explanation",
            "recommendations": [],
            "alerts": [{"patient_id": "P001", "message": "High risk", "severity": "high"}],
        }

        response = client.post(
            "/explain",
            json={
                "action": "explain_triage",
                "records": [{"Patient_ID": "P001", "Severity_Score": 9}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary"] == "Explanation"
        assert len(data["alerts"]) == 1

    @patch("app.main.call_gemini")
    def test_explain_default_action(self, mock_gemini):
        mock_gemini.return_value = {"summary": "ok", "recommendations": [], "alerts": []}

        client.post("/explain", json={"records": []})
        call_args = mock_gemini.call_args
        assert call_args[0][0] == "explain"


class TestDashboardResponseEndpoint:
    @patch("app.main.call_gemini")
    def test_dashboard_response_success(self, mock_gemini):
        mock_gemini.return_value = {
            "page_title": "Dashboard",
            "summary": "Overview",
            "kpis": [{"label": "Total", "value": 100}],
            "alerts": [],
            "patients": [],
            "charts": [],
            "recommendations": ["Monitor ICU capacity"],
        }

        response = client.post(
            "/dashboard-response",
            json={
                "record_date": "2026-03-18",
                "kpis": {"total_patients": 100, "critical": 10, "moderate": 30, "stable": 60},
                "region_counts": [{"Region": "North", "count": 25}],
                "critical_patients": [{"Patient_ID": "P001", "Primary_Condition": "Heart Failure"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["page_title"] == "Dashboard"
        assert len(data["recommendations"]) == 1

    @patch("app.main.call_gemini")
    def test_dashboard_response_calls_gemini_with_correct_action(self, mock_gemini):
        mock_gemini.return_value = {
            "page_title": "Dashboard",
            "summary": "ok",
            "kpis": [],
            "alerts": [],
            "patients": [],
            "charts": [],
            "recommendations": [],
        }

        client.post(
            "/dashboard-response",
            json={
                "record_date": "2026-03-18",
                "kpis": {},
                "region_counts": [],
                "critical_patients": [],
            },
        )
        call_args = mock_gemini.call_args
        assert call_args[0][0] == "dashboard_response"
