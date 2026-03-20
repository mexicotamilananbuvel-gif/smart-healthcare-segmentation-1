import math
from unittest.mock import patch, AsyncMock, MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import (
    app,
    load_data,
    clean_records,
    classify_row,
    AskRequest,
    TriageInput,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers – sample DataFrames
# ---------------------------------------------------------------------------

SAMPLE_CSV = """\
Patient_ID;Record_Date;Age;Gender;Primary_Condition;Severity_Score;Treatment_Urgency;Vital_Risk_Score;Comorbidity_Count;Recent_Admissions_Last_6M;Medication_Adherence;Last_Visit_Days_Ago;Insurance_Type;Care_Plan_Assigned;Doctor_Assigned;Follow_Up_Required;Readmission_Risk;Region;Hospital_Branch;Appointment_Status;Lab_Test_Pending;Bed_Required;ICU_Required;Discharge_Priority;Priority_Group
P001;18/03/2026;72;F;Heart Failure;9;10;9;3;2;68;5;Private;Yes;Dr. Smith;Yes;High;North;Central Hospital;Admitted;Yes;Yes;Yes;Low;Critical
P002;18/03/2026;58;M;Diabetes;6;7;5;2;1;74;14;Government;Yes;Dr. Lee;Yes;Medium;South;West Clinic;Scheduled;Yes;No;No;Medium;Moderate
P003;18/03/2026;34;F;Asthma;3;3;2;0;0;91;30;Private;No;Dr. Patel;No;Low;East;East Medical Center;Completed;No;No;No;High;Stable
P004;17/03/2026;81;M;Chronic Kidney Disease;8;9;8;4;3;61;4;Government;Yes;Dr. Brown;Yes;High;North;Central Hospital;Admitted;Yes;Yes;No;Low;Critical
"""


def _make_df() -> pd.DataFrame:
    """Return a small DataFrame that looks like the real patients.csv."""
    import io

    df = pd.read_csv(io.StringIO(SAMPLE_CSV), sep=";")
    df["Record_Date"] = pd.to_datetime(df["Record_Date"], dayfirst=True)
    return df


# ---------------------------------------------------------------------------
# Unit tests – pure functions
# ---------------------------------------------------------------------------


class TestClassifyRow:
    def test_critical_icu_required(self):
        row = pd.Series(
            {"ICU_Required": "Yes", "Severity_Score": 3, "Treatment_Urgency": 2, "Vital_Risk_Score": 1}
        )
        assert classify_row(row) == "Critical"

    def test_critical_high_severity(self):
        row = pd.Series(
            {"ICU_Required": "No", "Severity_Score": 8, "Treatment_Urgency": 2, "Vital_Risk_Score": 1}
        )
        assert classify_row(row) == "Critical"

    def test_critical_severity_9(self):
        row = pd.Series(
            {"ICU_Required": "No", "Severity_Score": 9, "Treatment_Urgency": 5, "Vital_Risk_Score": 5}
        )
        assert classify_row(row) == "Critical"

    def test_critical_urgency_and_vital(self):
        row = pd.Series(
            {"ICU_Required": "No", "Severity_Score": 4, "Treatment_Urgency": 8, "Vital_Risk_Score": 7}
        )
        assert classify_row(row) == "Critical"

    def test_moderate(self):
        row = pd.Series(
            {"ICU_Required": "No", "Severity_Score": 6, "Treatment_Urgency": 3, "Vital_Risk_Score": 3}
        )
        assert classify_row(row) == "Moderate"

    def test_moderate_severity_5(self):
        row = pd.Series(
            {"ICU_Required": "No", "Severity_Score": 5, "Treatment_Urgency": 3, "Vital_Risk_Score": 3}
        )
        assert classify_row(row) == "Moderate"

    def test_moderate_severity_7(self):
        row = pd.Series(
            {"ICU_Required": "No", "Severity_Score": 7, "Treatment_Urgency": 3, "Vital_Risk_Score": 3}
        )
        assert classify_row(row) == "Moderate"

    def test_stable(self):
        row = pd.Series(
            {"ICU_Required": "No", "Severity_Score": 3, "Treatment_Urgency": 2, "Vital_Risk_Score": 1}
        )
        assert classify_row(row) == "Stable"

    def test_stable_severity_0(self):
        row = pd.Series(
            {"ICU_Required": "No", "Severity_Score": 0, "Treatment_Urgency": 0, "Vital_Risk_Score": 0}
        )
        assert classify_row(row) == "Stable"

    def test_missing_icu_defaults_no(self):
        row = pd.Series({"Severity_Score": 3, "Treatment_Urgency": 2, "Vital_Risk_Score": 1})
        assert classify_row(row) == "Stable"


class TestCleanRecords:
    def test_converts_datetime_to_string(self):
        df = pd.DataFrame({"date_col": pd.to_datetime(["2026-03-18"]), "name": ["test"]})
        result = clean_records(df)
        assert result[0]["date_col"] == "2026-03-18"

    def test_replaces_nan_with_none(self):
        df = pd.DataFrame({"val": [float("nan")], "name": ["test"]})
        result = clean_records(df)
        assert result[0]["val"] is None

    def test_replaces_pd_nat_with_none(self):
        df = pd.DataFrame({"val": [pd.NaT], "name": ["test"]})
        result = clean_records(df)
        assert result[0]["val"] is None

    def test_preserves_normal_values(self):
        df = pd.DataFrame({"val": [42], "name": ["hello"]})
        result = clean_records(df)
        assert result[0]["val"] == 42
        assert result[0]["name"] == "hello"

    def test_multiple_rows(self):
        df = pd.DataFrame({"val": [1, float("nan"), 3]})
        result = clean_records(df)
        assert len(result) == 3
        assert result[0]["val"] == 1
        assert result[1]["val"] is None
        assert result[2]["val"] == 3


class TestLoadData:
    def test_load_data_returns_dataframe(self, tmp_path):
        csv_file = tmp_path / "patients.csv"
        csv_file.write_text(SAMPLE_CSV)
        with patch("app.main.DATA_PATH", csv_file):
            df = load_data()
            assert isinstance(df, pd.DataFrame)
            assert "Record_Date" in df.columns
            assert "Patient_ID" in df.columns

    def test_record_date_is_datetime(self, tmp_path):
        csv_file = tmp_path / "patients.csv"
        csv_file.write_text(SAMPLE_CSV)
        with patch("app.main.DATA_PATH", csv_file):
            df = load_data()
            assert pd.api.types.is_datetime64_any_dtype(df["Record_Date"])

    def test_load_data_raises_on_missing_column(self, tmp_path):
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("col_a;col_b\n1;2\n")
        with patch("app.main.DATA_PATH", csv_file):
            with pytest.raises(ValueError, match="Record_Date column not found"):
                load_data()

    def test_load_data_renames_record_date_variant(self, tmp_path):
        csv_file = tmp_path / "alt.csv"
        csv_file.write_text("Patient_ID;record_date\nP001;18/03/2026\n")
        with patch("app.main.DATA_PATH", csv_file):
            df = load_data()
            assert "Record_Date" in df.columns

    def test_load_data_renames_date_variant(self, tmp_path):
        csv_file = tmp_path / "alt2.csv"
        csv_file.write_text("Patient_ID;Date\nP001;18/03/2026\n")
        with patch("app.main.DATA_PATH", csv_file):
            df = load_data()
            assert "Record_Date" in df.columns

    def test_load_data_renames_recorddate_variant(self, tmp_path):
        csv_file = tmp_path / "alt3.csv"
        csv_file.write_text("Patient_ID;RecordDate\nP001;18/03/2026\n")
        with patch("app.main.DATA_PATH", csv_file):
            df = load_data()
            assert "Record_Date" in df.columns


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestPydanticModels:
    def test_ask_request_defaults(self):
        req = AskRequest(question="test?")
        assert req.question == "test?"
        assert req.region is None
        assert req.hospital_branch is None

    def test_ask_request_with_filters(self):
        req = AskRequest(question="q", region="North", hospital_branch="Central Hospital")
        assert req.region == "North"
        assert req.hospital_branch == "Central Hospital"

    def test_triage_input_defaults(self):
        t = TriageInput(patient_id="P001", severity_score=5, treatment_urgency=4, vital_risk_score=3)
        assert t.icu_required == "No"
        assert t.bed_required == "No"

    def test_triage_input_with_icu(self):
        t = TriageInput(
            patient_id="P001",
            severity_score=9,
            treatment_urgency=10,
            vital_risk_score=9,
            icu_required="Yes",
            bed_required="Yes",
        )
        assert t.icu_required == "Yes"
        assert t.bed_required == "Yes"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestGetPatients:
    @patch("app.main.load_data")
    def test_get_patients(self, mock_load):
        mock_load.return_value = _make_df()
        response = client.get("/patients")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 4


class TestGetLatestPatients:
    @patch("app.main.load_data")
    def test_returns_latest_date_only(self, mock_load):
        mock_load.return_value = _make_df()
        response = client.get("/patients/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["record_date"] == "2026-03-18"
        assert len(data["patients"]) == 3  # P001, P002, P003 on 18/03


class TestGetPatientHistory:
    @patch("app.main.load_data")
    def test_returns_history_for_patient(self, mock_load):
        mock_load.return_value = _make_df()
        response = client.get("/patients/P001/history")
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "P001"
        assert len(data["history"]) == 1

    @patch("app.main.load_data")
    def test_returns_empty_for_unknown_patient(self, mock_load):
        mock_load.return_value = _make_df()
        response = client.get("/patients/UNKNOWN/history")
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "UNKNOWN"
        assert len(data["history"]) == 0


class TestTriageClassify:
    def test_classify_critical(self):
        payload = {
            "patient_id": "P001",
            "severity_score": 9,
            "treatment_urgency": 10,
            "vital_risk_score": 9,
            "icu_required": "Yes",
        }
        response = client.post("/triage/classify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "P001"
        assert data["priority_group"] == "Critical"

    def test_classify_moderate(self):
        payload = {
            "patient_id": "P002",
            "severity_score": 6,
            "treatment_urgency": 3,
            "vital_risk_score": 3,
        }
        response = client.post("/triage/classify", json=payload)
        assert response.status_code == 200
        assert response.json()["priority_group"] == "Moderate"

    def test_classify_stable(self):
        payload = {
            "patient_id": "P003",
            "severity_score": 2,
            "treatment_urgency": 2,
            "vital_risk_score": 1,
        }
        response = client.post("/triage/classify", json=payload)
        assert response.status_code == 200
        assert response.json()["priority_group"] == "Stable"


class TestDashboardSummary:
    @patch("app.main.load_data")
    @patch("app.main.httpx.AsyncClient")
    def test_dashboard_summary_success(self, mock_httpx_cls, mock_load):
        mock_load.return_value = _make_df()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "page_title": "Dashboard",
            "summary": "All ok",
            "kpis": [],
            "alerts": [],
            "patients": [],
            "charts": [],
            "recommendations": [],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client_instance

        response = client.get("/dashboard/summary")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "llm" in data
        assert data["data"]["kpis"]["total_patients"] == 3


class TestDashboardPatientTrends:
    @patch("app.main.load_data")
    def test_patient_trends(self, mock_load):
        mock_load.return_value = _make_df()
        response = client.get("/dashboard/patient-trends")
        assert response.status_code == 200
        data = response.json()
        assert "latest_date" in data
        assert "latest_counts" in data
        assert "total_patients" in data
        assert "trend" in data
        assert data["latest_date"] == "2026-03-18"
        assert data["total_patients"] == 3

    @patch("app.main.load_data")
    def test_patient_trends_without_priority_group_column(self, mock_load):
        df = _make_df()
        df = df.drop(columns=["Priority_Group"])
        mock_load.return_value = df
        response = client.get("/dashboard/patient-trends")
        assert response.status_code == 200
        data = response.json()
        assert "trend" in data


class TestCriticalPatientsAction:
    @patch("app.main.load_data")
    @patch("app.main.httpx.AsyncClient")
    def test_critical_patients(self, mock_httpx_cls, mock_load):
        mock_load.return_value = _make_df()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "summary": "Critical patients identified",
            "recommendations": [],
            "alerts": [],
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client_instance

        response = client.post("/actions/critical-patients")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "patients" in data
        assert "llm" in data
        assert data["count"] >= 1


class TestDischargeCandidatesAction:
    @patch("app.main.load_data")
    @patch("app.main.httpx.AsyncClient")
    def test_discharge_candidates(self, mock_httpx_cls, mock_load):
        mock_load.return_value = _make_df()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "summary": "Discharge candidates",
            "recommendations": [],
            "alerts": [],
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client_instance

        response = client.post("/actions/discharge-candidates")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "patients" in data
        assert "llm" in data


class TestAskAssistant:
    @patch("app.main.load_data")
    @patch("app.main.httpx.AsyncClient")
    def test_ask_without_filters(self, mock_httpx_cls, mock_load):
        mock_load.return_value = _make_df()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "summary": "Answer",
            "recommendations": [],
            "alerts": [],
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client_instance

        response = client.post("/ask", json={"question": "How many patients?"})
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "How many patients?"
        assert "record_count" in data
        assert "llm" in data

    @patch("app.main.load_data")
    @patch("app.main.httpx.AsyncClient")
    def test_ask_with_region_filter(self, mock_httpx_cls, mock_load):
        mock_load.return_value = _make_df()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "summary": "Filtered answer",
            "recommendations": [],
            "alerts": [],
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client_instance

        response = client.post("/ask", json={"question": "Status?", "region": "North"})
        assert response.status_code == 200
        data = response.json()
        assert data["record_count"] == 1  # Only P001 in North on latest date

    @patch("app.main.load_data")
    @patch("app.main.httpx.AsyncClient")
    def test_ask_with_hospital_branch_filter(self, mock_httpx_cls, mock_load):
        mock_load.return_value = _make_df()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "summary": "Branch answer",
            "recommendations": [],
            "alerts": [],
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client_instance

        response = client.post(
            "/ask",
            json={"question": "Status?", "hospital_branch": "West Clinic"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["record_count"] == 1  # Only P002 at West Clinic on latest date
