import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import axios from "axios";
import App from "../App";

jest.mock("axios");

// Mock recharts to avoid rendering issues in test environment
jest.mock("recharts", () => ({
  BarChart: ({ children }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div />,
  XAxis: () => <div />,
  YAxis: () => <div />,
  Tooltip: () => <div />,
  CartesianGrid: () => <div />,
  ResponsiveContainer: ({ children }) => <div>{children}</div>,
}));

const MOCK_DASHBOARD = {
  data: {
    record_date: "2026-03-18",
    kpis: { total_patients: 100, critical: 10, moderate: 30, stable: 60 },
    region_counts: [{ Region: "North", count: 25 }],
    critical_patients: [],
  },
  llm: {
    page_title: "Smart Patient Dashboard",
    summary: "Overview of patient segmentation",
    kpis: [
      { label: "Total Patients", value: 100 },
      { label: "Critical", value: 10 },
    ],
    alerts: [
      { patient_id: "P001", message: "Needs ICU", severity: "high" },
    ],
    patients: [],
    charts: [
      {
        chart_type: "bar",
        title: "By Region",
        data: [
          { label: "North", value: 25 },
          { label: "South", value: 20 },
        ],
      },
    ],
    recommendations: ["Monitor ICU capacity", "Review discharge candidates"],
  },
};

const MOCK_TREND = {
  latest_date: "2026-03-18",
  latest_counts: { Critical: 10, Moderate: 30, Stable: 60 },
  total_patients: 100,
  trend: [
    { Record_Date: "2026-03-17", Critical: 8, Moderate: 28, Stable: 58, Total: 94 },
    { Record_Date: "2026-03-18", Critical: 10, Moderate: 30, Stable: 60, Total: 100 },
  ],
};

const MOCK_ASK_RESPONSE = {
  question: "How many critical patients?",
  record_count: 10,
  llm: {
    summary: "There are 10 critical patients requiring immediate attention.",
    recommendations: [],
    alerts: [],
  },
};

beforeEach(() => {
  jest.clearAllMocks();
});

describe("App Component", () => {
  test("renders loading state before data is fetched", () => {
    axios.get.mockImplementation(() => new Promise(() => {})); // never resolves
    render(<App />);
    // The title fallback is shown when llm is null
    expect(
      screen.getByText("Smart Patient Segmentation Dashboard")
    ).toBeInTheDocument();
  });

  test("renders dashboard title from LLM response", async () => {
    axios.get.mockImplementation((url) => {
      if (url.includes("/dashboard/summary")) {
        return Promise.resolve({ data: MOCK_DASHBOARD });
      }
      if (url.includes("/dashboard/patient-trends")) {
        return Promise.resolve({ data: MOCK_TREND });
      }
      return Promise.reject(new Error("Unknown URL"));
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Smart Patient Dashboard")).toBeInTheDocument();
    });
  });

  test("renders LLM summary text", async () => {
    axios.get.mockImplementation((url) => {
      if (url.includes("/dashboard/summary")) {
        return Promise.resolve({ data: MOCK_DASHBOARD });
      }
      if (url.includes("/dashboard/patient-trends")) {
        return Promise.resolve({ data: MOCK_TREND });
      }
      return Promise.reject(new Error("Unknown URL"));
    });

    render(<App />);

    await waitFor(() => {
      expect(
        screen.getByText("Overview of patient segmentation")
      ).toBeInTheDocument();
    });
  });

  test("renders KPI cards", async () => {
    axios.get.mockImplementation((url) => {
      if (url.includes("/dashboard/summary")) {
        return Promise.resolve({ data: MOCK_DASHBOARD });
      }
      if (url.includes("/dashboard/patient-trends")) {
        return Promise.resolve({ data: MOCK_TREND });
      }
      return Promise.reject(new Error("Unknown URL"));
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Total Patients")).toBeInTheDocument();
      expect(screen.getByText("Critical")).toBeInTheDocument();
      expect(screen.getByText("100")).toBeInTheDocument();
      expect(screen.getByText("10")).toBeInTheDocument();
    });
  });

  test("renders alerts section", async () => {
    axios.get.mockImplementation((url) => {
      if (url.includes("/dashboard/summary")) {
        return Promise.resolve({ data: MOCK_DASHBOARD });
      }
      if (url.includes("/dashboard/patient-trends")) {
        return Promise.resolve({ data: MOCK_TREND });
      }
      return Promise.reject(new Error("Unknown URL"));
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Alerts")).toBeInTheDocument();
      expect(screen.getByText(/P001/)).toBeInTheDocument();
      expect(screen.getByText(/Needs ICU/)).toBeInTheDocument();
    });
  });

  test("renders recommendations list", async () => {
    axios.get.mockImplementation((url) => {
      if (url.includes("/dashboard/summary")) {
        return Promise.resolve({ data: MOCK_DASHBOARD });
      }
      if (url.includes("/dashboard/patient-trends")) {
        return Promise.resolve({ data: MOCK_TREND });
      }
      return Promise.reject(new Error("Unknown URL"));
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Recommendations")).toBeInTheDocument();
      expect(screen.getByText("Monitor ICU capacity")).toBeInTheDocument();
      expect(
        screen.getByText("Review discharge candidates")
      ).toBeInTheDocument();
    });
  });

  test("renders patient trend chart section", async () => {
    axios.get.mockImplementation((url) => {
      if (url.includes("/dashboard/summary")) {
        return Promise.resolve({ data: MOCK_DASHBOARD });
      }
      if (url.includes("/dashboard/patient-trends")) {
        return Promise.resolve({ data: MOCK_TREND });
      }
      return Promise.reject(new Error("Unknown URL"));
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Patient Trend Over Time")).toBeInTheDocument();
    });
  });

  test("renders region chart section", async () => {
    axios.get.mockImplementation((url) => {
      if (url.includes("/dashboard/summary")) {
        return Promise.resolve({ data: MOCK_DASHBOARD });
      }
      if (url.includes("/dashboard/patient-trends")) {
        return Promise.resolve({ data: MOCK_TREND });
      }
      return Promise.reject(new Error("Unknown URL"));
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Patients by Region")).toBeInTheDocument();
    });
  });

  test("renders Ask Assistant section with input and button", async () => {
    axios.get.mockImplementation((url) => {
      if (url.includes("/dashboard/summary")) {
        return Promise.resolve({ data: MOCK_DASHBOARD });
      }
      if (url.includes("/dashboard/patient-trends")) {
        return Promise.resolve({ data: MOCK_TREND });
      }
      return Promise.reject(new Error("Unknown URL"));
    });

    render(<App />);

    expect(screen.getByText("Ask Assistant")).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText(
        "Which patients need immediate attention today?"
      )
    ).toBeInTheDocument();
    expect(screen.getByText("Ask")).toBeInTheDocument();
  });

  test("submits question and displays assistant response", async () => {
    axios.get.mockImplementation((url) => {
      if (url.includes("/dashboard/summary")) {
        return Promise.resolve({ data: MOCK_DASHBOARD });
      }
      if (url.includes("/dashboard/patient-trends")) {
        return Promise.resolve({ data: MOCK_TREND });
      }
      return Promise.reject(new Error("Unknown URL"));
    });

    axios.post.mockResolvedValueOnce({ data: MOCK_ASK_RESPONSE });

    render(<App />);

    const input = screen.getByPlaceholderText(
      "Which patients need immediate attention today?"
    );
    const button = screen.getByText("Ask");

    fireEvent.change(input, { target: { value: "How many critical patients?" } });
    fireEvent.click(button);

    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith(
        "http://localhost:8000/ask",
        { question: "How many critical patients?" }
      );
    });

    await waitFor(() => {
      expect(screen.getByText("Assistant Response")).toBeInTheDocument();
      expect(
        screen.getByText(
          "There are 10 critical patients requiring immediate attention."
        )
      ).toBeInTheDocument();
    });
  });

  test("calls both dashboard and trend APIs on mount", async () => {
    axios.get.mockImplementation((url) => {
      if (url.includes("/dashboard/summary")) {
        return Promise.resolve({ data: MOCK_DASHBOARD });
      }
      if (url.includes("/dashboard/patient-trends")) {
        return Promise.resolve({ data: MOCK_TREND });
      }
      return Promise.reject(new Error("Unknown URL"));
    });

    render(<App />);

    await waitFor(() => {
      expect(axios.get).toHaveBeenCalledWith(
        "http://localhost:8000/dashboard/summary"
      );
      expect(axios.get).toHaveBeenCalledWith(
        "http://localhost:8000/dashboard/patient-trends"
      );
    });
  });

  test("does not render trend chart when trend data is null", () => {
    axios.get.mockImplementation(() => new Promise(() => {}));
    render(<App />);
    expect(
      screen.queryByText("Patient Trend Over Time")
    ).not.toBeInTheDocument();
  });

  test("does not render assistant response when answer is null", () => {
    axios.get.mockImplementation(() => new Promise(() => {}));
    render(<App />);
    expect(screen.queryByText("Assistant Response")).not.toBeInTheDocument();
  });
});
