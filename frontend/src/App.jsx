import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer
} from "recharts";

const API_URL = "http://localhost:8000";

function App() {
  const [dashboard, setDashboard] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = async () => {
    const res = await axios.get(`${API_URL}/dashboard/summary`);
    setDashboard(res.data);
  };

  const askAssistant = async () => {
    const res = await axios.post(`${API_URL}/ask`, { question });
    setAnswer(res.data);
  };

  const llm = dashboard?.llm;
  const chart = llm?.charts?.[0]?.data || [];

  return (
    <div style={{ fontFamily: "Arial, sans-serif", padding: 24 }}>
      <h1>{llm?.page_title || "Smart Patient Segmentation Dashboard"}</h1>
      <p>{llm?.summary}</p>

      <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
        {llm?.kpis?.map((kpi, idx) => (
          <div
            key={idx}
            style={{
              border: "1px solid #ddd",
              borderRadius: 12,
              padding: 16,
              minWidth: 180,
              boxShadow: "0 2px 8px rgba(0,0,0,0.06)"
            }}
          >
            <h3>{kpi.label}</h3>
            <p style={{ fontSize: 24, margin: 0 }}>{kpi.value}</p>
          </div>
        ))}
      </div>

      <div style={{ width: "100%", height: 300, marginBottom: 24 }}>
        <h2>Patients by Region</h2>
        <ResponsiveContainer>
          <BarChart data={chart}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginBottom: 24 }}>
        <h2>Alerts</h2>
        {llm?.alerts?.map((alert, idx) => (
          <div
            key={idx}
            style={{
              border: "1px solid #f0c2c2",
              background: "#fff5f5",
              padding: 12,
              marginBottom: 8,
              borderRadius: 8
            }}
          >
            <strong>{alert.patient_id}</strong>: {alert.message}
          </div>
        ))}
      </div>

      <div style={{ marginBottom: 24 }}>
        <h2>Recommendations</h2>
        <ul>
          {llm?.recommendations?.map((rec, idx) => (
            <li key={idx}>{rec}</li>
          ))}
        </ul>
      </div>

      <div style={{ marginTop: 32 }}>
        <h2>Ask Assistant</h2>
        <div style={{ display: "flex", gap: 8 }}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Which patients need immediate attention today?"
            style={{ flex: 1, padding: 12, borderRadius: 8, border: "1px solid #ccc" }}
          />
          <button onClick={askAssistant} style={{ padding: "12px 18px" }}>
            Ask
          </button>
        </div>

        {answer?.llm && (
          <div style={{ marginTop: 16, border: "1px solid #ddd", padding: 16, borderRadius: 8 }}>
            <h3>Assistant Response</h3>
            <p>{answer.llm.summary}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;