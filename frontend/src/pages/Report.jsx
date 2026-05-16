import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "../styles/report.css";

const Report = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const {
    videoName = "Uploaded Image",
    status = "Unknown",
    confidence = "N/A",
    decision_source,
    gradcam_url,
    radar_url,
    bar_plot_url,
    watermark
  } = location.state?.videoDetails || {};

  const handleDownloadTxt = () => {
    const reportText = `
Deepfake Detection Report
--------------------------------
File Name       : ${videoName}
Analysis Date   : ${new Date().toLocaleDateString()}
Prediction      : ${status}
Confidence      : ${confidence}%
Decision Source : ${decision_source || "N/A"}

Watermark Signal:
  Probability   : ${watermark?.probability ?? "N/A"}
  Detected      : ${watermark?.detected ?? "N/A"}

Visual Evidence:
  Grad-CAM      : ${gradcam_url || "N/A"}
  Radar Plot    : ${radar_url || "N/A"}
  Bar Plot      : ${bar_plot_url || "N/A"}
--------------------------------
`;

    const blob = new Blob([reportText], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "Deepfake_Report.txt";
    link.click();
  };

  return (
    <div className="report-container">
      <h2 className="report-title">Deepfake Detection Report</h2>

      {/* SUMMARY */}
      <div className="report-summary">
        <p><strong>File:</strong> {videoName}</p>
        <p><strong>Status:</strong> {status}</p>
        <p><strong>Confidence:</strong> {confidence}%</p>
        {decision_source && (
          <p><strong>Decision Source:</strong> {decision_source}</p>
        )}
        {watermark && (
          <p>
            <strong>Watermark:</strong>{" "}
            {watermark.detected ? "🔴 STRONG" : "🟢 CLEAN"} ({watermark.probability})
          </p>
        )}
      </div>

      {/* VISUALS */}
      <div className="report-visual-grid">
        {gradcam_url && (
          <div className="report-card">
            <h4>Grad-CAM</h4>
            <img src={gradcam_url} alt="Grad-CAM" />
          </div>
        )}

        {radar_url && (
          <div className="report-card">
            <h4>Signal Radar</h4>
            <img src={radar_url} alt="Radar Plot" />
          </div>
        )}
      </div>

      {bar_plot_url && (
        <div className="report-card full-width">
          <h4>Model & Ensemble Scores</h4>
          <img src={bar_plot_url} alt="Bar Plot" />
        </div>
      )}

      {/* ACTIONS */}
      <div className="report-actions">
        <button onClick={handleDownloadTxt}>Download Report (TXT)</button>
        <button onClick={() => navigate("/dashboard")}>Back to Dashboard</button>
      </div>
    </div>
  );
};

export default Report;
