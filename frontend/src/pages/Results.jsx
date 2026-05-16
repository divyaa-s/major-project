import React, { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "../styles/results.css";

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    console.log("RESULTS PAGE STATE:", location.state);
  }, [location.state]);

  const {
    prediction,
    confidence,
    decision_source,
    gradcam_url,
    radar_url,
    bar_plot_url,
    watermark   // ← ADD THIS
  } = location.state || {};

  const handleReportClick = () => {
  navigate("/report", {
    state: {
      videoDetails: {
        videoName: "Uploaded Image",
        status: prediction,
        confidence: confidence ? (confidence * 100).toFixed(2) : "N/A",

        // 🔥 THESE WERE NEVER PASSED
        decision_source,
        watermark,

        gradcam_url,
        radar_url,
        bar_plot_url
      }
    }
  });
};


  return (
    <div className="results-wrapper">
      <div className="results-container">
        <h2>Deepfake Detection Results</h2>

        <p><strong>Prediction:</strong> {prediction ?? "N/A"}</p>
        <p><strong>Confidence:</strong> {confidence ? (confidence * 100).toFixed(2) : "N/A"}%</p>

        {decision_source && (
          <p className="decision-source">
            <strong>Decision Source:</strong> {decision_source}
          </p>
        )}
        {watermark?.probability !== undefined && (
        <div className="watermark-indicator">
          <strong>Watermark Signal:</strong>{" "}
          {watermark.probability > 0.5 ? "🔴 STRONG" :
          watermark.probability > 0.12 ? "🟡 WEAK" : "🟢 CLEAN"}
          {" "}({watermark.probability.toFixed(2)})
        </div>
      )}

        {/* ───────────── VISUALIZATIONS ───────────── */}
        <div className="visualizations">

          {!gradcam_url && !radar_url && !bar_plot_url && (
            <p style={{ color: "#999", textAlign: "center" }}>
              No visual explanations received from backend.
            </p>
          )}

          {gradcam_url && (
            <div className="viz-card">
              <h4>Grad-CAM (CNN Attention)</h4>
              <img src={gradcam_url} alt="Grad-CAM" className="viz-image" />
            </div>
          )}

          {radar_url && (
            <div className="viz-card">
              <h4>Signal Radar (CNN + Forensics + Watermark)</h4>
              <img src={radar_url} alt="Radar Plot" className="viz-image" />
            </div>
          )}

          {bar_plot_url && (
            <div className="viz-card full-width">
              <h4>Model & Ensemble Scores</h4>
              <img src={bar_plot_url} alt="Bar Plot" className="viz-image" />
            </div>
          )}

        </div>

        <button className="report" onClick={handleReportClick}>
          Reports
        </button>
      </div>
    </div>
  );
};

export default Results;
