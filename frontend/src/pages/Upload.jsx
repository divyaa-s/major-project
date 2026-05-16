import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import "../styles/upload.css";

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setUploadStatus(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus("Please select a file first!");
      return;
    }

    setIsUploading(true);
    setUploadStatus(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("http://localhost:8000/api/upload/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("Upload response:", response.data);

      if (response.status === 201 || response.status === 200) {
        // Redirect to results page with response data
        navigate("/results", {
            state: {
              prediction: response.data.prediction,
              confidence: response.data.confidence,
              decision_source: response.data.decision_source,

              // 🔥 VISUALS — MUST MATCH Results.jsx EXACTLY
              gradcam_url: response.data.gradcam_url,
              radar_url: response.data.radar_url,
              bar_plot_url: response.data.bar_plot_url,
              watermark: response.data.watermark   // ← REQUIRED

            },
          });

      }
    } catch (error) {
      console.error("Upload failed:", error);
      setUploadStatus("Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
 };

  return (
    <div className="container">

      <h2>Upload Deepfake Video or Image</h2>

      <input type="file" accept="video/*,image/*" onChange={handleFileChange} />

      <button onClick={handleUpload} disabled={isUploading} className="upload-btn">
        {isUploading ? "Uploading..." : "Upload"}
      </button>

      {uploadStatus && <p className="upload-status">{uploadStatus}</p>}

    </div>
  );
};

export default Upload;
