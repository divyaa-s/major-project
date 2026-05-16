"""
Standalone Flask Web App for Testing Temporal Analysis
No Django required - lightweight test server

Usage:
    pip install flask
    python test_webapp.py
    
Then open: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
import os
import sys
from pathlib import Path
import tempfile
import numpy as np
import json

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Import the full hybrid analyzer (temporal + CNN ensemble + quality forensics)
from hybrid_vid_improved import HybridVideoAnalyzer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.json_encoder = NumpyEncoder  # Use custom encoder

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Temporal Analysis Test - Standalone</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 800px;
            width: 100%;
            padding: 40px;
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9em;
        }

        .upload-section {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 20px;
        }

        .upload-section:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }

        .upload-icon {
            font-size: 60px;
            margin-bottom: 20px;
        }

        input[type="file"] {
            display: none;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 16px;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.2s;
            margin: 10px;
        }

        .btn:hover {
            transform: scale(1.05);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .settings {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
            padding: 20px;
            background: #f8f9ff;
            border-radius: 10px;
        }

        .setting-item label {
            display: block;
            margin-bottom: 5px;
            color: #666;
            font-size: 0.9em;
        }

        .setting-item input {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
        }

        .loading {
            text-align: center;
            padding: 30px;
            display: none;
        }

        .loading.active {
            display: block;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .results {
            margin-top: 30px;
            display: none;
        }

        .results.active {
            display: block;
        }

        .score-card {
            background: #f8f9ff;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
        }

        .score-header {
            font-size: 1.5em;
            margin-bottom: 15px;
            color: #333;
        }

        .score-value {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }

        .score-value.low { color: #4CAF50; }
        .score-value.medium { color: #FF9800; }
        .score-value.high { color: #f44336; }

        .interpretation {
            font-size: 1.1em;
            padding: 15px;
            background: white;
            border-radius: 8px;
            margin: 10px 0;
        }

        .details-section {
            margin-top: 20px;
        }

        .detail-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #2196F3;
        }

        .detail-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }

        .detail-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }

        .detail-item {
            padding: 10px;
            background: #f8f9ff;
            border-radius: 5px;
        }

        .detail-label {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 3px;
        }

        .detail-value {
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
        }

        .error {
            background: #ffebee;
            color: #c62828;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }

        .error.active {
            display: block;
        }

        .file-info {
            margin: 15px 0;
            color: #666;
            font-size: 0.9em;
        }

        .video-preview {
            max-width: 100%;
            border-radius: 10px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Deepfake Detection</h1>
        <p class="subtitle">Hybrid analysis: Temporal + CNN Ensemble (EfficientNet · Xception · ViT · ConvNeXt) + Quality Forensics</p>

        <div class="upload-section" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">🎬</div>
            <h3>Click to upload a video</h3>
            <p style="margin-top: 10px; color: #666;">
                Supported: MP4, AVI, MOV, MKV, WEBM
            </p>
            <input type="file" id="fileInput" accept="video/*">
        </div>

        <div class="file-info" id="fileInfo"></div>

        <video id="videoPreview" class="video-preview" controls style="display: none;"></video>

        <div class="settings">
            <div class="setting-item">
                <label>Max Frames to Analyze:</label>
                <input type="number" id="maxFrames" value="30" min="10" max="100">
            </div>
            <div class="setting-item">
                <label>Skip Frames (process every Nth):</label>
                <input type="number" id="skipFrames" value="2" min="0" max="10">
            </div>
        </div>

        <div style="text-align: center;">
            <button class="btn" id="analyzeBtn" onclick="analyzeVideo()" disabled>
                Analyze Video
            </button>
        </div>

        <div class="loading" id="loadingSection">
            <div class="spinner"></div>
            <p>Running full hybrid analysis...</p>
            <p style="color: #666; font-size: 0.9em; margin-top: 10px;">
                Temporal + CNN Ensemble + Quality Forensics. May take 1–3 minutes.
            </p>
        </div>

        <div class="error" id="errorSection"></div>

        <div class="results" id="resultsSection">

            <!-- FINAL VERDICT -->
            <div class="score-card" id="verdictCard">
                <div class="score-header">🏆 Final Verdict</div>
                <div class="score-value" id="verdictLabel">-</div>
                <div style="font-size:1.1em; margin:6px 0;" id="verdictConfidence">-</div>
                <div class="interpretation" id="verdictInterpretation">-</div>
                <div style="margin-top:12px; padding:10px; background:white; border-radius:5px; font-size:0.85em;">
                    <strong>Decision Source:</strong> <span id="decisionSource">-</span>
                </div>
            </div>

            <!-- COMPONENT SCORES BAR -->
            <div class="score-card" style="border-left-color:#764ba2;">
                <div class="score-header">📊 Component Scores</div>
                <div id="componentScores"></div>
            </div>

            <div class="details-section">

                <!-- TEMPORAL -->
                <div class="detail-card">
                    <div class="detail-title">⏱️ Temporal Analysis</div>
                    <div class="interpretation" id="overallInterpretation">-</div>
                    <div style="margin-top:10px; font-size:0.85em; color:#666;">
                        Weights: Blink <span id="blinkWeight">-</span> | Flow <span id="flowWeight">-</span> | Landmark <span id="landmarkWeight">-</span>
                    </div>
                    <div class="detail-grid" style="margin-top:12px;">
                        <div class="detail-item">
                            <div class="detail-label">Score</div>
                            <div class="detail-value" id="temporalScore">-</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Decision</div>
                            <div class="detail-value" id="temporalDecision">-</div>
                        </div>
                    </div>
                </div>

                <!-- BLINK -->
                <div class="detail-card">
                    <div class="detail-title">👁️ Blink Analysis</div>
                    <div class="interpretation" id="blinkInterpretation">-</div>
                    <div class="detail-grid" id="blinkDetails"></div>
                </div>

                <!-- OPTICAL FLOW -->
                <div class="detail-card">
                    <div class="detail-title">🌊 Optical Flow Analysis</div>
                    <div class="interpretation" id="flowInterpretation">-</div>
                    <div class="detail-grid" id="flowDetails"></div>
                </div>

                <!-- LANDMARKS -->
                <div class="detail-card">
                    <div class="detail-title">📍 Landmark Stability</div>
                    <div class="interpretation" id="landmarkInterpretation">-</div>
                    <div class="detail-grid" id="landmarkDetails"></div>
                </div>

                <!-- CNN ENSEMBLE -->
                <div class="detail-card" style="border-left-color:#9c27b0;">
                    <div class="detail-title">🤖 CNN Ensemble — Per-Model Scores</div>
                    <div style="font-size:0.85em; color:#666; margin-bottom:12px;">
                        EfficientNet-B3 · Xception · ViT · ConvNeXt (equal weights)
                    </div>
                    <div id="cnnModelScores">-</div>
                    <div style="margin-top:14px; font-size:0.85em; color:#666;" id="cnnMeta"></div>
                    <div id="cnnDisagreement" style="margin-top:10px;"></div>
                </div>

                <!-- QUALITY FORENSICS -->
                <div class="detail-card" style="border-left-color:#ff9800;">
                    <div class="detail-title">🔬 Quality Forensics</div>
                    <div class="interpretation" id="qualityInterpretation">-</div>
                    <div class="detail-grid" id="qualityDetails"></div>
                </div>

                <!-- VIDEO INFO -->
                <div class="detail-card">
                    <div class="detail-title">📹 Video Information</div>
                    <div class="detail-grid" id="videoInfo"></div>
                </div>

            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        document.getElementById('fileInput').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                
                // Show file info
                const fileInfo = document.getElementById('fileInfo');
                fileInfo.textContent = `Selected: ${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`;
                
                // Show video preview
                const video = document.getElementById('videoPreview');
                video.src = URL.createObjectURL(selectedFile);
                video.style.display = 'block';
                
                // Enable analyze button
                document.getElementById('analyzeBtn').disabled = false;
                
                // Hide previous results
                document.getElementById('resultsSection').classList.remove('active');
                document.getElementById('errorSection').classList.remove('active');
            }
        });

        async function analyzeVideo() {
            if (!selectedFile) {
                showError('Please select a video file first');
                return;
            }

            const loadingSection = document.getElementById('loadingSection');
            const resultsSection = document.getElementById('resultsSection');
            const errorSection = document.getElementById('errorSection');

            loadingSection.classList.add('active');
            resultsSection.classList.remove('active');
            errorSection.classList.remove('active');

            const formData = new FormData();
            formData.append('video', selectedFile);
            formData.append('max_frames', document.getElementById('maxFrames').value);
            formData.append('skip_frames', document.getElementById('skipFrames').value);

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
            } catch (error) {
                showError('Analysis failed: ' + error.message);
            } finally {
                loadingSection.classList.remove('active');
            }
        }

        function displayResults(data) {
            const resultsSection = document.getElementById('resultsSection');
            resultsSection.classList.add('active');

            // ── Final Verdict ──────────────────────────────────────────────
            const label      = data.label || 'Unknown';
            const finalScore = data.final_score ?? 0;
            const confidence = data.confidence ?? 0;

            const verdictEl = document.getElementById('verdictLabel');
            verdictEl.textContent = label;
            verdictEl.style.color = label === 'Fake' ? '#f44336' : '#4CAF50';
            // Set card border color based on label (can be overridden below for edge cases)
            document.getElementById('verdictCard').style.borderLeftColor =
                label === 'Fake' ? '#f44336' : '#4CAF50';

            document.getElementById('verdictConfidence').textContent =
                `Confidence: ${(confidence * 100).toFixed(1)}%  |  Final Score: ${finalScore.toFixed(4)}`;

            let verdict_text = '';
            if (data.needs_manual_review) {
                verdict_text = '⚠️ MANUAL REVIEW REQUIRED — ' + (data.review_reason || '');
                document.getElementById('verdictCard').style.borderLeftColor = '#ff9800';
            } else if (label === 'Real') {
                // Label is Real regardless of raw score — trust the fusion decision
                if (finalScore >= 0.65) {
                    verdict_text = '🟡 Classified as Real, but signals are mixed — review recommended.';
                    document.getElementById('verdictCard').style.borderLeftColor = '#ff9800';
                } else {
                    verdict_text = '✅ Likely authentic — no significant manipulation signals detected.';
                    document.getElementById('verdictCard').style.borderLeftColor = '#4CAF50';
                }
            } else {
                // Label is Fake
                if (finalScore >= 0.70) {
                    verdict_text = '❌ High confidence fake — multiple strong manipulation signals.';
                } else if (finalScore >= 0.50) {
                    verdict_text = '⚠️ Suspicious — evidence of manipulation present.';
                } else {
                    verdict_text = '🟡 Borderline — further review suggested.';
                }
            }
            document.getElementById('verdictInterpretation').textContent = verdict_text;
            document.getElementById('decisionSource').textContent = data.decision_source || 'N/A';

            // ── Component Score Bars ───────────────────────────────────────
            const compScores = [
                { label: 'Temporal Score', score: data.temporal_score ?? 0, color: '#667eea' },
                { label: 'CNN Ensemble',   score: data.cnn_score     ?? 0, color: '#9c27b0' },
                { label: 'Quality Score',  score: data.quality_score ?? 0, color: '#ff9800' },
                { label: 'Final Score',    score: finalScore,               color: label === 'Fake' ? '#f44336' : '#4CAF50' },
            ];
            document.getElementById('componentScores').innerHTML = compScores.map(c => `
                <div style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.9em; margin-bottom:3px;">
                        <span>${c.label}</span><span style="font-weight:bold;">${c.score.toFixed(4)}</span>
                    </div>
                    <div style="background:#e0e0e0; border-radius:4px; height:10px;">
                        <div style="width:${(c.score*100).toFixed(1)}%; background:${c.color}; height:10px; border-radius:4px; transition:width 0.5s;"></div>
                    </div>
                </div>`).join('');

            // ── Temporal ──────────────────────────────────────────────────
            const temp = data.temporal_analysis || {};
            document.getElementById('overallInterpretation').textContent = temp.interpretation || '-';
            document.getElementById('temporalScore').textContent = (temp.temporal_consistency_score ?? 0).toFixed(4);
            document.getElementById('temporalDecision').textContent = temp.decision_source || '-';

            const weights = temp.weights_used || {};
            document.getElementById('blinkWeight').textContent   = weights.blink        ? (weights.blink * 100).toFixed(0) + '%'        : '-';
            document.getElementById('flowWeight').textContent    = weights.optical_flow ? (weights.optical_flow * 100).toFixed(0) + '%' : '-';
            document.getElementById('landmarkWeight').textContent= weights.landmark     ? (weights.landmark * 100).toFixed(0) + '%'     : '-';

            const blink = temp.blink_analysis || {};
            document.getElementById('blinkInterpretation').textContent = blink.interpretation || '-';
            document.getElementById('blinkDetails').innerHTML = `
                <div class="detail-item"><div class="detail-label">Score</div><div class="detail-value">${(blink.score??0).toFixed(4)}</div></div>
                <div class="detail-item"><div class="detail-label">Blinks Detected</div><div class="detail-value">${blink.blink_count ?? '-'}</div></div>
                <div class="detail-item"><div class="detail-label">Avg Interval</div><div class="detail-value">${blink.avg_interval != null ? blink.avg_interval.toFixed(2) + ' frames' : '-'}</div></div>`;

            const flow = temp.optical_flow_analysis || {};
            document.getElementById('flowInterpretation').textContent = flow.interpretation || '-';
            document.getElementById('flowDetails').innerHTML = `
                <div class="detail-item"><div class="detail-label">Score</div><div class="detail-value">${(flow.score??0).toFixed(4)}</div></div>
                <div class="detail-item"><div class="detail-label">Variance</div><div class="detail-value">${flow.variance != null ? flow.variance.toFixed(4) : (flow.flow_inconsistency ?? '-')}</div></div>`;

            const landmark = temp.landmark_stability || {};
            document.getElementById('landmarkInterpretation').textContent = landmark.interpretation || '-';
            document.getElementById('landmarkDetails').innerHTML = `
                <div class="detail-item"><div class="detail-label">Score</div><div class="detail-value">${(landmark.score??0).toFixed(4)}</div></div>
                <div class="detail-item"><div class="detail-label">Frames Analyzed</div><div class="detail-value">${landmark.frames_analyzed ?? '-'}</div></div>`;

            // ── CNN Ensemble ───────────────────────────────────────────────
            const cnn         = data.cnn_analysis || {};
            const ensemble    = cnn.ensemble || {};
            const perModel    = ensemble.per_model_avg || {};
            const modelNames  = Object.keys(perModel);

            if (modelNames.length === 0) {
                document.getElementById('cnnModelScores').innerHTML = `
                    <div style="background:#fff3e0; border:1px solid #ff9800; border-radius:8px; padding:14px; color:#e65100;">
                        <strong>⚠️ NO CNN MODELS LOADED</strong><br><br>
                        All models failed to load. CNN score is neutral (0.5).<br>
                        Check the server console for <code>❌ FAILED</code> lines at startup.<br><br>
                        <strong>Common causes:</strong>
                        <ul style="margin:8px 0 0 18px; line-height:1.8;">
                            <li>Wrong path in <code>MODEL_CONFIGS</code> in <code>hybrid_vid_improved.py</code></li>
                            <li><code>num_classes</code> mismatch (saved with 1, loaded with 2)</li>
                            <li>Wrong <code>model_name</code> (architecture mismatch)</li>
                        </ul>
                    </div>`;
            } else {
                const scores  = Object.values(perModel);
                const maxDiff = scores.length >= 2 ? Math.max(...scores) - Math.min(...scores) : 0;

                document.getElementById('cnnModelScores').innerHTML = modelNames.map(name => {
                    const score  = perModel[name];
                    const pct    = (score * 100).toFixed(1);
                    const color  = score >= 0.80 ? '#f44336' : score >= 0.60 ? '#ff9800' : score >= 0.40 ? '#2196F3' : '#4CAF50';
                    const verdict= score >= 0.80 ? '⚠️ FAKE signal' : score >= 0.60 ? '🟡 Suspicious' : score >= 0.40 ? '🔵 Borderline' : '✅ Real signal';
                    const extreme= (score < 0.05 || score > 0.95) ? ' <span style="color:#f44336;font-size:0.8em;">[EXTREME]</span>' : '';
                    return `
                    <div style="margin-bottom:14px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.9em; margin-bottom:4px;">
                            <span style="font-weight:bold; font-family:monospace;">${name}</span>
                            <span>${verdict}${extreme} &nbsp;<strong>${score.toFixed(4)}</strong></span>
                        </div>
                        <div style="background:#e0e0e0; border-radius:4px; height:12px;">
                            <div style="width:${pct}%; background:${color}; height:12px; border-radius:4px; transition:width 0.5s;"></div>
                        </div>
                    </div>`;
                }).join('') + `
                    <div style="margin-top:10px; padding-top:10px; border-top:1px solid #e0e0e0;">
                        <div style="display:flex; justify-content:space-between; font-size:0.9em; margin-bottom:4px;">
                            <span style="font-weight:bold;">WEIGHTED ENSEMBLE</span>
                            <strong>${(ensemble.avg_fake_probability ?? 0).toFixed(4)}</strong>
                        </div>
                        <div style="background:#e0e0e0; border-radius:4px; height:12px;">
                            <div style="width:${((ensemble.avg_fake_probability??0)*100).toFixed(1)}%; background:#9c27b0; height:12px; border-radius:4px;"></div>
                        </div>
                    </div>`;

                document.getElementById('cnnMeta').textContent =
                    `Frames analyzed: ${ensemble.frames_analyzed ?? '-'}  |  Models loaded: ${modelNames.length}/4`;

                if (maxDiff > 0.50) {
                    document.getElementById('cnnDisagreement').innerHTML =
                        `<div style="background:#fff3e0; border-left:4px solid #ff9800; padding:10px; border-radius:4px; font-size:0.85em;">
                            ⚠️ <strong>High model disagreement:</strong> ${maxDiff.toFixed(4)} spread between models — treat result with caution.
                         </div>`;
                } else if (maxDiff > 0.30) {
                    document.getElementById('cnnDisagreement').innerHTML =
                        `<div style="font-size:0.85em; color:#888; margin-top:4px;">
                            🟡 Moderate disagreement: ${maxDiff.toFixed(4)} spread
                         </div>`;
                }
            }

            // ── Quality Forensics ──────────────────────────────────────────
            const quality = data.quality_analysis || {};
            document.getElementById('qualityInterpretation').textContent = quality.interpretation || '-';
            document.getElementById('qualityDetails').innerHTML = `
                <div class="detail-item"><div class="detail-label">Avg Mismatch</div><div class="detail-value">${(quality.avg_quality_mismatch??0).toFixed(4)}</div></div>
                <div class="detail-item"><div class="detail-label">Max Mismatch</div><div class="detail-value">${(quality.max_quality_mismatch??0).toFixed(4)}</div></div>
                <div class="detail-item"><div class="detail-label">Frames Analyzed</div><div class="detail-value">${quality.frames_analyzed ?? '-'}</div></div>`;

            // ── Video Info ────────────────────────────────────────────────
            const info = temp.video_info || {};
            document.getElementById('videoInfo').innerHTML = `
                <div class="detail-item"><div class="detail-label">Total Frames</div><div class="detail-value">${info.total_frames ?? '-'}</div></div>
                <div class="detail-item"><div class="detail-label">Processed Frames</div><div class="detail-value">${info.processed_frames ?? '-'}</div></div>
                <div class="detail-item"><div class="detail-label">Faces Detected</div><div class="detail-value">${info.faces_detected ?? '-'}</div></div>
                <div class="detail-item"><div class="detail-label">FPS</div><div class="detail-value">${info.fps != null ? info.fps.toFixed(2) : '-'}</div></div>
                <div class="detail-item"><div class="detail-label">Duration</div><div class="detail-value">${info.duration != null ? info.duration.toFixed(2) + 's' : '-'}</div></div>`;
        }

        function showError(message) {
            const errorSection = document.getElementById('errorSection');
            errorSection.textContent = '❌ ' + message;
            errorSection.classList.add('active');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the HTML interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle video analysis"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get parameters
    max_frames = int(request.form.get('max_frames', 30))
    skip_frames = int(request.form.get('skip_frames', 2))
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        video_file.save(tmp_file.name)
        temp_path = tmp_file.name
    
    try:
        # Run full hybrid analysis (temporal + CNN ensemble + quality forensics)
        analyzer = HybridVideoAnalyzer(
            num_keyframes=max_frames,
            temporal_max_frames=max_frames,
            temporal_skip=skip_frames
        )
        results = analyzer.analyze_video(temp_path)

        # Convert all NumPy types to native Python types
        results = json.loads(json.dumps(results, cls=NumpyEncoder))

        return jsonify(results)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: {error_details}")  # Print to console for debugging
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔍 HYBRID DEEPFAKE DETECTION - WEB SERVER")
    print("="*70)
    print("\n✅ Server starting...")
    print("📱 Open in browser: http://localhost:5000")
    print("\n💡 Make sure hybrid_vid_improved.py, temporal_analysis.py,")
    print("   and quality_forensic.py are in the same folder!")
    print("\n⚠️  CNN model load status will print here on first request.")
    print("\nPress Ctrl+C to stop the server\n")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000)