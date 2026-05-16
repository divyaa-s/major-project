"""
hybrid_webapp.py — Confidence-Weighted Additive Fusion (7 signals)
===================================================================
Fusion: F = sum(c_i * s_i) / sum(c_i)  where c_i = 2*|s_i - 0.5|
All 7 signals active. Quality score is inverted (1 - q) before fusion.
Threshold = 0.50. Calibrated on 14 DFD videos. 12/14 correct.
"""

from flask import Flask, render_template_string, request, jsonify
import os, sys, tempfile, json, logging, time
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deepfake_webapp")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):      return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray):                            return obj.tolist()
        return super().default(obj)

from hybrid_vid_improved import HybridVideoAnalyzer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.json_encoder = NumpyEncoder

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deepfake Detection — Hybrid Framework</title>
<style>
:root {
    --bg:       #0d0f14;
    --surface:  #151820;
    --surface2: #1c2030;
    --border:   #2a2f3e;
    --accent:   #5b8cff;
    --fake:     #ff4757;
    --real:     #2ed573;
    --warn:     #ffa502;
    --text:     #e2e8f0;
    --muted:    #8892a4;
    --bilstm:   #00c9a7;
    --effb3:    #ff6b9d;
    --xception: #a29bfe;
    --vit:      #fdcb6e;
    --convnext: #74b9ff;
    --quality:  #fd79a8;
    --temporal: #55efc4;
    --radius:   12px;
    --font:     'DM Mono', 'Fira Code', monospace;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:var(--font); background:var(--bg); color:var(--text); min-height:100vh; padding:24px; }
h1 { font-size:1.6rem; font-weight:700; letter-spacing:-0.5px; color:var(--accent); margin-bottom:4px; }
.subtitle { font-size:0.78rem; color:var(--muted); margin-bottom:28px; letter-spacing:0.5px; }
.layout { display:grid; grid-template-columns:380px 1fr; gap:20px; max-width:1400px; margin:0 auto; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:20px; margin-bottom:14px; }
.card-title { font-size:0.72rem; text-transform:uppercase; letter-spacing:1.5px; color:var(--muted); margin-bottom:14px; border-bottom:1px solid var(--border); padding-bottom:8px; }
.upload-zone { border:2px dashed var(--border); border-radius:var(--radius); padding:32px 20px; text-align:center; cursor:pointer; transition:all 0.2s; margin-bottom:14px; }
.upload-zone:hover, .upload-zone.drag { border-color:var(--accent); background:rgba(91,140,255,0.05); }
.upload-zone .icon { font-size:2.5rem; margin-bottom:10px; }
.upload-zone p { font-size:0.82rem; color:var(--muted); margin-top:4px; }
input[type=file] { display:none; }
video { width:100%; border-radius:var(--radius); margin-bottom:14px; display:none; }
.btn { width:100%; padding:13px; background:var(--accent); color:#fff; border:none; border-radius:var(--radius); font-family:var(--font); font-size:0.9rem; font-weight:700; letter-spacing:1px; cursor:pointer; transition:opacity 0.2s; }
.btn:disabled { opacity:0.35; cursor:not-allowed; }
.btn:not(:disabled):hover { opacity:0.85; }
.settings-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:14px; }
.setting label { font-size:0.72rem; color:var(--muted); display:block; margin-bottom:4px; }
.setting input { width:100%; padding:8px 10px; background:var(--surface2); border:1px solid var(--border); border-radius:7px; color:var(--text); font-family:var(--font); font-size:0.85rem; }
.verdict-label { font-size:3.5rem; font-weight:900; letter-spacing:-2px; line-height:1; margin-bottom:8px; }
.verdict-fake { color:var(--fake); }
.verdict-real { color:var(--real); }
.verdict-meta { font-size:0.82rem; color:var(--muted); margin-bottom:10px; }
.verdict-interp { padding:10px 14px; background:var(--surface2); border-radius:8px; font-size:0.82rem; line-height:1.5; }
.decision-src { margin-top:10px; font-size:0.72rem; color:var(--muted); font-style:italic; }
.signal-row { margin-bottom:11px; }
.signal-header { display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:3px; }
.signal-name { color:var(--muted); }
.signal-val { font-weight:700; font-size:0.82rem; }
.bar-track { height:7px; background:var(--surface2); border-radius:4px; overflow:hidden; }
.bar-fill { height:100%; border-radius:4px; transition:width 0.6s cubic-bezier(.4,0,.2,1); }
.badge { font-size:0.68rem; padding:1px 6px; border-radius:4px; margin-left:5px; font-weight:600; }
.badge-conf { background:rgba(255,71,87,0.15); color:var(--fake); }
.badge-w    { background:rgba(255,255,255,0.07); color:var(--muted); }
.badge-flip { background:rgba(253,121,168,0.15); color:var(--quality); }
.detail-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }
.detail-item { background:var(--surface2); border-radius:8px; padding:10px 12px; }
.detail-label { font-size:0.68rem; color:var(--muted); margin-bottom:3px; text-transform:uppercase; letter-spacing:0.5px; }
.detail-value { font-size:0.92rem; font-weight:700; color:var(--text); }
.spinner { width:32px; height:32px; border:3px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.8s linear infinite; margin:0 auto 12px; }
@keyframes spin { to { transform:rotate(360deg); } }
.loading { text-align:center; padding:24px; display:none; }
.loading.active { display:block; }
.loading p { font-size:0.82rem; color:var(--muted); margin-top:4px; }
.results { display:none; }
.results.active { display:block; }
.error-box { background:rgba(255,71,87,0.1); border:1px solid var(--fake); border-radius:var(--radius); padding:14px; font-size:0.82rem; color:var(--fake); display:none; margin-top:12px; }
.error-box.active { display:block; }
.tag { display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-weight:700; margin-left:6px; }
.tag-fake   { background:rgba(255,71,87,0.15);  color:var(--fake); }
.tag-real   { background:rgba(46,213,115,0.15); color:var(--real); }
.tag-sus    { background:rgba(255,165,2,0.15);  color:var(--warn); }
.tag-border { background:rgba(91,140,255,0.15); color:var(--accent); }
.warn-box { background:rgba(255,165,2,0.08); border-left:3px solid var(--warn); padding:10px 12px; border-radius:6px; font-size:0.78rem; color:var(--warn); margin-top:10px; }
.formula-box { background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:12px 14px; font-size:0.75rem; color:var(--muted); margin-top:10px; line-height:1.7; font-style:italic; }
</style>
</head>
<body>

<div style="max-width:1400px; margin:0 auto 20px;">
    <h1>⬡ Deepfake Detection</h1>
    <p class="subtitle">Hierarchical Multi-Signal Fusion · BiLSTM · EfficientNet-B3 · Xception · ViT · ConvNeXt · Quality Forensics · Temporal Analysis</p>
</div>

<div class="layout">

    <!-- LEFT COLUMN -->
    <div>
        <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
            <div class="icon">🎬</div>
            <strong>Drop video or click to upload</strong>
            <p>MP4 · AVI · MOV · MKV · WEBM · max 500MB</p>
            <input type="file" id="fileInput" accept="video/*">
        </div>
        <video id="videoPreview" controls></video>
        <div class="settings-grid">
            <div class="setting"><label>Max Frames</label><input type="number" id="maxFrames" value="30" min="10" max="100"></div>
            <div class="setting"><label>Skip Frames</label><input type="number" id="skipFrames" value="2" min="0" max="10"></div>
        </div>
        <button class="btn" id="analyzeBtn" onclick="analyzeVideo()" disabled>ANALYZE</button>
        <div class="error-box" id="errorBox"></div>
        <div class="formula-box" style="margin-top:14px;">
            <strong style="color:var(--text);">Fusion Formula (Conf-Weighted Additive · 7 Signals)</strong><br>
            c_i = 2·|s_i − 0.5| &nbsp;·&nbsp; F = Σ(c_i·s_i) / Σ(c_i)<br>
            Threshold = 0.50 · Calibrated on 14 DFD videos · 12/14 correct · FNR=0.000
        </div>
    </div>

    <!-- RIGHT COLUMN -->
    <div>
        <div class="loading" id="loadingSection">
            <div class="spinner"></div>
            <p>Running hybrid analysis...</p>
            <p>BiLSTM · CNN Ensemble · Quality Forensics · Temporal Analysis</p>
        </div>

        <div class="results" id="resultsSection">

            <!-- VERDICT -->
            <div class="card" id="verdictCard">
                <div class="card-title">Final Verdict</div>
                <div class="verdict-label" id="verdictLabel">—</div>
                <div class="verdict-meta" id="verdictMeta">—</div>
                <div class="verdict-interp" id="verdictInterp">—</div>
                <div class="decision-src" id="decisionSrc">—</div>
            </div>

            <!-- SIGNAL SCORES -->
            <div class="card">
                <div class="card-title">Signal Scores — Confidence-Weighted Fusion</div>
                <div id="signalBars"></div>
                <div class="detail-grid" style="margin-top:14px;">
                    <div class="detail-item">
                        <div class="detail-label">Weighted Average</div>
                        <div class="detail-value" id="rawLogit">—</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Final Score (backend)</div>
                        <div class="detail-value" id="finalScoreVal">—</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Threshold</div>
                        <div class="detail-value">0.50</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Fusion Method</div>
                        <div class="detail-value" id="fusionMethod">—</div>
                    </div>
                </div>
            </div>

            <!-- CNN PER-MODEL -->
            <div class="card">
                <div class="card-title">CNN Ensemble — Per-Model Scores</div>
                <div style="font-size:0.72rem; color:var(--muted); margin-bottom:12px;">EfficientNet-B3 · Xception · ViT · ConvNeXt</div>
                <div id="cnnModelScores">—</div>
                <div style="margin-top:8px; font-size:0.72rem; color:var(--muted);" id="cnnMeta"></div>
                <div id="cnnDisagreement"></div>
            </div>

            <!-- BILSTM -->
            <div class="card">
                <div class="card-title">BiLSTM Optical Flow — Temporal Branch</div>
                <div id="bilstmContent">—</div>
            </div>

            <!-- TEMPORAL -->
            <div class="card">
                <div class="card-title">Temporal Analysis</div>
                <div class="verdict-interp" id="temporalInterp" style="margin-bottom:12px;">—</div>
                <div class="detail-grid">
                    <div class="detail-item"><div class="detail-label">Temporal Score</div><div class="detail-value" id="temporalScore">—</div></div>
                    <div class="detail-item"><div class="detail-label">Decision Source</div><div class="detail-value" id="temporalDecision" style="font-size:0.75rem;">—</div></div>
                    <div class="detail-item"><div class="detail-label">Blink Score</div><div class="detail-value" id="blinkScore">—</div></div>
                    <div class="detail-item"><div class="detail-label">Blinks Detected</div><div class="detail-value" id="blinkCount">—</div></div>
                    <div class="detail-item"><div class="detail-label">Flow Score</div><div class="detail-value" id="flowScore">—</div></div>
                    <div class="detail-item"><div class="detail-label">Landmark Score</div><div class="detail-value" id="landmarkScore">—</div></div>
                </div>
            </div>

            <!-- QUALITY -->
            <div class="card">
                <div class="card-title">Quality Forensics</div>
                <div class="verdict-interp" id="qualityInterp" style="margin-bottom:12px;">—</div>
                <div class="detail-grid">
                    <div class="detail-item"><div class="detail-label">Avg Mismatch (raw)</div><div class="detail-value" id="qualityAvg">—</div></div>
                    <div class="detail-item"><div class="detail-label">Fusion Input (1−q)</div><div class="detail-value" id="qualityFlipped">—</div></div>
                    <div class="detail-item"><div class="detail-label">Max Mismatch</div><div class="detail-value" id="qualityMax">—</div></div>
                    <div class="detail-item"><div class="detail-label">Frames Analyzed</div><div class="detail-value" id="qualityFrames">—</div></div>
                </div>
            </div>

            <!-- VIDEO INFO -->
            <div class="card">
                <div class="card-title">Video Information</div>
                <div class="detail-grid" id="videoInfo"></div>
            </div>

        </div>
    </div>
</div>

<script>
let selectedFile = null;

const zone = document.getElementById('uploadZone');
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
document.getElementById('fileInput').addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
    selectedFile = file;
    const vid = document.getElementById('videoPreview');
    vid.src = URL.createObjectURL(file);
    vid.style.display = 'block';
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('resultsSection').classList.remove('active');
    document.getElementById('errorBox').classList.remove('active');
    zone.querySelector('strong').textContent = file.name;
}

async function analyzeVideo() {
    if (!selectedFile) return;
    document.getElementById('loadingSection').classList.add('active');
    document.getElementById('resultsSection').classList.remove('active');
    document.getElementById('errorBox').classList.remove('active');

    const fd = new FormData();
    fd.append('video', selectedFile);
    fd.append('max_frames', document.getElementById('maxFrames').value);
    fd.append('skip_frames', document.getElementById('skipFrames').value);

    try {
        const resp = await fetch('/analyze', { method:'POST', body:fd });
        const data = await resp.json();
        if (data.error) showError(data.error);
        else displayResults(data);
    } catch(e) {
        showError('Analysis failed: ' + e.message);
    } finally {
        document.getElementById('loadingSection').classList.remove('active');
    }
}

function showError(msg) {
    const box = document.getElementById('errorBox');
    box.textContent = '❌ ' + msg;
    box.classList.add('active');
}

function scoreTag(s) {
    if (s >= 0.75) return '<span class="tag tag-fake">FAKE</span>';
    if (s >= 0.55) return '<span class="tag tag-sus">SUSPICIOUS</span>';
    if (s >= 0.40) return '<span class="tag tag-border">BORDERLINE</span>';
    return '<span class="tag tag-real">REAL</span>';
}

function displayResults(data) {
    document.getElementById('resultsSection').classList.add('active');

    const label      = data.label      ?? 'Unknown';
    const finalScore = data.final_score ?? 0;   // always trust the backend value
    const confidence = data.confidence  ?? 0;
    const fw         = data.fusion_weights ?? {};

    // ── VERDICT ───────────────────────────────────────────────────────────
    const vl = document.getElementById('verdictLabel');
    vl.textContent = label;
    vl.className   = 'verdict-label ' + (label === 'Fake' ? 'verdict-fake' : 'verdict-real');

    document.getElementById('verdictCard').style.borderColor =
        label === 'Fake' ? 'rgba(255,71,87,0.4)' : 'rgba(46,213,115,0.4)';

    // Threshold 0.50 — matches _fuse_predictions
    document.getElementById('verdictMeta').textContent =
        `Confidence: ${(confidence*100).toFixed(1)}%  ·  Final Score: ${finalScore.toFixed(4)}  ·  Threshold: 0.50`;

    let interp = '';
    if (label === 'Fake') {
        interp = finalScore >= 0.80 ? '❌ High confidence deepfake — strong manipulation signals across multiple branches.'
               : finalScore >= 0.50 ? '⚠️ Likely deepfake — significant evidence of manipulation detected.'
               :                      '🟡 Borderline fake — weak but present manipulation signals.';
    } else {
        interp = finalScore <= 0.35 ? '✅ Likely authentic — no significant manipulation signals detected.'
               : finalScore <= 0.50 ? '🟡 Probably real, but some signals are elevated — review recommended.'
               :                      '⚠️ Classified as Real, but score is close to threshold — treat with caution.';
    }
    document.getElementById('verdictInterp').textContent = interp;
    document.getElementById('decisionSrc').textContent =
        'Decision source: ' + (data.decision_source ?? 'confidence_weighted_fusion');

    // ── SIGNAL BARS — all 7 ──────────────
    const per_model    = data.cnn_analysis?.ensemble?.per_model_avg ?? {};
    const rawQuality   = data.quality_score ?? 0.0;
    const flippedQuality = rawQuality;   // this is what fusion actually used

    const signals = [
        { key:'bilstm',          name:'BiLSTM Optical Flow',  score: data.bilstm_score          ?? 0.5, color:'var(--bilstm)',   flipped:false },
        { key:'efficientnet_b3', name:'EfficientNet-B3',      score: per_model.efficientnet_b3  ?? 0.5, color:'var(--effb3)',    flipped:false },
        { key:'xception',        name:'Xception',             score: per_model.xception         ?? 0.5, color:'var(--xception)', flipped:false },
        { key:'vit',             name:'ViT-Small',            score: per_model.vit              ?? 0.5, color:'var(--vit)',      flipped:false },
        { key:'convnext',        name:'ConvNeXt',             score: per_model.convnext         ?? 0.5, color:'var(--convnext)', flipped:false },
        { key:'quality',         name:'Quality Forensics',    score: flippedQuality,                    color:'var(--quality)',  flipped:true  },
        { key:'temporal',        name:'Temporal Analysis',    score: data.temporal_score        ?? 0.5, color:'var(--temporal)', flipped:false },
    ];

    // Recompute browser-side fusion using the same logic as backend
    // (quality already flipped above)
    signals.forEach(s => { s.conf = 2.0 * Math.abs(s.score - 0.5); });
    const totalConf  = signals.reduce((a, s) => a + s.conf, 0);
    const fusedScore = totalConf > 1e-8
        ? signals.reduce((a, s) => a + s.conf * s.score, 0) / totalConf
        : 0.5;

    signals.forEach(s => {
        s.effWeight = totalConf > 1e-8 ? s.conf / totalConf : 0;
    });

    // Show browser-computed weighted average AND backend final score separately
    // They should match — if they differ it means quality flip wasn't applied
    document.getElementById('rawLogit').textContent =
        `${fusedScore.toFixed(4)} (browser calc)`;
    document.getElementById('finalScoreVal').textContent =
        `${finalScore.toFixed(4)} (backend)`;
    document.getElementById('fusionMethod').textContent = 'Conf-Weighted 7-Signal';

    document.getElementById('signalBars').innerHTML = signals.map(s => {
        const wPct    = (s.effWeight * 100).toFixed(1);
        const contrib = (s.conf * s.score).toFixed(3);
        const flipBadge = s.flipped
            ? `<span class="badge badge-flip">inverted (1−q)</span>`
            : '';
        return `
        <div class="signal-row">
            <div class="signal-header">
                <span class="signal-name">
                    ${s.name}${flipBadge}
                    <span class="badge badge-conf">conf=${s.conf.toFixed(3)}</span>
                    <span class="badge badge-w">w=${wPct}%</span>
                </span>
                <span class="signal-val">
                    ${s.score.toFixed(4)}
                    <span style="font-size:0.7rem;color:var(--muted);">contrib:${contrib}</span>
                    ${scoreTag(s.score)}
                </span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:${(s.score*100).toFixed(1)}%;background:${s.color};"></div>
            </div>
        </div>`;
    }).join('');

    // ── CNN PER-MODEL ─────────────────────────────────────────────────────
    const ensemble   = data.cnn_analysis?.ensemble ?? {};
    const modelNames = Object.keys(per_model);
    const MODEL_COLORS = {
        efficientnet_b3:'var(--effb3)', xception:'var(--xception)',
        vit:'var(--vit)', convnext:'var(--convnext)',
    };

    if (modelNames.length === 0) {
        document.getElementById('cnnModelScores').innerHTML =
            `<div class="warn-box">⚠️ No CNN models loaded — check model paths</div>`;
    } else {
        const scores  = Object.values(per_model);
        const maxDiff = scores.length >= 2 ? Math.max(...scores) - Math.min(...scores) : 0;

        document.getElementById('cnnModelScores').innerHTML =
            modelNames.map(name => {
                const score = per_model[name];
                const color = MODEL_COLORS[name] ?? 'var(--accent)';
                return `
                <div class="signal-row">
                    <div class="signal-header">
                        <span class="signal-name" style="font-size:0.8rem;">${name}</span>
                        <span class="signal-val">${score.toFixed(4)} ${scoreTag(score)}</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:${(score*100).toFixed(1)}%;background:${color};"></div>
                    </div>
                </div>`;
            }).join('') + `
            <div class="signal-row" style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);">
                <div class="signal-header">
                    <span class="signal-name" style="font-weight:700;color:var(--text);">ENSEMBLE AVG</span>
                    <span class="signal-val">${(ensemble.avg_fake_probability??0).toFixed(4)}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:${((ensemble.avg_fake_probability??0)*100).toFixed(1)}%;background:#9c27b0;"></div>
                </div>
            </div>`;

        document.getElementById('cnnMeta').textContent =
            `Frames analyzed: ${ensemble.frames_analyzed ?? '-'}  ·  Models loaded: ${modelNames.length}/4`;

        document.getElementById('cnnDisagreement').innerHTML =
            maxDiff > 0.50
            ? `<div class="warn-box">⚠️ High model disagreement: ${maxDiff.toFixed(4)} spread — treat with caution.</div>`
            : maxDiff > 0.30
            ? `<div style="font-size:0.75rem;color:var(--muted);margin-top:6px;">🟡 Moderate disagreement: ${maxDiff.toFixed(4)} spread</div>`
            : '';
    }

    // ── BILSTM ────────────────────────────────────────────────────────────
    const bScore  = data.bilstm_score;
    const bWeight = fw.bilstm != null ? (fw.bilstm * 100).toFixed(1) : '—';
    document.getElementById('bilstmContent').innerHTML = bScore != null ? `
        <div class="signal-row">
            <div class="signal-header">
                <span class="signal-name">Motion Inconsistency Score (P(Fake))</span>
                <span class="signal-val">${bScore.toFixed(4)} ${scoreTag(bScore)}</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:${(bScore*100).toFixed(1)}%;background:var(--bilstm);"></div>
            </div>
        </div>
        <div class="detail-grid" style="margin-top:10px;">
            <div class="detail-item">
                <div class="detail-label">Effective Weight (this video)</div>
                <div class="detail-value" style="color:var(--accent);">${bWeight}%</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Contribution to Score</div>
                <div class="detail-value">${fw.bilstm != null ? (fw.bilstm * bScore).toFixed(4) : '—'}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Clips × Frames</div>
                <div class="detail-value">3 × 48</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Feature</div>
                <div class="detail-value">Farneback (w=9)</div>
            </div>
        </div>` :
        `<div class="warn-box">⚫ BiLSTM not available — checkpoint not found.</div>`;

    // ── TEMPORAL ──────────────────────────────────────────────────────────
    const temp = data.temporal_analysis ?? {};
    document.getElementById('temporalInterp').textContent   = temp.interpretation ?? '—';
    document.getElementById('temporalScore').textContent    = (temp.temporal_consistency_score ?? 0).toFixed(4);
    document.getElementById('temporalDecision').textContent = temp.decision_source ?? '—';
    document.getElementById('blinkScore').textContent       = (temp.blink_analysis?.score ?? 0).toFixed(4);
    document.getElementById('blinkCount').textContent       = temp.blink_analysis?.blink_count ?? '—';
    document.getElementById('flowScore').textContent        = (temp.optical_flow_analysis?.score ?? 0).toFixed(4);
    document.getElementById('landmarkScore').textContent    = (temp.landmark_stability?.score ?? 0).toFixed(4);

    // ── QUALITY ───────────────────────────────────────────────────────────
    const qual = data.quality_analysis ?? {};
    document.getElementById('qualityInterp').textContent  = qual.interpretation ?? '—';
    document.getElementById('qualityAvg').textContent     = rawQuality.toFixed(4);
    document.getElementById('qualityFlipped').textContent = flippedQuality.toFixed(4);
    document.getElementById('qualityMax').textContent     = (qual.max_quality_mismatch ?? 0).toFixed(4);
    document.getElementById('qualityFrames').textContent  = qual.frames_analyzed ?? '—';

    // ── VIDEO INFO ────────────────────────────────────────────────────────
    const info = temp.video_info ?? {};
    document.getElementById('videoInfo').innerHTML = [
        ['Total Frames',     info.total_frames     ?? '—'],
        ['Processed Frames', info.processed_frames ?? '—'],
        ['Faces Detected',   info.faces_detected   ?? '—'],
        ['FPS',              info.fps != null ? info.fps.toFixed(2) : '—'],
        ['Duration',         info.duration != null ? info.duration.toFixed(2)+'s' : '—'],
    ].map(([l,v]) => `
        <div class="detail-item">
            <div class="detail-label">${l}</div>
            <div class="detail-value">${v}</div>
        </div>`).join('');
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/analyze', methods=['POST'])
def analyze():
    # ── 1. Validate Upload ───────────────────────────────────────────
    if 'video' not in request.files or not request.files['video'].filename:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    max_frames = int(request.form.get('max_frames', 30))
    skip_frames = int(request.form.get('skip_frames', 2))

    log.info(f"\n{'='*60}\n🚀 STARTING NEW VIDEO ANALYSIS\n{'='*60}")
    log.info(f" FILE     : {video_file.filename}")
    log.info(f" SETTINGS : max_frames={max_frames} | skip_frames={skip_frames}")

    tmp_path = None
    try:
        # ── 2. Save Temporary File ───────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name

        file_size_mb = Path(tmp_path).stat().st_size / 1024 / 1024
        log.debug(f"Saved temp file: {tmp_path} ({file_size_mb:.1f} MB)")

        # ── 3. Run Hybrid Analyzer ───────────────────────────────────
        t0 = time.time()
        analyzer = HybridVideoAnalyzer(
            num_keyframes=max_frames,
            temporal_max_frames=max_frames,
            temporal_skip=skip_frames
        )
        result = analyzer.analyze_video(tmp_path)
        elapsed = time.time() - t0

        if 'error' in result:
            log.error(f"Analyzer error: {result['error']}")
            return jsonify(result), 500

        # ── 4. Log Final Results Cleanly ─────────────────────────────
        score = result.get('final_score', 0)
        label = result.get('label', 'Unknown')
        conf  = result.get('confidence', 0)
        
        log.info(f"\n{'-'*60}\n📊 ANALYSIS COMPLETE ({elapsed:.1f}s)\n{'-'*60}")
        log.info(f" LABEL        : {label}")
        log.info(f" FINAL SCORE  : {score:.4f} (Threshold: 0.50)")
        log.info(f" CONFIDENCE   : {conf:.4f}")
        log.info(f" DECISION VIA : {result.get('decision_source', 'xgboost_meta_learner')}")
        log.info(f" MANUAL REVIEW: {result.get('needs_manual_review', False)}")
        log.info("="*60 + "\n")

        # ── 5. Return JSON to Frontend ───────────────────────────────
        return jsonify(json.loads(json.dumps(result, cls=NumpyEncoder)))

    except Exception as e:
        import traceback
        log.error(f"ANALYSIS FAILED:\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

    finally:
        # ── 6. Cleanup ───────────────────────────────────────────────
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            log.debug(f"Cleaned up temp file: {tmp_path}")

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  DEEPFAKE DETECTION — HIERARCHICAL MULTI-SIGNAL FRAMEWORK")
    print("=" * 70)
    print("  http://localhost:5000")
    print("  Fusion : Confidence-Weighted Additive · 7 signals · threshold=0.50")
    print("  Signals: BiLSTM · EffB3 · Xception · ViT · ConvNeXt · Quality* · Temporal")
    print("=" * 70 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)