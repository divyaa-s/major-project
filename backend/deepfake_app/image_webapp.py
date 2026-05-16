"""
image_webapp.py — Stacking Ensemble Meta-Learner Pipeline
===================================================================
A single-file Flask web application for the Image Deepfake Pipeline.
Features: CNN Ensemble, Invisible Watermarks, Meta-Learner Stacking, Visual Diagnostics.
"""

from flask import Flask, render_template_string, request, jsonify, url_for
import os, time, logging, uuid, hashlib
from werkzeug.utils import secure_filename

# =====================================================================
# CONFIGURATION
# =====================================================================
USE_REAL_BACKEND = True 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("image_webapp")

if USE_REAL_BACKEND:
    try:
        from deepfake_detection import generate_gradcam_and_ensemble_predict
        log.info("✅ Real backend loaded successfully.")
    except ImportError as e:
        log.error(f"❌ Failed to load backend. Ensure deepfake_detection.py is available. Error: {e}")
        USE_REAL_BACKEND = False


# =====================================================================
# FRONTEND HTML TEMPLATE
# =====================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deepfake Detection — Image Meta-Learner</title>
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
    --effb3:    #ff6b9d;
    --xception: #a29bfe;
    --vit:      #fdcb6e;
    --convnext: #74b9ff;
    --watermark:#00c9a7;
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
img.preview { width:100%; border-radius:var(--radius); margin-bottom:14px; display:none; object-fit: cover; }
.btn { width:100%; padding:13px; background:var(--accent); color:#fff; border:none; border-radius:var(--radius); font-family:var(--font); font-size:0.9rem; font-weight:700; letter-spacing:1px; cursor:pointer; transition:opacity 0.2s; }
.btn:disabled { opacity:0.35; cursor:not-allowed; }
.btn:not(:disabled):hover { opacity:0.85; }
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
.detail-grid { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:8px; margin-top:10px; }
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
.tag { display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-weight:700; margin-left:6px; }
.tag-fake   { background:rgba(255,71,87,0.15);  color:var(--fake); }
.tag-real   { background:rgba(46,213,115,0.15); color:var(--real); }
.tag-sus    { background:rgba(255,165,2,0.15);  color:var(--warn); }
.formula-box { background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:12px 14px; font-size:0.75rem; color:var(--muted); margin-top:10px; line-height:1.7; font-style:italic; }
.visual-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:14px; margin-top:14px; }
.visual-box { background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:10px; text-align:center; }
.visual-box img { width:100%; height:200px; object-fit:contain; border-radius:4px; margin-top:8px; background:#ffffff; border:1px solid var(--border); }
</style>
</head>
<body>

<div style="max-width:1400px; margin:0 auto 20px;">
    <h1>⬡ Image Deepfake Detection</h1>
    <p class="subtitle">Stacking Ensemble Meta-Learner · CNN Fusion </p>
</div>

<div class="layout">
    <div>
        <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
            <div class="icon">🖼️</div>
            <strong>Drop image or click to upload</strong>
            <p>PNG · JPG · WEBP · max 10MB</p>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        <img id="imagePreview" class="preview" alt="Preview">
        
        <button class="btn" id="analyzeBtn" onclick="analyzeImage()" disabled>ANALYZE IMAGE</button>
        
        <div class="formula-box" style="margin-top:14px;">
            <strong style="color:var(--text);">Stacking Ensemble Logic (Meta-Learner v4)</strong><br>
            Analyzes CNN spatial anomalies, and Watermark frequencies.<br>
            Threshold = 0.50 · Accuracy = 97.6%
        </div>
    </div>

    <div>
        <div class="loading" id="loadingSection">
            <div class="spinner"></div>
            <p>Running Meta-Learner Inference...</p>
            <p>Extracting Forensics & Sensor Noise Fingerprints</p>
        </div>

        <div class="results" id="resultsSection">
            <div class="card" id="verdictCard">
                <div class="card-title">Final Verdict</div>
                <div class="verdict-label" id="verdictLabel">—</div>
                <div class="verdict-meta" id="verdictMeta">—</div>
                <div class="verdict-interp" id="verdictInterp">—</div>
                <div class="decision-src" id="decisionSrc">—</div>
            </div>

            <div class="card">
                <div class="card-title">Meta-Learner Inputs & Component Scores</div>
                <div id="signalBars"></div>
                <div class="detail-grid" style="margin-top:14px;">
                    <div class="detail-item">
                        <div class="detail-label">Base CNN Avg</div>
                        <div class="detail-value" id="cnnAvg">—</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Watermark</div>
                        <div class="detail-value" id="wmProb">—</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Final Output</div>
                        <div class="detail-value" id="finalMeta" style="color:var(--accent);">—</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Visual Diagnostics</div>
                <div class="visual-grid">
                    <div class="visual-box">
                        <div class="detail-label">GradCAM (Spatial Anomaly)</div>
                        <img id="imgGradcam" src="" alt="GradCAM will appear here">
                    </div>
                    <div class="visual-box">
                        <div class="detail-label">Signal Radar Plot</div>
                        <img id="imgRadar" src="" alt="Radar Plot will appear here">
                    </div>
                    <div class="visual-box">
                        <div class="detail-label">Ensemble Breakdown</div>
                        <img id="imgBarchart" src="" alt="Bar Chart will appear here">
                    </div>
                </div>
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
    const img = document.getElementById('imagePreview');
    img.src = URL.createObjectURL(file);
    img.style.display = 'block';
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('resultsSection').classList.remove('active');
    zone.querySelector('strong').textContent = file.name;
}

// ==========================================
// API CALL TO FLASK BACKEND
// ==========================================
async function analyzeImage() {
    if (!selectedFile) return;
    document.getElementById('loadingSection').classList.add('active');
    document.getElementById('resultsSection').classList.remove('active');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            displayResults(data);
        }
    } catch (error) {
        alert('Server communication failed. Check console.');
        console.error(error);
    } finally {
        document.getElementById('loadingSection').classList.remove('active');
    }
}

function scoreTag(s) {
    if (s >= 0.75) return '<span class="tag tag-fake">FAKE</span>';
    if (s >= 0.50) return '<span class="tag tag-sus">SUSPICIOUS</span>';
    return '<span class="tag tag-real">REAL</span>';
}

function displayResults(data) {
    document.getElementById('resultsSection').classList.add('active');

    const label = data.label;
    const finalScore = data.final_fake_probability;
    
    // VERDICT BOX
    const vl = document.getElementById('verdictLabel');
    vl.textContent = label;
    vl.className = 'verdict-label ' + (label === 'Fake' ? 'verdict-fake' : 'verdict-real');

    document.getElementById('verdictCard').style.borderColor =
        label === 'Fake' ? 'rgba(255,71,87,0.4)' : 'rgba(46,213,115,0.4)';

    document.getElementById('verdictMeta').textContent =
        `Confidence: ${(data.confidence*100).toFixed(1)}%  ·  Score: ${finalScore.toFixed(4)}  ·  Threshold: 0.50`;

    document.getElementById('verdictInterp').textContent = 
        label === 'Fake' 
        ? '⚠️ High confidence AI-generated imagery detected by Meta-Learner logic.'
        : '✅ Image appears authentic. No significant generation artifacts found.';
        
    document.getElementById('decisionSrc').textContent = 'Decision source: ' + data.decision_source;



    const signals = [
        { name: 'Watermark (DCT)', score: data.watermark_probability, color: 'var(--watermark)' },
        { name: 'EfficientNet-B3', score: data.models.efficientnet_b3.prob_fake, color: 'var(--effb3)' },
        { name: 'Xception', score: data.models.xception.prob_fake, color: 'var(--xception)' },
        { name: 'ViT-Small', score: data.models.vit.prob_fake, color: 'var(--vit)' },
        { name: 'ConvNeXt-Small', score: data.models.convnext.prob_fake, color: 'var(--convnext)' }
    ];

    document.getElementById('signalBars').innerHTML = signals.map(s => {
        return `
        <div class="signal-row">
            <div class="signal-header">
                <span class="signal-name">${s.name}</span>
                <span class="signal-val">${s.score.toFixed(4)} ${scoreTag(s.score)}</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:${(s.score*100).toFixed(1)}%;background:${s.color};"></div>
            </div>
        </div>`;
    }).join('');

    // STATS GRID
    document.getElementById('cnnAvg').textContent = data.ensemble_fake_probability.toFixed(4);
    document.getElementById('wmProb').textContent = data.watermark_probability.toFixed(4);
    document.getElementById('finalMeta').textContent = data.final_fake_probability.toFixed(4);
    
    // VISUAL DIAGNOSTICS
    if (data.gradcam_url) document.getElementById('imgGradcam').src = data.gradcam_url;
    if (data.radar_url) document.getElementById('imgRadar').src = data.radar_url;
    if (data.barchart_url) document.getElementById('imgBarchart').src = data.barchart_url;
}
</script>
</body>
</html>
"""

# =====================================================================
# FLASK ROUTES
# =====================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    unique_filename = f"upload_{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    log.info(f"Received file: {unique_filename} | Hash: {file_hash[:8]}")

    try:
        if USE_REAL_BACKEND:
            class DummyRequest:
                def build_absolute_uri(self, uri):
                    return uri

            result = generate_gradcam_and_ensemble_predict(DummyRequest(), filepath)
            
            print("\n" + "="*60)
            print(f"🎯 INFERENCE COMPLETED: {unique_filename}")
            print(f"File MD5 Hash : {file_hash}")
            print(f"Final Label   : {result['label']}")
            print(f"Final Score   : {result['final_fake_probability']:.4f}")
            print("-" * 60)
            for model_name, metrics in result['models'].items():
                print(f" > {model_name:15} -> {metrics['prob_fake']:.4f}")
            print("="*60 + "\n")
            
            return jsonify(result)
            
        else:
            time.sleep(1.5)
            return jsonify({"error": "Simulation mode activated by mistake."})
            
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  IMAGE META-LEARNER PIPELINE")
    print(f"  REAL BACKEND ENABLED: {USE_REAL_BACKEND}")
    print("=" * 70)
    print("  Running on: http://127.0.0.1:5000")
    print("=" * 70 + "\n")
    app.run(debug=True, port=5000)