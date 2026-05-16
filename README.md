# Hybrid Deepfake Detection System

This project is a professional-grade forensic suite designed for high-accuracy deepfake detection in images and videos. By fusing state-of-the-art Deep Learning models with classical forensic signal analysis, it provides a robust defense against increasingly sophisticated AI-generated media.

## 🚀 Key Features

- **Multi-Model CNN Ensemble**: Integrates predictions from four leading architectures:
  - **Vision Transformer (ViT)**
  - **ConvNeXt**
  - **EfficientNet B3**
  - **Xception**
- **Forensic Signal Suite**: Goes beyond simple classification by analyzing:
  - **Frequency Artifacts**: FFT-based analysis to detect upsampling patterns.
  - **GAN Fingerprints**: DCT-based detection of GAN-specific noise.
  - **Color Entropy**: Monitoring anomalies in LAB color space distribution.
  - **Invisible Watermarks**: Dedicated classifier for AI-generated watermarking.
- **Explainable AI (XAI)**: Generates **Grad-CAM** heatmaps to highlight precisely which parts of an image the model finds suspicious.
- **Clinical Dashboard**: A premium React-based interface featuring:
  - **Radar Plots**: Multi-dimensional signal visualization.
  - **Analysis Bar Charts**: Detailed breakdown of individual model confidence.
  - **Diagnostic Trace**: Step-by-step transparency into the final decision logic.
- **Robust Decision Engine**: Smart fusion logic with safety overrides for high-confidence signals and disagreement handling.

## 🛠️ Technology Stack

### Backend
- **Framework**: Django, Django REST Framework
- **Deep Learning**: PyTorch, Torchvision, TIMM (Torch Image Models)
- **Computer Vision**: OpenCV, FaceNet (MTCNN for face extraction)
- **Forensics**: NumPy, SciPy (Entropy/FFT), Joblib
- **Visualization**: Matplotlib (Server-side plot generation)

### Frontend
- **Framework**: React 19, React Router
- **UI Components**: Radix UI, Lucide Icons
- **Styling**: Tailwind CSS
- **Interactions**: Axios for seamless API communication

## 📂 Project Structure

```text
├── backend/                # Django REST API logic
│   ├── deepfake_app/       # Main app views and serializers
│   ├── media/              # Generated Grad-CAMs and Radar plots
│   ├── static/             # Static assets
│   └── manage.py           # Django management script
├── frontend/               # React application
│   ├── src/                # UI components and pages
│   └── package.json        # Frontend dependencies
├── models/                 # Pre-trained model weights (.pth files)
├── invisible_watermark/    # Specialized watermark detection module
├── deepfake_detection.py   # Core analysis & fusion pipeline
└── requirements.txt        # Backend dependencies
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- CUDA-enabled GPU (optional but recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
```

### 4. Running the Application
DeepFocus includes a convenience script to run both servers simultaneously:
```bash
npm start
```
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`

## 📊 Methodology

This system uses a **Hybrid Fusion** approach. Unlike standard detectors that rely solely on CNNs, it balances neural network intuition with mathematical forensics:

1. **Preprocessing**: MTCNN extracts the primary face from the input.
2. **Parallel Processing**: The face is fed into the CNN ensemble and the forensic signal extractors simultaneously.
3. **Watermark Check**: A specialized model checks for invisible AI signatures.
4. **Weighted Fusion**: An ensemble score is calculated based on architecture reliability.
5. **Decision Logic**: If signals disagree (e.g., CNN says Real but Forensics show high GAN noise), the system triggers safety overrides to ensure the most cautious result.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed as part of a Major Project on AI Integrity and Digital Forensics.*
