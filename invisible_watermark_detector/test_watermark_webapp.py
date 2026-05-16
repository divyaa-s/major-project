import streamlit as st
import os
os.makedirs("outputs", exist_ok=True)
import cv2
import numpy as np
import joblib
from features.dct_features import extract_dct_features, extract_frequency_statistics

# ─── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    clf = joblib.load("models/new_watermark_classifier.pkl")
    scaler = joblib.load("models/new_feature_scaler.pkl")
    return clf, scaler

clf, scaler = load_model()

# ─── Config ──────────────────────────────────────────────────────────────────
CONF_VERY_REAL     = 0.25
CONF_LIKELY_REAL   = 0.45
CONF_UNCERTAIN_HIGH = 0.80
CONF_HIGH_AI       = 0.82

# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_why_sentence(ai_prob, decision):
    if "REAL" in decision:
        return "This image shows natural-looking details and imperfections typical of real camera photos."
    elif "UNCERTAIN" in decision:
        return "The patterns are somewhat unusual, but not strong enough to be sure — could be light editing or compression."
    else:
        if ai_prob > 0.90:
            return "The image has unusually smooth and perfect-looking surfaces very common in AI-generated pictures."
        else:
            return "There are signs of unnatural uniformity and reduced fine details often seen in synthetic or watermarked images."

def get_risk_level(ai_prob, decision):
    if "REAL" in decision and ai_prob < CONF_VERY_REAL:
        return "✅ Low risk – likely real"
    elif "REAL" in decision:
        return "⚠️ Medium risk – possibly edited"
    elif "UNCERTAIN" in decision:
        return "⚠️ Medium risk – uncertain"
    else:
        return "🔥 High risk – very likely AI-generated"

def get_confidence_bar(confidence):
    bar_length = 10
    filled = int(bar_length * confidence)
    return "█" * filled + "░" * (bar_length - filled)

def visualize_dct_heatmap(image_path, save_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = np.float32(img)

    # Block-wise DCT
    block_size = 8
    h, w = img.shape
    heatmap = np.zeros_like(img)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img[y:y+block_size, x:x+block_size]
            if block.shape != (block_size, block_size):
                continue

            dct_block = cv2.dct(block)

            # High-frequency energy (bottom-right area)
            hf_energy = np.sum(np.abs(dct_block[4:, 4:]))

            heatmap[y:y+block_size, x:x+block_size] = hf_energy

    # Normalize & colorize
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    cv2.imwrite(save_path, heatmap)


# ─── Main prediction ─────────────────────────────────────────────────────────
def analyze_image(image_path):
    name = os.path.splitext(os.path.basename(image_path))[0]

    dct_feats  = extract_dct_features(image_path)
    freq_stats = extract_frequency_statistics(image_path)

    if dct_feats is None or freq_stats is None:
        return {"error": "Could not process image"}

    combined = np.hstack([dct_feats, freq_stats])
    scaled   = scaler.transform(combined.reshape(1, -1))
    probas   = clf.predict_proba(scaled)[0]
    ai_prob  = probas[1]

    # Decision
    if ai_prob >= CONF_HIGH_AI:
        decision = "WATERMARKED / AI-GENERATED (high confidence)"
        confidence = ai_prob
    elif ai_prob >= CONF_UNCERTAIN_HIGH:
        decision = "LIKELY WATERMARKED / AI (medium confidence)"
        confidence = ai_prob
    elif ai_prob <= CONF_VERY_REAL:
        decision = "REAL (high confidence)"
        confidence = 1 - ai_prob
    elif ai_prob <= CONF_LIKELY_REAL:
        decision = "LIKELY REAL / camera image"
        confidence = 1 - ai_prob
    else:
        decision = "UNCERTAIN / possible editing or weak watermark"
        confidence = 0.50

    # Placeholder for heatmap (replace with real visualize_dct call)
    
    heatmap_path = f"outputs/{name}_heatmap.png"
    visualize_dct_heatmap(image_path, heatmap_path)


    return {
        "filename": name,
        "ai_probability": ai_prob,
        "decision": decision,
        "confidence": confidence,
        "why": get_why_sentence(ai_prob, decision),
        "risk": get_risk_level(ai_prob, decision),
        "bar": get_confidence_bar(confidence),
        "heatmap": heatmap_path
    }


# ─── Streamlit UI ────────────────────────────────────────────────────────────
st.title("Invisible Watermark Detector")
st.write("Upload an image to check if it's likely real or AI-generated / watermarked")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temporarily
    with open("temp_upload.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Analyzing..."):
        result = analyze_image("temp_upload.jpg")
    
    if "error" in result:
        st.error(result["error"])
    else:
        # Main result
        st.subheader(result["risk"])
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Confidence:** {result['confidence']*100:.0f}%")
            st.text(result["bar"])
        with col2:
            st.metric("AI Probability", f"{result['ai_probability']*100:.1f}%")

        st.info(result["why"])

        # Expandable details
        with st.expander("More explanation", expanded=False):
            st.write("• This looks most similar to other AI-generated images (sharp edges, uniform textures).")
            st.write("• Real photos usually have more natural grain, tiny imperfections, and uneven lighting.")
            st.write("• Analogy: It’s like noticing a drawing has perfectly straight lines everywhere — cameras almost never produce that.")
            
            st.markdown("**Technical summary**")
            st.write(f"- AI probability: {result['ai_probability']*100:.1f}%")
            st.write(f"- Heatmap saved: {result['heatmap']}")

        # Show uploaded image + placeholder for heatmap
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
        with col2:
            if os.path.exists(result["heatmap"]):
                st.image(result["heatmap"], caption="Frequency Heatmap (suspicious areas in bright colors)", use_column_width=True)
            else:
                st.info("Heatmap would appear here (implement visualize_dct to generate it)")

def visualize_dct_heatmap(image_path, save_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = np.float32(img)

    # Block-wise DCT
    block_size = 8
    h, w = img.shape
    heatmap = np.zeros_like(img)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img[y:y+block_size, x:x+block_size]
            if block.shape != (block_size, block_size):
                continue

            dct_block = cv2.dct(block)

            # High-frequency energy (bottom-right area)
            hf_energy = np.sum(np.abs(dct_block[4:, 4:]))

            heatmap[y:y+block_size, x:x+block_size] = hf_energy

    # Normalize & colorize
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    cv2.imwrite(save_path, heatmap)


    # Cleanup
    if os.path.exists("temp_upload.jpg"):
        os.remove("temp_upload.jpg")