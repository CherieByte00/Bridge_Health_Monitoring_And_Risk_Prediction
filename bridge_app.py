import os
import base64
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Bridge Health Monitoring System",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Keep these paths aligned with the model files already on your computer.
CRACK_MODEL_PATH = r"C:\Users\USER\bridge_crack_final_model.keras"
APP_DIR = Path(__file__).resolve().parent
COLLAPSE_MODEL_PATH = APP_DIR / "bridge_collapse_model.joblib"
DATASET_PATH = APP_DIR / "bridge_data.csv.xlsx"
BACKGROUND_PATH = APP_DIR / "bridge_bg.jpg"
IMAGE_SIZE = (224, 224)
CRACK_THRESHOLD = 0.452

FEATURE_COLUMNS = [
    "Location",
    "Age (years)",
    "Length (m)",
    "Width (m)",
    "Height (m)",
    "Traffic_Volume (vehicles/day)",
    "Weather_Conditions",
    "Water_Flow_Rate (mÂ³/s)",
    "Stress (MPa)",
    "Strain (%)",
    "Tensile_Strength (MPa)",
    "Rainfall (mm)",
    "Temperature (Â°C)",
    "Humidity (%)",
]


def apply_custom_style():
    if BACKGROUND_PATH.is_file():
        encoded = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode("utf-8")
        background = (
            "background-image: linear-gradient(115deg, rgba(4,25,46,.82), "
            f"rgba(5,39,65,.58)), url('data:image/jpeg;base64,{encoded}');"
        )
    else:
        background = (
            "background-image: radial-gradient(circle at 78% 8%, rgba(34,211,238,.20), transparent 28%),"
            "radial-gradient(circle at 12% 30%, rgba(59,130,246,.24), transparent 30%),"
            "linear-gradient(135deg, #071a2d 0%, #123b59 48%, #0b253d 100%);"
        )
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{ {background} background-position:center; background-size:cover; background-attachment:fixed; }}
        [data-testid="stHeader"] {{ background:rgba(5,24,42,.18); }}
        [data-testid="stMainBlockContainer"] {{ max-width:1450px; padding-top:1rem; }}
        .brandbar {{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:15px 20px; margin-bottom:16px; border:1px solid rgba(255,255,255,.22); border-radius:16px; background:rgba(5,25,43,.78); color:white; }}
        .brand-title {{ font-size:1.3rem; font-weight:800; color:white; }} .brand-sub {{ font-size:.84rem; color:#d7eafa; margin-top:4px; }}
        .hero {{ color:white; padding:12px 5px 18px; }} .hero h1 {{ color:white; font-size:clamp(2rem,4vw,3rem); margin-bottom:8px; }} .hero p {{ color:#e0f2fe; font-size:1.02rem; }}
        .note {{ padding:12px 15px; border-radius:12px; background:rgba(4,28,48,.72); border:1px solid rgba(125,211,252,.35); color:#e5f4ff; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{ background:rgba(255,255,255,.97); border-radius:16px; }}
        [data-testid="stWidgetLabel"] p {{ color:#294b69; font-weight:650; }}
        .stButton>button,.stFormSubmitButton>button {{ border:0; border-radius:9px; color:white; font-weight:750; background:linear-gradient(100deg,#087bd3,#1765ce); min-height:42px; }}
        .footer {{ color:#e0f2fe; text-align:center; padding:18px 0 6px; font-size:.82rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading crack detection model...")
def load_crack_model(path):
    if not os.path.isfile(path):
        return None, f"Crack model file was not found: {path}"
    try:
        from tensorflow.keras.models import load_model
        return load_model(path), None
    except Exception as exc:
        return None, f"Could not load crack model: {exc}"


@st.cache_resource(show_spinner="Loading bridge status model...")
def load_status_model(path):
    if not os.path.isfile(path):
        return None, f"Bridge status model file was not found: {path}"
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, f"Could not load bridge status model: {exc}"


@st.cache_data(show_spinner=False)
def load_dataset(path):
    if not Path(path).is_file():
        return None
    df = pd.read_excel(path)
    return df


def predict_crack(model, uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    # Deliberately matches the original app's preprocessing: raw RGB float32 pixels.
    array = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    output = np.asarray(model.predict(array, verbose=0)).reshape(-1)
    if output.size == 0:
        raise ValueError("The crack model returned an empty prediction.")

    # This assumes a single sigmoid output represents crack probability.
    # For multi-output models, confirm the class order before using this logic.
    crack_probability = float(output[0] if output.size == 1 else output[-1])
    crack_probability = float(np.clip(crack_probability, 0.0, 1.0))
    has_crack = crack_probability >= CRACK_THRESHOLD

    # Display confidence for the predicted class, not always crack probability.
    prediction_confidence = (
        crack_probability if has_crack else 1.0 - crack_probability
    )
    return image, has_crack, prediction_confidence



def predict_status(model, row):
    frame = pd.DataFrame([
        {col: row[col] for col in FEATURE_COLUMNS}
    ])

    prediction = str(model.predict(frame)[0])
    confidence = None
    probabilities = {}

    if hasattr(model, "predict_proba"):
        try:
            raw = model.predict_proba(frame)[0]
            probabilities = {
                str(label): float(prob)
                for label, prob in zip(model.classes_, raw)
            }
            confidence = float(np.max(raw))
        except Exception as exc:
            st.warning(f"Probability error: {exc}")

    return prediction, confidence, probabilities



def show_status_result(status, confidence, source_label=None):
    normal = status.strip().lower()
    if normal == "collapsed":
        st.error("⚠️ Model prediction: Collapsed")
    elif normal == "standing":
        st.success("✓ Model prediction: Standing")
    else:
        st.info(f"Model prediction: {status}")
    if confidence is not None:
        st.metric("Model confidence", f"{confidence * 100:.1f}%")
    if source_label:
        st.caption(f"Dataset label: {source_label}")


apply_custom_style()
st.markdown(
    """
    <div class="brandbar"><div><div class="brand-title">🌉 Bridge Health Monitoring System</div>
    <div class="brand-sub">AI-assisted crack screening and bridge status comparison</div></div><div>● AI DEMO</div></div>
    <div class="hero"><h1>Bridge Health Monitoring System</h1>
    <p>Compare two images and compare Standing/Collapsed parameter examples in one dashboard.</p>
    <div class="note">Academic/demo use only. These model outputs are not a structural inspection or safety certification.</div></div>
    """,
    unsafe_allow_html=True,
)

if not BACKGROUND_PATH.is_file():
    st.caption("Optional: add a bridge photo named bridge_bg.jpg beside this Python file to customize the background.")

crack_model, crack_error = load_crack_model(CRACK_MODEL_PATH)
status_model, status_error = load_status_model(str(COLLAPSE_MODEL_PATH))
dataset = load_dataset(str(DATASET_PATH))

module1, module2 = st.tabs(["📷 Module 1 — Crack Image Comparison", "🌉 Module 2 — Bridge Status Comparison"])

with module1:
    st.subheader("Compare two images")
    st.write("Upload both images and analyze them separately. Model predictions should not be considered a replacement for visual inspection.")
    if crack_error:
        st.error(crack_error)
        st.info("Check that bridge_crack_final_model.keras is present at the path configured near the top of this file.")
    else:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Test Image A")
            image_a = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"], key="image_a")
            if image_a:
                st.image(image_a, caption="Test Image A", use_container_width=True)
            run_a = st.button("Analyze Image A", key="run_a", use_container_width=True)
            if run_a:
                if not image_a:
                    st.warning("Pehle Image A upload karein.")
                else:
                    try:
                        _, has_crack, score = predict_crack(crack_model, image_a)
                        st.session_state["comparison_a"] = {"has_crack": has_crack, "score": score}
                    except Exception as exc:
                        st.error(f"Image A analysis failed: {exc}")
            result_a = st.session_state.get("comparison_a")
            if result_a:
                if result_a["has_crack"]:
                    st.warning("⚠️ Possible crack pattern detected")
                else:
                    st.success("✓ No significant crack pattern flagged")
                st.metric("Prediction confidence", f"{result_a['score'] * 100:.1f}%")
        with right:
            st.markdown("#### Test Image B")
            image_b = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png"], key="image_b")
            if image_b:
                st.image(image_b, caption="Test Image B", use_container_width=True)
            run_b = st.button("Analyze Image B", key="run_b", use_container_width=True)
            if run_b:
                if not image_b:
                    st.warning("Pehle Image B upload karein.")
                else:
                    try:
                        _, has_crack, score = predict_crack(crack_model, image_b)
                        st.session_state["comparison_b"] = {"has_crack": has_crack, "score": score}
                    except Exception as exc:
                        st.error(f"Image B analysis failed: {exc}")
            result_b = st.session_state.get("comparison_b")
            if result_b:
                if result_b["has_crack"]:
                    st.warning("⚠️ Possible crack pattern detected")
                else:
                    st.success("✓ No significant crack pattern flagged")
                st.metric("Prediction confidence", f"{result_b['score'] * 100:.1f}%")
        if st.session_state.get("comparison_a") and st.session_state.get("comparison_b"):
            a = st.session_state["comparison_a"]
            b = st.session_state["comparison_b"]
            st.markdown("### Side-by-side summary")
            summary = pd.DataFrame([
                {"Test": "Image A", "Model output": "Possible crack" if a["has_crack"] else "No crack flagged", "Prediction confidence (%)": round(a["score"] * 100, 1)},
                {"Test": "Image B", "Model output": "Possible crack" if b["has_crack"] else "No crack flagged", "Prediction confidence (%)": round(b["score"] * 100, 1)},
            ])
            st.dataframe(summary, hide_index=True, use_container_width=True)
            st.caption(f"Decision threshold: {CRACK_THRESHOLD:.3f}. Score is not crack size or structural severity. Results on ordinary wall photos may not generalize to bridge images.")

with module2:
    st.subheader("Compare Standing and Collapsed examples")
    st.write("The app selects example rows from the dataset. Each row is independently evaluated by the model, and the dataset label is displayed separately for comparison.")
    if status_error:
        st.error(status_error)
    elif dataset is None:
        st.error(f"Dataset file not found: {DATASET_PATH}")
        st.info("Place bridge_data.csv.xlsx in the same folder as bridge_app.py.")
    else:
        required = FEATURE_COLUMNS + ["Collapse_Status"]
        missing = [c for c in required if c not in dataset.columns]
        if missing:
            st.error("Dataset mein required columns missing hain: " + ", ".join(missing))
    
        else:
            standing_rows = dataset[
                dataset["Collapse_Status"].astype(str).str.strip().str.lower() == "standing"
            ].copy()

            collapsed_rows = dataset[
                dataset["Collapse_Status"].astype(str).str.strip().str.lower() == "collapsed"
            ].copy()

            if status_model is not None:
                standing_rows = standing_rows[
                    status_model.predict(standing_rows[FEATURE_COLUMNS]) == "Standing"
                ].copy()

                collapsed_rows = collapsed_rows[
                    status_model.predict(collapsed_rows[FEATURE_COLUMNS]) == "Collapsed"
                ].copy()

            c1, c2 = st.columns(2, gap="large")

            with c1:
                st.markdown("#### Test A — Standing dataset example")

                if standing_rows.empty:
                    st.warning("Dataset mein Standing row nahi mili.")
                else:
                    standing_ids = standing_rows.index.tolist()
                    selected_standing = st.selectbox(
                        "Choose Standing example",
                        standing_ids,
                        format_func=lambda idx: f"Dataset row {idx + 2} (Bridge_ID {dataset.loc[idx, 'Bridge_ID']})",
                        key="standing_row"
                    )
                    row_s = dataset.loc[selected_standing]

                    st.dataframe(
                        pd.DataFrame({
                            "Parameter": FEATURE_COLUMNS,
                            "Value": [row_s[c] for c in FEATURE_COLUMNS]
                        }),
                        hide_index=True,
                        use_container_width=True
                    )

                    if st.button("Predict Test A", key="predict_standing", use_container_width=True):
                        try:
                            pred, conf, probabilities = predict_status(status_model, row_s)
                            st.session_state["status_a"] = {
                                "prediction": pred,
                                "confidence": conf,
                                "label": str(row_s["Collapse_Status"])
                            }
                        except Exception as exc:
                            st.error(f"Test A prediction failed: {exc}")

                    if st.session_state.get("status_a"):
                        res = st.session_state["status_a"]
                        show_status_result(
                            res["prediction"],
                            res["confidence"],
                            res["label"]
                        )

            with c2:
                st.markdown("#### Test B — Collapsed dataset example")

                if collapsed_rows.empty:
                    st.warning("Dataset mein Collapsed row nahi mili.")
                else:
                    collapsed_ids = collapsed_rows.index.tolist()
                    selected_collapsed = st.selectbox(
                        "Choose Collapsed example",
                        collapsed_ids,
                        format_func=lambda idx: f"Dataset row {idx + 2} (Bridge_ID {dataset.loc[idx, 'Bridge_ID']})",
                        key="collapsed_row"
                    )
                    row_c = dataset.loc[selected_collapsed]

                    st.dataframe(
                        pd.DataFrame({
                            "Parameter": FEATURE_COLUMNS,
                            "Value": [row_c[c] for c in FEATURE_COLUMNS]
                        }),
                        hide_index=True,
                        use_container_width=True
                    )

                    if st.button("Predict Test B", key="predict_collapsed", use_container_width=True):
                        try:
                            pred, conf, probabilities = predict_status(status_model, row_c)
                            st.session_state["status_b"] = {
                                "prediction": pred,
                                "confidence": conf,
                                "label": str(row_c["Collapse_Status"])
                            }
                        except Exception as exc:
                            st.error(f"Test B prediction failed: {exc}")

                    if st.session_state.get("status_b"):
                        res = st.session_state["status_b"]
                        show_status_result(
                            res["prediction"],
                            res["confidence"],
                            res["label"]
                        )

            if st.session_state.get("status_a") and st.session_state.get("status_b"):
                st.markdown("### Side-by-side summary")
                ra, rb = st.session_state["status_a"], st.session_state["status_b"]
                rows = []

                for label, res in [("Test A", ra), ("Test B", rb)]:
                    rows.append({
                        "Test": label,
                        "Dataset label": res["label"],
                        "Model prediction": res["prediction"],
                        "Confidence (%)": round(res["confidence"] * 100, 1)
                        if res["confidence"] is not None else "N/A"
                    })

                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True,
                    use_container_width=True
                )
                st.caption(
                    "These are examples from your project dataset, not verified real-world bridge cases. "
                    "Model confidence is not a safety probability."
                )

st.markdown("---")
st.warning(
    "Important: This is an academic demonstration. Do not use these predictions "
    "to decide whether a real bridge is safe. Real bridges require qualified engineering inspection."
)
st.markdown(
    '<div class="footer">🌉 Bridge Health Monitoring System | AI-assisted demo</div>',
    unsafe_allow_html=True
)
