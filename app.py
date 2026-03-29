import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import time
from ultralytics import YOLO

# ─────────────────────────────────────────
#  Page config  
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DentalScan AI · YOLOv10",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
#  Classes  (from data.yaml)
# ─────────────────────────────────────────
CLASS_NAMES = {
    0: "Caries",
    1: "Infection",
    2: "Impacted",
    3: "BDC/BDR",
    4: "Fractured",
    5: "Healthy",
}

CLASS_ICONS = {
    0: "🦠", 1: "🔴", 2: "📌", 3: "📐", 4: "💔", 5: "✅"
}

CLASS_COLORS_BGR = {
    0: (0,   200, 255),   # Caries     – amber
    1: (0,    60, 255),   # Infection  – red
    2: (255, 160,   0),   # Impacted   – blue
    3: (50,  255, 100),   # BDC/BDR    – green
    4: (180,   0, 255),   # Fractured  – purple
    5: (0,   220, 110),   # Healthy    – teal
}

def bgr_to_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"

CLASS_COLORS_HEX = {k: bgr_to_hex(v) for k, v in CLASS_COLORS_BGR.items()}

# ─────────────────────────────────────────
#  CSS — premium dark theme, full visibility
# ─────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #060b18; color: #e2e8f0; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #2d4a6e; border-radius: 3px; }

/* ── Hero ── */
.hero {
    position: relative;
    padding: 44px 52px 40px;
    border-radius: 24px;
    overflow: hidden;
    margin-bottom: 28px;
    background: linear-gradient(135deg, #0d1b33 0%, #112244 40%, #0a1628 100%);
    border: 1px solid rgba(96,165,250,0.18);
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(99,179,237,0.14) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 10%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(139,92,246,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.28);
    color: #bfdbfe;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1.1;
    margin: 0 0 14px;
    background: linear-gradient(135deg, #bfdbfe 0%, #60a5fa 35%, #818cf8 65%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
/* FIXED: was #64748b — now #94a3b8 for clear readability */
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.7;
    max-width: 580px;
    margin: 0;
}
.hero-stats { display: flex; gap: 28px; margin-top: 28px; }
/* FIXED: was #475569 — now #8bafc8 */
.h-stat { color: #8bafc8; font-size: 0.82rem; font-weight: 500; }
.h-stat span {
    color: #93c5fd;
    font-weight: 700;
    font-size: 1.15rem;
    font-family: 'JetBrains Mono', monospace;
    display: block;
}

/* ── Panel cards ── */
.panel {
    background: rgba(10,18,35,0.9);
    border: 1px solid rgba(96,165,250,0.12);
    border-radius: 18px;
    padding: 22px 20px;
    margin-bottom: 14px;
    backdrop-filter: blur(10px);
}
/* FIXED: was #334155 — now #7da8cc */
.panel-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #7da8cc;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 7px;
}
.panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(96,165,250,0.12);
}

/* ── File upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(10,18,35,0.8) !important;
    border: 2px dashed rgba(96,165,250,0.25) !important;
    border-radius: 14px !important;
    transition: all 0.25s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(96,165,250,0.55) !important;
    background: rgba(15,30,60,0.8) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 1.8rem !important;
    width: 100% !important;
    letter-spacing: 0.4px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(29,78,216,0.35) !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(29,78,216,0.45) !important;
}

/* ── Sliders ── */
/* FIXED: label was #64748b — now #9db8d0 */
.stSlider label { color: #9db8d0 !important; font-size: 0.84rem !important; font-weight: 500 !important; }
[data-testid="stSlider"] p { color: #bfdbfe !important; font-weight: 600 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10,18,35,0.9) !important;
    border-radius: 12px 12px 0 0 !important;
    border: 1px solid rgba(96,165,250,0.12) !important;
    border-bottom: none !important;
    padding: 4px !important;
    gap: 2px !important;
}
/* FIXED: inactive tab was #475569 — now #8bafc8 */
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 9px !important;
    color: #8bafc8 !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(37,99,235,0.22) !important;
    color: #bfdbfe !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: rgba(10,18,35,0.9) !important;
    border: 1px solid rgba(96,165,250,0.12) !important;
    border-radius: 0 12px 12px 12px !important;
    padding: 16px !important;
}

/* ── Legend ── */
.legend-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
/* FIXED: was #64748b — now #b0c8e0 */
.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    color: #b0c8e0;
    font-weight: 500;
    padding: 7px 9px;
    border-radius: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
}
.legend-swatch { width: 28px; height: 10px; border-radius: 3px; flex-shrink: 0; }

/* ── Metrics ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 16px 0 8px;
}
.metric-card {
    background: rgba(15,25,50,0.95);
    border: 1px solid rgba(96,165,250,0.12);
    border-radius: 12px;
    padding: 14px 12px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
}
.metric-num {
    font-size: 1.75rem;
    font-weight: 800;
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
/* FIXED: was #334155 — now #8bafc8 */
.metric-lbl {
    font-size: 0.68rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #8bafc8;
    margin-top: 6px;
    font-weight: 600;
}

/* ── Detection rows ── */
.det-row {
    display: flex;
    align-items: center;
    padding: 11px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid;
    gap: 10px;
}
.det-icon { font-size: 1.1rem; }
.det-name { font-weight: 600; font-size: 0.9rem; color: #e2e8f0; flex: 1; }
/* FIXED: bbox was #334155 — now #7da8cc */
.det-bbox { font-size: 0.70rem; color: #7da8cc; font-family: 'JetBrains Mono', monospace; }
/* FIXED: confidence pill was #94a3b8 — now #bfdbfe, bolder */
.det-conf-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    background: rgba(255,255,255,0.08);
    color: #bfdbfe;
}
.conf-track {
    height: 3px;
    border-radius: 3px;
    background: rgba(255,255,255,0.07);
    margin-top: 2px;
    overflow: hidden;
}
.conf-fill { height: 100%; border-radius: 3px; opacity: 0.75; }

/* ── Empty states ── */
.empty-state { text-align: center; padding: 48px 20px; }
.empty-icon { font-size: 3.5rem; margin-bottom: 14px; }
/* FIXED: title was #2563eb (dark blue — low contrast) — now #60a5fa */
.empty-title { font-size: 1rem; font-weight: 700; color: #60a5fa; margin-bottom: 8px; }
/* FIXED: sub-text was #1e3a5f (nearly invisible) — now #8bafc8 */
.empty-sub { font-size: 0.85rem; color: #8bafc8; line-height: 1.7; }
.empty-sub strong { color: #bfdbfe; }

/* ── Model status bar ── */
/* FIXED: text was #334155 — now #8bafc8 */
.model-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 16px;
    background: rgba(10,18,35,0.85);
    border: 1px solid rgba(96,165,250,0.10);
    border-radius: 10px;
    margin-bottom: 20px;
    font-size: 0.78rem;
    color: #8bafc8;
    font-weight: 500;
}
.model-bar code {
    color: #7da8cc !important;
    background: rgba(255,255,255,0.06);
    padding: 1px 6px;
    border-radius: 4px;
}
.model-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 7px #22c55e;
    flex-shrink: 0;
}
.model-dot.warn { background: #f59e0b; box-shadow: 0 0 7px #f59e0b; }

/* ── Inference time ── */
/* FIXED: was #1e3a5f — now #7da8cc */
.infer-time {
    text-align: right;
    font-size: 0.72rem;
    color: #7da8cc;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 6px;
    font-weight: 500;
}

/* ── st.caption ── */
[data-testid="stCaptionContainer"] p { color: #7da8cc !important; font-size: 0.78rem !important; }

/* ── Misc ── */
hr { border-color: rgba(96,165,250,0.09) !important; margin: 18px 0 !important; }
.stSpinner > div { border-top-color: #60a5fa !important; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Load fine-tuned model
# ─────────────────────────────────────────
FINETUNED_PATH = "./runs/detect/Yolo_10s_train/weights/best.pt"

@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(FINETUNED_PATH):
        return YOLO(FINETUNED_PATH), FINETUNED_PATH, True
    return YOLO("yolov10s.pt"), "yolov10s.pt", False

model, model_path_used, is_finetuned = load_model()

# ─────────────────────────────────────────
#  Hero
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-chip">🦷 &nbsp;YOLOv10s · Dental AI · v2.0</div>
  <h1 class="hero-title">DentalScan AI</h1>
  <p class="hero-sub">
    Upload a dental X-ray image and our fine-tuned YOLOv10s model instantly detects
    and classifies dental conditions with bounding box precision.
  </p>
  <div class="hero-stats">
    <div class="h-stat"><span>6</span>Classes</div>
    <div class="h-stat"><span>250</span>Epochs</div>
    <div class="h-stat"><span>640</span>Image size</div>
    <div class="h-stat"><span>10s</span>Architecture</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Model status bar
dot_cls  = "model-dot" if is_finetuned else "model-dot warn"
dot_info = (f"Fine-tuned &nbsp;·&nbsp; <code>{FINETUNED_PATH}</code>"
            if is_finetuned
            else "Fine-tuned weights not found — using base yolov10s.pt (class names will be wrong)")
st.markdown(f"""
<div class="model-bar">
  <div class="{dot_cls}"></div>
  <span>Model loaded &nbsp;·&nbsp; {dot_info}</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────
left_col, right_col = st.columns([1, 2.2], gap="large")

# ══════════ LEFT PANEL ══════════
with left_col:

    # Upload
    st.markdown('<div class="panel"><div class="panel-title">📂 Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Settings
    st.markdown('<div class="panel"><div class="panel-title">⚙️ Detection Settings</div>', unsafe_allow_html=True)
    conf_thresh = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.25, 0.05,
        help="Minimum score to display a detection.",
    )
    iou_thresh = st.slider(
        "IoU / NMS threshold", 0.10, 0.90, 0.45, 0.05,
        help="Controls overlapping-box suppression.",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Legend
    st.markdown('<div class="panel"><div class="panel-title">🎨 Class Legend</div>', unsafe_allow_html=True)
    legend_html = '<div class="legend-grid">'
    for cid, cname in CLASS_NAMES.items():
        col_hex = CLASS_COLORS_HEX[cid]
        icon    = CLASS_ICONS[cid]
        legend_html += (
            f'<div class="legend-item">'
            f'<div class="legend-swatch" style="background:{col_hex};"></div>'
            f'<span>{icon} {cname}</span>'
            f'</div>'
        )
    legend_html += '</div>'
    st.markdown(legend_html + '</div>', unsafe_allow_html=True)

# ══════════ RIGHT PANEL ══════════
with right_col:
    if uploaded_file is None:
        st.markdown("""
        <div class="panel" style="padding:0;">
          <div class="empty-state">
            <div class="empty-icon">🦷</div>
            <div class="empty-title">Ready to Analyse</div>
            <div class="empty-sub">
              Upload a dental X-ray on the left panel,<br>
              then click <strong>Run Detection</strong> to see results.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        image    = Image.open(uploaded_file).convert("RGB")
        w, h     = image.size

        tab_orig, tab_result = st.tabs(["📷  Original Image", "🎯  Detection Result"])

        with tab_orig:
            st.image(image, use_container_width=True)
            st.caption(f"Image size: {w} × {h} px")

        with tab_result:
            run_col, _ = st.columns([1, 2])
            with run_col:
                run_btn = st.button("🔍  Run Detection", key="run_detect")

            if run_btn:
                start = time.time()
                with st.spinner("Running YOLOv10s inference…"):
                    results = model(
                        image,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        verbose=False,
                    )
                elapsed_ms = (time.time() - start) * 1000

                boxes = results[0].boxes

                # ── Draw bounding boxes (per-class colours, only our 6 classes) ──
                img_cv     = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                detections = []

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        if cls_id not in CLASS_NAMES:
                            continue
                        conf  = float(box.conf[0].item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        color = CLASS_COLORS_BGR[cls_id]
                        label = f"{CLASS_NAMES[cls_id]}  {conf*100:.0f}%"

                        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                        pad = 4
                        cv2.rectangle(img_cv,
                                      (x1, y1 - th - pad * 2),
                                      (x1 + tw + pad * 2, y1),
                                      color, -1)
                        cv2.putText(img_cv, label,
                                    (x1 + pad, y1 - pad),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                    (0, 0, 0), 1, cv2.LINE_AA)

                        detections.append({
                            "cls_id": cls_id,
                            "conf":   conf,
                            "bbox":   (x1, y1, x2, y2),
                        })

                res_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                st.image(res_img, use_container_width=True)
                st.markdown(
                    f'<div class="infer-time">⚡ {elapsed_ms:.0f} ms</div>',
                    unsafe_allow_html=True,
                )

                # ── Summary metrics ──
                n_dets = len(detections)
                n_cls  = len(set(d["cls_id"] for d in detections))
                avg_c  = (sum(d["conf"] for d in detections) / n_dets * 100) if n_dets else 0

                st.markdown(f"""
                <div class="metric-grid">
                  <div class="metric-card">
                    <div class="metric-num">{n_dets}</div>
                    <div class="metric-lbl">Detections</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-num">{n_cls}</div>
                    <div class="metric-lbl">Classes Found</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-num">{avg_c:.0f}%</div>
                    <div class="metric-lbl">Avg Confidence</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)

                # ── Detection list ──
                if detections:
                    st.markdown(
                        '<div class="panel-title" style="margin-bottom:12px;">🔬 Findings</div>',
                        unsafe_allow_html=True,
                    )
                    detections.sort(key=lambda d: d["conf"], reverse=True)
                    for d in detections:
                        cid   = d["cls_id"]
                        cname = CLASS_NAMES[cid]
                        icon  = CLASS_ICONS[cid]
                        conf  = d["conf"] * 100
                        hexc  = CLASS_COLORS_HEX[cid]
                        x1, y1, x2, y2 = d["bbox"]
                        bar_w = int(conf)

                        st.markdown(f"""
                        <div class="det-row" style="border-left-color:{hexc};">
                          <span class="det-icon">{icon}</span>
                          <span class="det-name">{cname}</span>
                          <span class="det-bbox">[{x1},{y1} → {x2},{y2}]</span>
                          <span class="det-conf-pill">{conf:.1f}%</span>
                        </div>
                        <div class="conf-track" style="margin-top:-6px;margin-bottom:8px;">
                          <div class="conf-fill" style="width:{bar_w}%;background:{hexc};"></div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="empty-state" style="padding:28px 16px;">
                      <div class="empty-icon">🔎</div>
                      <div class="empty-title">No Findings Detected</div>
                      <div class="empty-sub">
                        Nothing above <strong>{conf_thresh*100:.0f}%</strong> confidence.<br>
                        Try <strong>lowering the Confidence Threshold</strong> in the left panel.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
