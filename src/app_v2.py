import streamlit as st
import base64, pathlib


# Page configuration
st.set_page_config(
    page_title="African Text · AI Detector",
    page_icon="🌍",
    layout="centered",
)

MODEL_DIR      = "models/afroxlmr_detector"             # the trained AfroXLMR detector
#BASE_TOKENIZER = "Davlan/afro-xlmr-base"      # tokenizer (unchanged by fine-tuning)
# Tokenizer lives inside MODEL_DIR (saved alongside the fine-tuned weights)
#BASELINE_PKL   = "tfidf_lr_baseline.pkl"      # TF-IDF + Logistic Regression baseline
BASELINE_PKL   = "models/tfidf_lr_baseline.pkl"
# Styling

st.markdown("""
<style>
    .block-container { max-width: 760px; padding-top: 2.2rem; }

    .hero {
        text-align: center;
        padding: 1.4rem 1rem 1.6rem;
        background: linear-gradient(135deg, #1d3557 0%, #2a6f97 55%, #43aa8b 100%);
        border-radius: 18px;
        color: #fff;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(29,53,87,0.25);
    }
    .hero h1 { font-size: 1.9rem; margin: 0 0 .35rem; font-weight: 800; }
    .hero p  { font-size: .95rem; margin: 0; opacity: .92; }

    .chips { text-align: center; margin: -0.6rem 0 1.4rem; }
    .chip {
        display: inline-block; padding: 4px 13px; margin: 3px;
        border-radius: 999px; font-size: .8rem; font-weight: 600;
        background: #eef3f8; color: #1d3557; border: 1px solid #d5e2ee;
    }

    .result-card {
        padding: 1.5rem 1.8rem; border-radius: 16px; margin-top: 1.4rem;
        animation: fade .4s ease;
    }
    @keyframes fade { from {opacity:0; transform:translateY(6px);} to {opacity:1; transform:none;} }
    .human-card { background: #e9f7ef; border: 1px solid #b6e2c6; border-left: 7px solid #2e9e5b; }
    .ai-card    { background: #fdecea; border: 1px solid #f5c6c0; border-left: 7px solid #e0473a; }
    .verdict    { font-size: 1.7rem; font-weight: 800; margin: 0 0 .25rem; color: #1a1a1a; }
    .verdict-sub{ font-size: .9rem; color: #555; margin: 0; }

    .stButton > button { border-radius: 10px; font-weight: 700; padding: .55rem 0; }
    [data-testid="stBaseButton-primary"] { background-color: #1d3557 !important; border-color: #1d3557 !important; }
    [data-testid="stBaseButton-primary"]:hover { background-color: #142540 !important; border-color: #142540 !important; }
    .stTextArea textarea { font-size: 1.02rem; border-radius: 12px; }
    div[data-testid="stProgress"] > div { height: 13px; border-radius: 7px; }
    footer, #MainMenu { visibility: hidden; }
    [data-testid="InputInstructions"] { display: none !important; }
    [data-testid="stFileUploaderDropzone"] { display: flex; flex-direction: row; align-items: center; justify-content: space-between; gap: 1rem; padding: 0.35rem 0.75rem; min-height: unset; }
    [data-testid="stFileUploaderDropzoneInstructions"] { flex: 1; }
    [data-testid="stFileUploaderDropzone"] button { margin-left: auto; flex-shrink: 0; }
    [data-testid="stFileUploaderDropzone"] button span { visibility: hidden; position: relative; }
    [data-testid="stFileUploaderDropzone"] button span::after { content: "Upload File"; visibility: visible; position: absolute; left: 50%; transform: translateX(-50%); }
</style>
""", unsafe_allow_html=True)


# Header
#banner = base64.b64encode(pathlib.Path("africa.jpg").read_bytes()).decode()
banner = base64.b64encode(
    (pathlib.Path(__file__).parent / "africa.jpg").read_bytes()
).decode()
st.markdown(f"""
<div style="
    position: relative; border-radius: 18px; overflow: hidden;
    margin-bottom: 1.6rem; box-shadow: 0 8px 24px rgba(0,0,0,0.3);">
    <img src="data:image/jpeg;base64,{banner}"
         style="width:100%; height:190px; object-fit:cover; display:block;">
    <div style="
        position:absolute; inset:0;
        background:linear-gradient(90deg, rgba(10,15,30,0.78) 0%, rgba(10,15,30,0.35) 100%);
        display:flex; flex-direction:column; justify-content:center;
        padding:0 1.8rem; color:#fff;">
        <h1 style="font-size:1.8rem; font-weight:800; margin:0 0 .35rem;">
            African Language · AI Text Detector</h1>
    </div>
</div>
""", unsafe_allow_html=True)



# Model loaders (cached)

@st.cache_resource(show_spinner="Loading AfroXLMR detector… (first run only)")
def load_afroxlmr():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    #tok   = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tok, model

@st.cache_resource(show_spinner="Loading baseline model…")
def load_baseline():
    import joblib
    return joblib.load(BASELINE_PKL)


# Model selector + input

model_choice = st.selectbox(
    "Model",
    ["AfroXLMR (more accurate)", "TF-IDF + Logistic Regression (baseline)"],
)
use_baseline = "Logistic" in model_choice

if "clear_count" not in st.session_state:
    st.session_state.clear_count = 0

text_typed = st.text_area(
    "Enter text to analyse",
    height=180,
    placeholder="Type or paste your text here…",
    label_visibility="collapsed",
    key=f"text_input_{st.session_state.clear_count}",
)
st.markdown(
    f'<div style="position:relative;margin-top:-1.2rem;padding-right:.6rem;'
    f'text-align:right;font-size:.75rem;color:#999;pointer-events:none;z-index:10;">'
    f'{len(text_typed):,} / 15,000 Characters</div>',
    unsafe_allow_html=True,
)

text_file = ""
uploaded = st.file_uploader(
    "Upload File",
    type=["txt"],
    label_visibility="collapsed",
)
if uploaded is not None:
    text_file = uploaded.read().decode("utf-8", errors="ignore")[:15000]

col_run, col_clear = st.columns([3, 1])
with col_run:
    run = st.button("Detect Text", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.clear_count += 1
        st.rerun()

text = text_typed if text_typed.strip() else text_file

st.caption("Best results with isiXhosa, isiZulu, Sepedi, or Kiswahili. Other languages may be unreliable.")

# Prediction

if run:
    word_count = len(text.split())
    if word_count < 40:
        needed = 40 - word_count
        st.warning(f"⚠️ Not enough text to analyse. Please add {needed} more word{'s' if needed != 1 else ''} (minimum 40).")
        st.stop()

    import re as _re
    alpha_ratio = len(_re.findall(r"[a-zA-Z-￿]", text)) / max(len(text), 1)
    if alpha_ratio < 0.4:
        st.warning("⚠️ The text appears to contain mostly numbers or symbols. Please enter readable text.")
        st.stop()

    with st.spinner("Analysing…"):
        if use_baseline:
            pipe  = load_baseline()
            proba = pipe.predict_proba([text])[0]      # [P(human), P(AI)]
            label = int(pipe.predict([text])[0])
            human, ai = float(proba[0]), float(proba[1])
        else:
            import torch
            tok, model = load_afroxlmr()
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            label = int(torch.argmax(logits).item())
            human, ai = probs[0], probs[1]

    is_ai = label == 1
    conf  = max(ai, human) * 100

    card  = "ai-card" if is_ai else "human-card"
    icon  = "🤖" if is_ai else "👤"
    title = "AI-generated" if is_ai else "Human-written"

    if conf >= 85:    strength = "High confidence"
    elif conf >= 65:  strength = "Moderate confidence"
    else:             strength = "Low confidence - result is uncertain"

    st.markdown(f"""
    <div class="result-card {card}">
        <p class="verdict">{icon} {title}</p>
        <p class="verdict-sub">{strength} · {conf:.1f}% · {model_choice.split('(')[0].strip()}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("👤 Human", f"{human*100:.1f}%")
        st.progress(human)
    with c2:
        st.metric("🤖 AI", f"{ai*100:.1f}%")
        st.progress(ai)


    import re, numpy as np

    sent_pat = r"(?<=[.!?])\s+"
    sentences = [s.strip() for s in re.split(sent_pat, text.strip()) if s.strip()]
    pred_conf = conf / 100

    with st.spinner("Analyzing ..."):
        import shap
        sv = None
        if len(sentences) > 1:
            masker = shap.maskers.Text(tokenizer=sent_pat)
            if use_baseline:
                _pipe = load_baseline()
                def _baseline_pred(texts):
                    return _pipe.predict_proba(list(texts))
                sv = shap.Explainer(_baseline_pred, masker)([text])
            else:
                import torch
                _tok, _mdl = load_afroxlmr()
                def _xlmr_pred(texts):
                    out = []
                    for t in texts:
                        t = t.strip() or " "
                        enc = _tok(t, return_tensors="pt", truncation=True, max_length=512)
                        with torch.no_grad():
                            lg = _mdl(**enc).logits
                        out.append(torch.softmax(lg, dim=-1)[0].tolist())
                    return np.array(out)
                explainer = shap.Explainer(_xlmr_pred, masker)
                sv = explainer([text], max_evals=min(2 * len(sentences) + 2, 30))

    if sv is not None:
        shap_raw = sv.values[0]
        ai_vals = shap_raw[:, 1] if shap_raw.ndim == 2 else shap_raw
    else:
        ai_vals = [ai - human] * len(sentences)

    n = min(len(sentences), len(ai_vals))
    sentences, ai_vals = sentences[:n], list(ai_vals)[:n]

    mean_abs = float(np.mean(np.abs(ai_vals))) if n > 0 else 1.0
    ref = mean_abs * 1.5 if mean_abs > 0 else 1.0

    # Highlight exactly round(ai × n) sentences — top-ranked by SHAP value
    ranked = sorted(range(n), key=lambda i: ai_vals[i], reverse=True)
    n_highlight = round(ai * n)
    highlight_set = set(ranked[:n_highlight])

    spans = []
    for idx, (sent, val) in enumerate(zip(sentences, ai_vals)):
        val = float(val)
        shap_intensity = min(1.0, abs(val) / ref)
        if idx in highlight_set:
            alpha = min(0.92, pred_conf * 0.9 + 0.08 * shap_intensity)
            bg = f"rgba(220,53,69,{alpha:.2f})"
            fg = "#fff" if alpha > 0.4 else "#222"
        else:
            bg, fg = "transparent", "#1a1a1a"
        tip = f"SHAP {val:+.4f} → {'AI' if val > 0 else 'Human'}"
        spans.append(
            f'<span style="background:{bg};color:{fg};padding:2px 5px;'
            f'border-radius:4px;cursor:help;" title="{tip}">{sent}</span>'
        )

    st.markdown(
        '<div style="line-height:2.4;font-size:1.05rem;padding:1.2rem 1.5rem;'
        'background:#f8f9fa;border-radius:14px;border:1px solid #dee2e6;">'
        + " ".join(spans)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    with st.expander("📊  Text statistics"):
        words = len(text.split())
        chars = len(text)
        sents = max(1, text.count(".") + text.count("!") + text.count("?"))
        s1, s2, s3 = st.columns(3)
        s1.metric("Characters", chars)
        s2.metric("Words", words)
        s3.metric("Sentences", sents)
