import streamlit as st
import base64, pathlib


# Page configuration
st.set_page_config(
    page_title="African Text · AI Detector",
    page_icon="🌍",
    layout="centered",
)

MODEL_DIR      = "checkpoint-319"            # the trained AfroXLMR detector
BASE_TOKENIZER = "Davlan/afro-xlmr-base"     # tokenizer (unchanged by fine-tuning)

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
    .verdict    { font-size: 1.7rem; font-weight: 800; margin: 0 0 .25rem; }
    .verdict-sub{ font-size: .9rem; color: #555; margin: 0; }

    .stButton > button { border-radius: 10px; font-weight: 700; padding: .55rem 0; }
    .stTextArea textarea { font-size: 1.02rem; border-radius: 12px; }
    div[data-testid="stProgress"] > div { height: 13px; border-radius: 7px; }
    footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# Header
banner = base64.b64encode(pathlib.Path("Africa.jpg").read_bytes()).decode()
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

st.markdown("""
<div class="chips">
    <span class="chip">isiXhosa</span>
    <span class="chip">isiZulu</span>
    <span class="chip">Sepedi</span>
    <span class="chip">Kiswahili</span>
</div>
""", unsafe_allow_html=True)


# Model loader (cached)

@st.cache_resource(show_spinner="Loading AfroXLMR detector… (first run only)")
def load_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok   = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tok, model

# Input

text = st.text_area(
    "Enter text to analyse",
    height=180,
    placeholder="Ngena isicatshulwa sakho lapha…\nAndika maandishi yako hapa…\nFaka mokwalo wa gago fa…",
)

col_run, col_clear = st.columns([3, 1])
with col_run:
    run = st.button("🔍  Analyse text", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.rerun()

st.caption("Best results with isiXhosa, isiZulu, Sepedi, or Kiswahili. Other languages may be unreliable.")

# Prediction

if run:
    if len(text.strip()) < 10:
        st.warning("⚠️  Please enter at least 10 characters for a reliable result.")
        st.stop()

    import torch
    with st.spinner("Analysing…"):
        tok, model = load_model()
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1)[0].tolist()
        label  = int(torch.argmax(logits).item())

    ai, human = probs[1], probs[0]
    is_ai     = label == 1
    conf      = max(ai, human) * 100

    card  = "ai-card" if is_ai else "human-card"
    icon  = "🤖" if is_ai else "👤"
    title = "AI-generated" if is_ai else "Human-written"

    if conf >= 85:    strength = "High confidence"
    elif conf >= 65:  strength = "Moderate confidence"
    else:             strength = "Low confidence — result is uncertain"

    st.markdown(f"""
    <div class="result-card {card}">
        <p class="verdict">{icon} {title}</p>
        <p class="verdict-sub">{strength} · {conf:.1f}%</p>
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

    with st.expander("📊  Text statistics"):
        words = len(text.split())
        chars = len(text)
        sents = max(1, text.count(".") + text.count("!") + text.count("?"))
        s1, s2, s3 = st.columns(3)
        s1.metric("Characters", chars)
        s2.metric("Words", words)
        s3.metric("Sentences", sents)