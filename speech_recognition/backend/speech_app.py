import streamlit as st
import tempfile
import sys
import os
import numpy as np
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from speech_handler import SpeechHandler

# ─────────────────────────────────────────────
# 1. إعداد الصفحة والـ CSS
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="SMAR-MED | نظام التعرف على الكلام",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* ── خطوط ── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans Arabic', sans-serif;
        direction: rtl;
    }

    /* ── خلفية الصفحة ── */
    .stApp {
        background: linear-gradient(160deg, #0a1628 0%, #0d2137 50%, #071220 100%);
        min-height: 100vh;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #0e2540 100%);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 20px;
        padding: 32px 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 {
        color: #e0f2fe;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #7dd3fc;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 300;
    }
    .header-badge {
        display: inline-block;
        background: rgba(56,189,248,0.12);
        color: #38bdf8;
        border: 1px solid rgba(56,189,248,0.3);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── بطاقات الأقسام ── */
    .section-card {
        background: linear-gradient(145deg, #132035 0%, #0f1c2e 100%);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        transition: border-color 0.3s;
    }
    .section-card:hover {
        border-color: rgba(56, 189, 248, 0.25);
    }
    .section-title {
        color: #7dd3fc;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(56,189,248,0.1);
    }

    /* ── بطاقات النتائج ── */
    .result-card {
        background: rgba(15, 28, 46, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 20px;
    }
    .result-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        gap: 16px;
    }
    .result-row:last-child { border-bottom: none; }
    .result-label {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 500;
        min-width: 130px;
        padding-top: 2px;
    }
    .result-value {
        color: #e2e8f0;
        font-size: 0.9rem;
        font-weight: 400;
        text-align: right;
        flex: 1;
        line-height: 1.5;
    }

    /* ── شارات الحالة ── */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-emergency { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
    .badge-high      { background: rgba(249,115,22,0.15); color: #fb923c; border: 1px solid rgba(249,115,22,0.3); }
    .badge-medium    { background: rgba(234,179,8,0.15);  color: #facc15; border: 1px solid rgba(234,179,8,0.3); }
    .badge-normal    { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }

    /* ── تنبيه الطوارئ ── */
    .alert-emergency {
        background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(185,28,28,0.08));
        border: 1px solid rgba(239,68,68,0.35);
        border-right: 4px solid #ef4444;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 16px;
        color: #fca5a5;
        font-weight: 500;
        font-size: 0.9rem;
        animation: pulse-border 2s infinite;
    }
    @keyframes pulse-border {
        0%, 100% { border-right-color: #ef4444; }
        50%       { border-right-color: #fca5a5; }
    }

    /* ── نص النتيجة الكبير ── */
    .transcribed-text {
        background: rgba(56,189,248,0.05);
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 10px;
        padding: 16px 20px;
        color: #bae6fd;
        font-size: 1.05rem;
        line-height: 1.8;
        margin-top: 12px;
        font-style: italic;
    }

    /* ── شارات الأعراض ── */
    .symptom-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: flex-end;
    }
    .symptom-tag {
        background: rgba(56,189,248,0.1);
        color: #7dd3fc;
        border: 1px solid rgba(56,189,248,0.2);
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* ── تخصيص عناصر Streamlit ── */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7, #0369a1) !important;
        color: #e0f2fe !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 28px !important;
        font-family: 'IBM Plex Sans Arabic', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0369a1, #075985) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(3,105,161,0.4) !important;
    }
    div[data-testid="stFileUploader"] {
        background: rgba(19,32,53,0.6) !important;
        border: 2px dashed rgba(56,189,248,0.25) !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(56,189,248,0.5) !important;
    }
    .stAudio { border-radius: 10px; overflow: hidden; }

    /* ── إخفاء عناصر Streamlit الافتراضية ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 900px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. تحميل النظام (مرة واحدة فقط)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_speech_handler():
    return SpeechHandler()

with st.spinner("⏳ جارٍ تحميل النماذج..."):
    speech_handler = load_speech_handler()


# ─────────────────────────────────────────────
# 3. دالة التحليل المركزية
# ─────────────────────────────────────────────

def run_analysis(audio_path: str, section_key: str):
    """
    تشغيل التحليل الكامل وعرض النتائج.
    section_key: مفتاح فريد لتمييز كل قسم في session_state.
    """
    cache_key = f"result_{section_key}"

    # تجنب إعادة التحليل عند كل re-run
    if cache_key not in st.session_state:
        with st.spinner("🧠 جارٍ التحليل..."):
            raw = speech_handler.recognizer.model.transcribe(
                audio_path, language="ar"
            )
            text      = raw.get("text", "").strip()
            normalized = speech_handler.processor.normalize(text)
            intent    = speech_handler.processor.detect_intent(normalized)
            symptoms  = speech_handler.processor.extract_symptoms(normalized)
            urgency   = speech_handler.processor.calculate_urgency(intent, symptoms)

        st.session_state[cache_key] = {
            "text": text,
            "normalized": normalized,
            "intent": intent.value,
            "symptoms": symptoms,
            "urgency": urgency,
        }

    r = st.session_state[cache_key]
    _render_results(r)


def _get_urgency_class(urgency: str) -> str:
    if "طارئ" in urgency:  return "emergency"
    if "عالي" in urgency:  return "high"
    if "متوسط" in urgency: return "medium"
    return "normal"


def _render_results(r: dict):
    """عرض نتائج التحليل بشكل موحد."""

    urgency_class = _get_urgency_class(r["urgency"])

    # النص المستخرج
    st.markdown(f'<div class="transcribed-text">❝ {r["text"]} ❞</div>', unsafe_allow_html=True)

    # بطاقة التفاصيل
    symptoms_html = (
        '<div class="symptom-tags">' +
        "".join(f'<span class="symptom-tag">{s}</span>' for s in r["symptoms"]) +
        "</div>"
        if r["symptoms"] else '<span style="color:#64748b;font-size:0.85rem;">لا توجد أعراض واضحة</span>'
    )

    st.markdown(f"""
    <div class="result-card">
        <div class="result-row">
            <span class="result-label">النية المكتشفة</span>
            <span class="result-value" style="color:#7dd3fc; font-weight:600;">{r["intent"]}</span>
        </div>
        <div class="result-row">
            <span class="result-label">النص بعد المعالجة</span>
            <span class="result-value">{r["normalized"]}</span>
        </div>
        <div class="result-row">
            <span class="result-label">الأعراض المكتشفة</span>
            <span class="result-value">{symptoms_html}</span>
        </div>
        <div class="result-row">
            <span class="result-label">درجة الاستعجال</span>
            <span class="result-value">
                <span class="badge badge-{urgency_class}">{r["urgency"]}</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # تنبيه طوارئ
    if urgency_class == "emergency":
        st.markdown("""
        <div class="alert-emergency">
            🚨 &nbsp; <strong>تنبيه طارئ!</strong> — يُنصح باستدعاء الطاقم الطبي فوراً.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 4. واجهة المستخدم
# ─────────────────────────────────────────────

# ── Header ──────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="header-badge">🏥 SMAR-MED</div>
    <h1>🩺 نظام التعرف على الكلام الطبي</h1>
    <p>تحليل آني لكلام المريض — كشف النية، الأعراض، ودرجة الإلحاحية</p>
</div>
""", unsafe_allow_html=True)


# ── القسمان جنباً إلى جنب ──────────────────
col_upload, col_record = st.columns(2, gap="large")


# ════════════════════════════════════════════
# القسم الأول: رفع ملف صوتي
# ════════════════════════════════════════════
with col_upload:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📂 رفع ملف صوتي</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "ارفع ملف صوتي",
        type=["wav", "m4a", "mp3", "ogg"],
        label_visibility="collapsed",
        key="uploader"
    )

    if uploaded_file:
        # إذا تغير الملف، امسح النتيجة القديمة
        if st.session_state.get("last_uploaded") != uploaded_file.name:
            st.session_state.pop("result_upload", None)
            st.session_state["last_uploaded"] = uploaded_file.name

        suffix = os.path.splitext(uploaded_file.name)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

        st.audio(audio_path)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔍 تحليل الملف", key="btn_upload"):
            run_analysis(audio_path, "upload")

        # عرض النتيجة المحفوظة (بعد الضغط)
        elif "result_upload" in st.session_state:
            _render_results(st.session_state["result_upload"])

    else:
        st.markdown("""
        <div style="text-align:center; padding:32px 0; color:#334155;">
            <div style="font-size:2.5rem; margin-bottom:10px;">🎵</div>
            <div style="font-size:0.85rem;">WAV · M4A · MP3 · OGG</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════
# القسم الثاني: تسجيل مباشر
# ════════════════════════════════════════════
with col_record:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎙️ تسجيل صوتي مباشر</div>', unsafe_allow_html=True)

    class AudioRecorder(AudioProcessorBase):
        def __init__(self):
            self.frames = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            pcm = frame.to_ndarray()
            self.frames.append(pcm)
            return frame

    webrtc_ctx = webrtc_streamer(
        key="speech-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioRecorder,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    if webrtc_ctx.audio_processor:
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⏹️ إيقاف وتحليل التسجيل", key="btn_record"):
            recorder: AudioRecorder = webrtc_ctx.audio_processor

            if not recorder.frames:
                st.warning("⚠️ لم يُسجَّل أي صوت بعد.")
            else:
                try:
                    audio_data = np.concatenate(recorder.frames, axis=1)
                    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(tmp_file.name, audio_data.T, 48000)

                    st.audio(tmp_file.name, format="audio/wav")

                    # مسح نتيجة قديمة لكل تسجيل جديد
                    st.session_state.pop("result_record", None)
                    run_analysis(tmp_file.name, "record")

                except Exception as e:
                    st.error(f"❌ خطأ في معالجة الصوت: {e}")

        # عرض النتيجة المحفوظة
        elif "result_record" in st.session_state:
            _render_results(st.session_state["result_record"])

    else:
        st.markdown("""
        <div style="text-align:center; padding:20px 0; color:#334155; font-size:0.85rem;">
            اضغط <strong style="color:#38bdf8;">START</strong> للبدء في التسجيل
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 5. Footer
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    text-align: center;
    padding: 24px 0 8px;
    color: #1e3a5f;
    font-size: 0.78rem;
    border-top: 1px solid rgba(56,189,248,0.08);
    margin-top: 16px;
">
    SMAR-MED Speech Module &nbsp;·&nbsp; Powered by OpenAI Whisper &nbsp;·&nbsp; v1.0
</div>
""", unsafe_allow_html=True)