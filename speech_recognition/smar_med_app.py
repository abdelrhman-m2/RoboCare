"""
smar_med_app.py (V3.0)
======================
تحديثات:
  - يستخدم handler.transcribe_file() بدل إعادة كتابة منطق Whisper
  - يعرض confidence الحقيقي
  - رسالة واضحة عند رصد الهلوسة
  - استيراد من config.py المركزي
"""

import streamlit as st
import streamlit.components.v1 as components
import sounddevice as sd
import numpy as np
import tempfile
import os
from scipy.io.wavfile import write

from speech_handler import SpeechHandler, SpeechResult
from config import AudioConfig, AppConfig
from arabic_processor import IntentType

# ─────────────────────────────────────────────
# 1. إعداد الصفحة
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=AppConfig.PAGE_TITLE,
    page_icon=AppConfig.PAGE_ICON,
    layout=AppConfig.LAYOUT,
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    direction: rtl;
}
.stApp { background: #070e1a; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 720px; }
h1 { color: #e0f2fe !important; font-size: 1.6rem !important; }
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0369a1) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 12px 32px !important;
    font-weight: 600 !important; width: 100%;
}
.result-card {
    background: linear-gradient(145deg, #0f1e30, #0a1628);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 16px; padding: 24px 28px; margin-top: 20px;
}
.result-row {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); gap: 12px;
}
.result-label { color: #64748b; font-size: 0.82rem; min-width: 120px; }
.result-value { color: #cbd5e1; font-size: 0.9rem; text-align: right; flex: 1; }
.badge { display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-emergency { background:rgba(239,68,68,.15); color:#f87171; border:1px solid rgba(239,68,68,.3); }
.badge-high      { background:rgba(249,115,22,.15); color:#fb923c; border:1px solid rgba(249,115,22,.3); }
.badge-normal    { background:rgba(34,197,94,.15);  color:#4ade80; border:1px solid rgba(34,197,94,.3); }
.symptom-tag {
    display:inline-block; margin:3px; background:rgba(56,189,248,.1); color:#7dd3fc;
    border:1px solid rgba(56,189,248,.2); padding:2px 10px; border-radius:20px; font-size:0.78rem;
}
.confidence-bar-wrap { background: rgba(255,255,255,0.08); border-radius:10px; height:8px; width:100%; margin-top:4px; }
.confidence-bar      { height:8px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. Visualizer الصوتي
# ─────────────────────────────────────────────

def audio_visualizer(state: str = "idle"):
    icons    = {"recording": "🔴", "idle": "🎙️", "processing": "⏳", "done": "✅"}
    colors   = {"recording": "#f87171", "idle": "#7dd3fc", "processing": "#facc15", "done": "#4ade80"}
    labels   = {
        "recording":  "🔴 جارٍ التسجيل...",
        "idle":       "🎙️ جاهز للاستماع",
        "processing": "⏳ جارٍ المعالجة...",
        "done":       "✅ تمت المعالجة",
    }
    pulse = "animation: mic-pulse 1.2s infinite;" if state == "recording" else ""
    bg    = "rgba(239,68,68,0.12)" if state == "recording" else "rgba(56,189,248,0.08)"
    border= "rgba(239,68,68,0.4)"  if state == "recording" else "rgba(56,189,248,0.25)"

    components.html(f"""
    <div style="background:linear-gradient(145deg,#0f1e30,#0a1628);border:1px solid rgba(56,189,248,0.15);
         border-radius:20px;padding:32px 24px;margin:8px 0 20px;display:flex;flex-direction:column;
         align-items:center;gap:20px;font-family:'IBM Plex Sans Arabic',sans-serif;direction:rtl;">
      <div style="width:72px;height:72px;border-radius:50%;background:{bg};border:2px solid {border};
           display:flex;align-items:center;justify-content:center;font-size:2rem;{pulse}">
        {icons.get(state, "🎙️")}
      </div>
      <div style="color:{colors.get(state,'#7dd3fc')};font-size:0.9rem;font-weight:600;">
        {labels.get(state,'...')}
      </div>
    </div>
    <style>
      @keyframes mic-pulse {{0%{{transform:scale(1)}}50%{{transform:scale(1.05)}}100%{{transform:scale(1)}}}}
    </style>
    """, height=180)


# ─────────────────────────────────────────────
# 3. عرض النتائج
# ─────────────────────────────────────────────

def display_result(data: SpeechResult):
    """عرض نتائج SpeechResult في بطاقة موحدة"""
    urgency      = data.urgency_level
    u_class      = "emergency" if "🚨" in urgency else "high" if "⚠️" in urgency else "normal"
    symptoms_html = "".join(f'<span class="symptom-tag">{s}</span>' for s in data.detected_symptoms) \
                    if data.detected_symptoms else '<span style="color:#475569">لا توجد أعراض مكتشفة</span>'

    # شريط confidence
    conf_pct    = int(data.confidence * 100)
    conf_color  = "#4ade80" if conf_pct >= 70 else "#facc15" if conf_pct >= 40 else "#f87171"

    st.markdown(f"""
    <div class="result-card">
      <div style="background:rgba(56,189,248,0.06);border-right:3px solid #0ea5e9;
           border-radius:8px;padding:14px 18px;color:#bae6fd;font-style:italic;margin-bottom:15px;">
        ❝ {data.original_text} ❞
      </div>
      <div class="result-row">
        <span class="result-label">النص الموحد</span>
        <span class="result-value">{data.normalized_text}</span>
      </div>
      <div class="result-row">
        <span class="result-label">النية المكتشفة</span>
        <span class="result-value" style="color:#7dd3fc;">{data.detected_intent.value}</span>
      </div>
      <div class="result-row">
        <span class="result-label">الأعراض</span>
        <span class="result-value">{symptoms_html}</span>
      </div>
      <div class="result-row">
        <span class="result-label">التقييم الطبي</span>
        <span class="result-value"><span class="badge badge-{u_class}">{urgency}</span></span>
      </div>
      <div class="result-row" style="border-bottom:none;">
        <span class="result-label">دقة التعرف</span>
        <span class="result-value">
          <span style="color:{conf_color};font-weight:600;">{conf_pct}%</span>
          <div class="confidence-bar-wrap">
            <div class="confidence-bar" style="width:{conf_pct}%;background:{conf_color};"></div>
          </div>
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    handler.generate_smart_response(data.detected_intent, data.detected_symptoms)
    st.toast("🔊 تم إرسال الرد الصوتي")


# ─────────────────────────────────────────────
# 4. التهيئة
# ─────────────────────────────────────────────

@st.cache_resource
def init_handler():
    return SpeechHandler()

handler = init_handler()


# ─────────────────────────────────────────────
# 5. الواجهة
# ─────────────────────────────────────────────

st.markdown("<h1>🩺 مساعد SMAR-MED الذكي</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎙️ تسجيل مباشر", "📂 رفع ملف"])

# ── تبويب التسجيل المباشر ──────────────────────
with tab1:
    vis_ph = st.empty()
    with vis_ph:
        audio_visualizer("idle")

    if st.button("🔴 ابدأ التسجيل المباشر", key="rec_btn"):
        # تسجيل
        with vis_ph:
            audio_visualizer("recording")

        audio_data = sd.rec(
            int(AudioConfig.RECORDING_DURATION * AudioConfig.SAMPLE_RATE),
            samplerate=AudioConfig.SAMPLE_RATE,
            channels=AudioConfig.CHANNELS,
            dtype=AudioConfig.DTYPE,
        )
        sd.wait()

        # معالجة
        with vis_ph:
            audio_visualizer("processing")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            write(tmp_path, AudioConfig.SAMPLE_RATE,
                  (audio_data * 32767).astype(np.int16))

            data = handler.transcribe_file(tmp_path)

            if data:
                display_result(data)
                with vis_ph:
                    audio_visualizer("done")
            else:
                st.warning("⚠️ لم يتم التعرف على صوت واضح. تأكد من القرب من الميكروفون والتحدث بوضوح.")
                with vis_ph:
                    audio_visualizer("idle")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

# ── تبويب رفع الملف ───────────────────────────
with tab2:
    uploaded = st.file_uploader("اختر ملف صوتي", type=["wav", "mp3", "m4a"])
    if uploaded and st.button("🔍 تحليل الملف المرفوع"):
        with st.spinner("⏳ جاري المعالجة..."):
            ext = os.path.splitext(uploaded.name)[1]
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                data = handler.transcribe_file(tmp_path)

                if data:
                    display_result(data)
                else:
                    st.warning("⚠️ لم يتم التعرف على كلام واضح في الملف.")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)