import streamlit as st
import tempfile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from speech_handler import SpeechHandler

# تهيئة النظام
st.set_page_config(page_title="SMAR-MED Speech", layout="centered")
st.title("🤖 SMAR-MED Speech Recognition")

speech_handler = SpeechHandler()

# رفع الملف الصوتي
uploaded_file = st.file_uploader("ارفع الملف الصوتي (.wav, .m4a)", type=["wav", "m4a"])

if uploaded_file is not None:
    # حفظ مؤقت
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(audio_path, format='audio/wav')
    
    # تحويل الصوت إلى نص والتحليل
    with st.spinner("جارٍ تحويل الصوت وتحليل النص..."):
        result = speech_handler.recognizer.model.transcribe(audio_path, language="ar")
        text = result['text']
        
        normalized = speech_handler.processor.normalize(text)
        intent = speech_handler.processor.detect_intent(normalized)
        symptoms = speech_handler.processor.extract_symptoms(normalized)
        urgency = speech_handler.processor.calculate_urgency(intent, symptoms)

    st.success("✅ تم التحويل والتحليل!")
    st.write("### النص المستخرج:")
    st.write(text)
    
    st.write("### التحليل الشامل:")
    st.write(f"النص بعد المعالجة: {normalized}")
    st.write(f"النية (Intent): {intent.value}")
    st.write(f"الأعراض (Symptoms): {symptoms}")
    st.write(f"درجة الاستعجال: {urgency}")
