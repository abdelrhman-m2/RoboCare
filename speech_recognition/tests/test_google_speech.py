import streamlit as st
import speech_recognition as sr

st.title("🎙️ Speech Test")

if st.button("ابدأ التسجيل"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("اتكلم...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
        
    text = recognizer.recognize_google(audio, language="ar-EG")
    st.write("📝 ما قلته:")
    st.write(text)
