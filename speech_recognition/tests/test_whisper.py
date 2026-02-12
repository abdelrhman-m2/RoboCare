import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os

# تثبيت sounddevice: pip install sounddevice scipy

def record_audio(duration=5, sample_rate=16000):
    """
    تسجيل صوت من الميكروفون
    """
    print(f"🎙️ اتكلم... ({duration} ثواني)")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # انتظر حتى ينتهي التسجيل
    print("✅ انتهى التسجيل!")
    return audio, sample_rate

def test_whisper():
    """
    اختبار OpenAI Whisper
    """
    # تحميل النموذج (أول مرة هيتحمل تلقائي)
    # الأحجام: tiny, base, small, medium, large
    # للتطوير استخدم: base أو small
    print("📦 تحميل نموذج Whisper (base)...")
    model = whisper.load_model("base")
    print("✅ تم تحميل النموذج!")
    
    # تسجيل الصوت
    audio, sr = record_audio(duration=5)
    
    # حفظ مؤقت
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav.write(tmp.name, sr, (audio * 32767).astype(np.int16))
        tmp_path = tmp.name
    
    # التحليل
    print("🧠 بيحلل الكلام...")
    result = model.transcribe(
        tmp_path,
        language="ar",  # عربي
        task="transcribe"
    )
    
    print(f"📝 النص: {result['text']}")
    print(f"🌍 اللغة المكتشفة: {result['language']}")
    
    # حذف الملف المؤقت
    os.unlink(tmp_path)
    
    return result['text']

if __name__ == "__main__":
    test_whisper()