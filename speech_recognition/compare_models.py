import speech_recognition as sr
import whisper
import time
import numpy as np

# جمل اختبار عربية طبية
TEST_SENTENCES = [
    "أنا عندي ألم في صدري",
    "محتاج دواء السكر",
    "درجة حرارتي عالية",
    "أشعر بدوار وصداع",
    "ضغطي مش تمام",
]

def test_accuracy(recognizer, whisper_model, audio_files):
    """
    مقارنة دقة النموذجين
    """
    results = []
    
    for audio_file, expected in zip(audio_files, TEST_SENTENCES):
        
        # Google Speech
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        
        start_time = time.time()
        try:
            google_text = recognizer.recognize_google(audio, language="ar-EG")
            google_time = time.time() - start_time
        except:
            google_text = "فشل"
            google_time = 0
        
        # Whisper
        start_time = time.time()
        whisper_result = whisper_model.transcribe(audio_file, language="ar")
        whisper_time = time.time() - start_time
        whisper_text = whisper_result['text']
        
        results.append({
            'expected': expected,
            'google': google_text,
            'google_time': f"{google_time:.2f}s",
            'whisper': whisper_text,
            'whisper_time': f"{whisper_time:.2f}s",
        })
    
    return results

def print_comparison(results):
    """
    طباعة نتائج المقارنة
    """
    print("\n" + "="*60)
    print("📊 نتائج المقارنة: Google vs Whisper")
    print("="*60)
    
    for i, r in enumerate(results, 1):
        print(f"\n🔢 الجملة {i}:")
        print(f"   المتوقع:   {r['expected']}")
        print(f"   Google:    {r['google']} ({r['google_time']})")
        print(f"   Whisper:   {r['whisper']} ({r['whisper_time']})")