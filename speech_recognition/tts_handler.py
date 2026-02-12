from gtts import gTTS
from playsound import playsound
import os
import tempfile
import time

class TextToSpeechHandler:
    """
    نظام تحويل النص لكلام
    """
    
    def __init__(self, language="ar", slow=False):
        self.language = language
        self.slow = slow
        self.temp_dir = tempfile.gettempdir()
    
    def speak(self, text: str) -> bool:
        """
        تحويل النص لصوت وتشغيله
        """
        try:
            # إنشاء ملف صوتي مؤقت
            temp_file = os.path.join(
                self.temp_dir, 
                f"speech_{int(time.time())}.mp3"
            )
            
            # تحويل النص لصوت
            tts = gTTS(text=text, lang=self.language, slow=self.slow)
            tts.save(temp_file)
            
            # تشغيل الصوت
            playsound(temp_file)
            
            # حذف الملف المؤقت
            os.remove(temp_file)
            
            return True
            
        except Exception as e:
            print(f"❌ خطأ في TTS: {e}")
            return False
    
    def speak_greeting(self):
        """ترحيب بالمريض"""
        self.speak("مرحباً! أنا روبوت سمار ميد. كيف يمكنني مساعدتك اليوم؟")
    
    def speak_vitals_result(self, heart_rate, temp, spo2):
        """قراءة نتائج المؤشرات"""
        message = f"نبضك {heart_rate} نبضة في الدقيقة، "
        message += f"درجة حرارتك {temp} درجة، "
        message += f"نسبة الأكسجين {spo2} بالمئة."
        self.speak(message)
    
    def speak_alert(self, alert_type):
        """تنبيهات طارئة"""
        alerts = {
            "high_temp": "تحذير! درجة الحرارة مرتفعة. سأخبر الطبيب فوراً.",
            "low_spo2": "تحذير! نسبة الأكسجين منخفضة. هذا مهم جداً.",
            "high_heartrate": "نبضك سريع جداً. سأستدعي الطبيب.",
        }
        self.speak(alerts.get(alert_type, "تنبيه طبي مهم!"))

# اختبار
if __name__ == "__main__":
    tts = TextToSpeechHandler()
    tts.speak_greeting()
    time.sleep(2)
    tts.speak("أنا هنا لمساعدتك")