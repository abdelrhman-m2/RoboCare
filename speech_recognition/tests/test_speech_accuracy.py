import unittest
import sys
import os

# إضافة المسار لضمان الوصول للملفات
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# لاحظ تغيير الاسم هنا لـ ArabicMedicalProcessor
from speech_handler import SpeechHandler, ArabicMedicalProcessor, IntentType

class TestArabicProcessor(unittest.TestCase):
    """
    اختبار معالج النص العربي المطور لـ SMAR-MED
    """
    
    def setUp(self):
        # استخدام الكلاس المطور
        self.processor = ArabicMedicalProcessor()
    
    def test_logic_and_dialect(self):
        """اختبار تحويل العامية والنية في خطوة واحدة"""
        # في الكود الجديد، process ترجع 4 قيم: (النص، النية، الأعراض، الخطورة)
        norm_text, intent, symptoms, urgency = self.processor.process("أنا تعبان")
        
        self.assertIn("مريض", norm_text)
        # كلمة تعبان عادة تصنف كـ FEELING_BAD أو PAIN حسب القاموس
        self.assertEqual(intent, IntentType.PAIN_COMPLAINT) 

    def test_detect_emergency(self):
        """اختبار كشف حالات الطوارئ الحرجة"""
        emergency_texts = [
            "مش قادر أتنفس",
            "إلحقوني بموت",
            "جلطة"
        ]
        for text in emergency_texts:
            _, intent, _, urgency = self.processor.process(text)
            self.assertEqual(intent, IntentType.EMERGENCY, f"فشل في اكتشاف نية الطوارئ لـ: {text}")
            self.assertIn("🚨", urgency, f"فشل في تحديد مستوى الخطورة لـ: {text}")

    def test_extract_symptoms(self):
        """اختبار استخراج الأعراض الطبية"""
        text = "عندي صداع وحمى ودوار"
        _, _, symptoms, _ = self.processor.process(text)
        
        self.assertIn("صداع", symptoms)
        self.assertIn("حمى", symptoms)
        self.assertIn("دوار", symptoms)

if __name__ == "__main__":
    # تشغيل الاختبارات مع إظهار التفاصيل
    unittest.main(verbosity=2)