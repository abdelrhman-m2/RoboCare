"""
config.py - الإعدادات المركزية لنظام SMAR-MED
==============================================
جميع الإعدادات في مكان واحد - لا تكرار في الملفات الأخرى
"""

class WhisperConfig:
    """إعدادات نموذج Whisper للتعرف على الكلام"""
    MODEL_SIZE          = "small"       # خيارات: tiny, base, small, medium, large
    LANGUAGE            = "ar"
    TEMPERATURE         = 0.0          # 0.0 = أقل هلوسة، أكثر دقة
    NO_SPEECH_THRESHOLD = 0.6          # تجاهل النتيجة إذا كان الصمت > 60%
    CONDITION_ON_PREV   = False        # منع الهلوسة التكرارية
    INITIAL_PROMPT      = (
        "المريض يتحدث باللهجة المصرية أو العربية الفصحى عن أعراض طبية. "
        "أمثلة: عندي وجع في صدري، أنا تعبان، محتاج دواء، عندي حرارة."
    )


class AudioConfig:
    """إعدادات التسجيل الصوتي"""
    SAMPLE_RATE         = 16000        # هرتز - مطلوب من Whisper
    RECORDING_DURATION  = 7            # ثواني
    CHANNELS            = 1            # mono
    DTYPE               = 'float32'


class TTSConfig:
    """إعدادات تحويل النص لكلام"""
    LANGUAGE            = "ar"
    SLOW                = False


class LogConfig:
    """إعدادات الـ Logging"""
    FILE_PATH           = "speech_logs.log"
    LEVEL               = "INFO"
    FORMAT              = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class AppConfig:
    """إعدادات تطبيق Streamlit"""
    PAGE_TITLE          = "SMAR-MED Speech"
    PAGE_ICON           = "🩺"
    LAYOUT              = "centered"