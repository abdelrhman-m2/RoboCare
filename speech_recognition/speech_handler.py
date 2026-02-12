"""
speech_handler.py (V3.2 - Mac Fix)
====================================
الإصلاح الجذري لمشكلة الهلوسة على ماك:
  - التسجيل يتم بالـ SR الطبيعي للجهاز (44100Hz على ماك)
  - Resample صح بـ scipy بعد التسجيل مش أثناءه
  - Pre-emphasis لتحسين أصوات العربية (ع، ح، خ، ش)
  - فحص مستوى الصوت قبل Whisper
  - كل حمايات anti-hallucination من V3.0 محتفظ بيها
"""

import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly
from math import gcd
import tempfile
import os
import time
import logging
import threading
from dataclasses import dataclass
from typing import Optional, List
from gtts import gTTS

from config import WhisperConfig, AudioConfig, TTSConfig, LogConfig
from arabic_processor import ArabicMedicalProcessor, IntentType

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    try:
        from playsound import playsound
        PYGAME_AVAILABLE = False
    except ImportError:
        PYGAME_AVAILABLE = False

logging.basicConfig(
    level=getattr(logging, LogConfig.LEVEL),
    format=LogConfig.FORMAT,
    handlers=[
        logging.FileHandler(LogConfig.FILE_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SMAR_MED_VOICE")


# ─────────────────────────────────────────────
# نتيجة المعالجة
# ─────────────────────────────────────────────
@dataclass
class SpeechResult:
    original_text:     str
    normalized_text:   str
    detected_intent:   IntentType
    detected_symptoms: List[str]
    confidence:        float
    urgency_level:     str
    processing_time:   float


# ─────────────────────────────────────────────
# مشغل الصوت
# ─────────────────────────────────────────────
class AudioPlayer:
    def __init__(self):
        if PYGAME_AVAILABLE:
            pygame.mixer.init()

    def play(self, file_path: str):
        try:
            if PYGAME_AVAILABLE:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                pygame.mixer.music.unload()
            else:
                playsound(file_path)
        except Exception as e:
            logger.error(f"خطأ في الصوت: {e}")


# ─────────────────────────────────────────────
# المحرك الرئيسي
# ─────────────────────────────────────────────
class SpeechHandler:

    WHISPER_SR = 16000  # Whisper دايماً محتاج 16kHz

    def __init__(self):
        logger.info("تهيئة SMAR-MED V3.2 (Mac Fix)...")
        self.processor  = ArabicMedicalProcessor()
        self.player     = AudioPlayer()
        self.native_sr  = self._get_native_sr()
        logger.info(f"تحميل Whisper [{WhisperConfig.MODEL_SIZE}]...")
        self.model = whisper.load_model(WhisperConfig.MODEL_SIZE)
        logger.info(f"النظام جاهز | SR الجهاز: {self.native_sr}Hz")

    # ── اكتشاف SR الحقيقي ──────────────────────
    def _get_native_sr(self) -> int:
        """يرجع الـ sample rate الحقيقي للميكروفون - مش نعمل override عليه"""
        try:
            idx  = sd.default.device[0]
            info = sd.query_devices(idx)
            sr   = int(info['default_samplerate'])
            logger.info(f"SR الميكروفون الافتراضي: {sr}Hz")
            return sr
        except Exception:
            logger.warning("مش قادر يكتشف SR الجهاز - هيستخدم 44100Hz")
            return 44100

    # ── Resample صح بعد التسجيل ────────────────
    def _resample_to_whisper(self, audio: np.ndarray, from_sr: int) -> np.ndarray:
        """
        Resample من SR الجهاز (مثلاً 44100) لـ 16000
        بيستخدم resample_poly اللي بيديك نتيجة أنظف من resample العادي
        """
        if from_sr == self.WHISPER_SR:
            return audio

        g   = gcd(from_sr, self.WHISPER_SR)
        up  = self.WHISPER_SR // g
        down = from_sr // g
        resampled = resample_poly(audio, up, down)
        logger.info(f"Resample: {from_sr}Hz → {self.WHISPER_SR}Hz "
                    f"({len(audio)} → {len(resampled)} samples)")
        return resampled.astype(np.float32)

    # ── Pre-emphasis ────────────────────────────
    def _pre_emphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """
        يعزز الترددات العالية - بيحسن أصوات (ع، ح، خ، ش، س)
        معيار صوتيات الكلام العربي
        """
        return np.append(audio[0], audio[1:] - coef * audio[:-1])

    # ── Normalize ───────────────────────────────
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """يمنع الـ clipping ويضمن مستوى صوت ثابت"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return (audio / max_val * 0.95).astype(np.float32)
        return audio

    # ── فحص مستوى الصوت ─────────────────────────
    def _check_level(self, audio: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        logger.info(f"مستوى الصوت RMS: {rms:.5f}")
        if rms < 0.001:
            logger.warning("الميكروفون صامت تقريباً - تأكد من الإعدادات")
            return False
        if rms < 0.005:
            logger.warning("الصوت ضعيف - قرّب من الميكروفون")
        return True

    # ── Pipeline كامل لتجهيز الصوت ──────────────
    def _process_audio(self, raw: np.ndarray) -> np.ndarray:
        """
        Pipeline:
        flatten → resample → pre-emphasis → normalize
        """
        audio = raw.flatten().astype(np.float32)
        audio = self._resample_to_whisper(audio, self.native_sr)
        audio = self._pre_emphasis(audio)
        audio = self._normalize(audio)
        return audio

    # ── Whisper transcription ───────────────────
    def _transcribe_path(self, file_path: str) -> dict:
        return self.model.transcribe(
            file_path,
            language=WhisperConfig.LANGUAGE,
            initial_prompt=WhisperConfig.INITIAL_PROMPT,
            temperature=WhisperConfig.TEMPERATURE,
            no_speech_threshold=WhisperConfig.NO_SPEECH_THRESHOLD,
            condition_on_previous_text=WhisperConfig.CONDITION_ON_PREV,
        )

    def _get_confidence(self, result: dict) -> float:
        segs = result.get("segments", [])
        if not segs:
            return 0.0
        avg = sum(s.get("no_speech_prob", 0.0) for s in segs) / len(segs)
        return round(1.0 - avg, 3)

    def _is_hallucination(self, result: dict) -> bool:
        segs = result.get("segments", [])
        if not segs:
            return True

        avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segs) / len(segs)
        if avg_no_speech > WhisperConfig.NO_SPEECH_THRESHOLD:
            logger.warning(f"Whisper غير واثق: no_speech={avg_no_speech:.2f}")
            return True

        text  = result.get("text", "").strip()
        words = text.split()
        if len(words) > 3 and len(set(words)) / len(words) < 0.4:
            logger.warning(f"هلوسة تكرارية: '{text}'")
            return True

        return False

    # ── TTS ─────────────────────────────────────
    def speak(self, text: str):
        def _thread():
            tmp_path = None
            try:
                tts = gTTS(text=text, lang=TTSConfig.LANGUAGE, slow=TTSConfig.SLOW)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = tmp.name
                tts.save(tmp_path)
                self.player.play(tmp_path)
            except Exception as e:
                logger.error(f"TTS Error: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        threading.Thread(target=_thread, daemon=True).start()

    # ── الدالة الرئيسية ─────────────────────────
    def listen_and_process(self) -> Optional[SpeechResult]:
        start = time.time()

        # 1. سجّل بـ SR الطبيعي للجهاز (مش 16000 مباشرة!)
        logger.info(f"تسجيل {AudioConfig.RECORDING_DURATION}s على {self.native_sr}Hz...")
        raw = sd.rec(
            int(AudioConfig.RECORDING_DURATION * self.native_sr),
            samplerate=self.native_sr,   # ← الفرق الجوهري
            channels=1,
            dtype='float32'
        )
        sd.wait()

        # 2. فحص الصوت
        if not self._check_level(raw):
            return None

        # 3. معالجة (resample + pre-emphasis + normalize)
        audio = self._process_audio(raw)

        # 4. حفظ مؤقت بـ 16kHz وتحويل
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            wav.write(tmp_path, self.WHISPER_SR, (audio * 32767).astype(np.int16))
            result = self._transcribe_path(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 5. فلتر الهلوسة
        if self._is_hallucination(result):
            return None

        original = result['text'].strip()
        if not original:
            return None

        # 6. معالجة ذكية
        norm, intent, symptoms, urgency = self.processor.process(original)
        confidence = self._get_confidence(result)
        logger.info(f"النتيجة: '{original}' | ثقة: {confidence:.0%}")

        self.generate_smart_response(intent, symptoms)

        return SpeechResult(
            original_text=original,
            normalized_text=norm,
            detected_intent=intent,
            detected_symptoms=symptoms,
            confidence=confidence,
            urgency_level=urgency,
            processing_time=round(time.time() - start, 2)
        )

    def transcribe_file(self, file_path: str) -> Optional[SpeechResult]:
        """تحليل ملف مباشرة - للاستخدام في Streamlit"""
        start  = time.time()
        result = self._transcribe_path(file_path)

        if self._is_hallucination(result):
            return None

        original = result['text'].strip()
        if not original:
            return None

        norm, intent, symptoms, urgency = self.processor.process(original)
        confidence = self._get_confidence(result)

        return SpeechResult(
            original_text=original,
            normalized_text=norm,
            detected_intent=intent,
            detected_symptoms=symptoms,
            confidence=confidence,
            urgency_level=urgency,
            processing_time=round(time.time() - start, 2)
        )

    def generate_smart_response(self, intent: IntentType, symptoms: List[str]):
        if intent == IntentType.EMERGENCY:
            r = "لا تقلق، أنا أستدعي الطبيب الآن. حاول التنفس ببطء."
        elif symptoms:
            r = f"سلامتك. سجلت أنك تشعر بـ {' و '.join(symptoms)}."
        elif intent == IntentType.NEED_MEDICATION:
            r = "فهمت. سأخبر الممرضة المسؤولة."
        elif intent == IntentType.MEASURE_VITALS:
            r = "سأقوم بقياس مؤشراتك الحيوية الآن."
        else:
            r = "فهمت ما تقوله. كيف يمكنني مساعدتك؟"
        self.speak(r)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    handler = SpeechHandler()
    print("ابدأ التحدث...")
    data = handler.listen_and_process()
    if data:
        print(f"\nالنص:    {data.original_text}")
        print(f"النية:   {data.detected_intent.value}")
        print(f"الأعراض: {data.detected_symptoms}")
        print(f"الحالة:  {data.urgency_level}")
        print(f"الثقة:   {data.confidence:.0%}")
    else:
        print("لم يتم التعرف على صوت.")