from pydub import AudioSegment

# تحميل الملف
audio = AudioSegment.from_file("audio_samples/New Recording 3.m4a", format="m4a")

# حفظه كـ wav
audio.export("New_Recording_3.wav", format="wav")

print("✅ تم التحويل: New_Recording_3.wav")
