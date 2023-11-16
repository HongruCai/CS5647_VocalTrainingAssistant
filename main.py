import speech_recognition as sr
import librosa
import numpy as np


# 提取歌词
def extract_lyrics(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='zh-CN')
    return text


# 提取音高和持续时间
def extract_pitch_and_duration(audio_file):
    y, sr = librosa.load(audio_file)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # 简化: 只考虑最大振幅的音高
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch.append(pitches[index, t])

    pitch = np.array(pitch)

    # 计算音符开始的位置
    onsets = librosa.onset.onset_detect(y=y, sr=sr)

    # 使用midi表示音高
    midi_notes = librosa.hz_to_midi(pitch[pitch > 0])

    # 根据onsets计算每个音符的持续时间
    durations = np.diff(onsets, append=len(y) / sr * librosa.time_to_frames(1, sr=sr))
    durations = durations / sr  # convert frames to seconds

    return midi_notes, durations


audio_file = 'data/test/tmpgvocwpiq.wav'

lyrics = extract_lyrics(audio_file)
midi_notes, duration_per_note = extract_pitch_and_duration(audio_file)

print("Lyrics:", lyrics)
print("MIDI Notes:", midi_notes)
print("Duration:", duration_per_note)
