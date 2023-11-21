import whisper
from basic_pitch.inference import predict_and_save
import mido
import pandas as pd
import glob
import os
import csv
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import librosa

import tensorflow as tf
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

matplotlib.rcParams['font.family'] = 'SimHei'
basic_pitch_model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))


def melody_analysis_and_save(file_list, output_dir):
    predict_and_save(
        file_list,
        output_dir,
        save_midi=False,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=True,
        model_path=ICASSP_2022_MODEL_PATH,
    )


def get_lyrics(audio_file):
    model = whisper.load_model("base")
    audio_path = audio_file
    result = model.transcribe(audio_path)
    return result["text"]


def note_to_midi(note):
    rec = []
    for item in note:
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = int(item[-1])
        note_name = item[:-1]
        note_index = notes.index(note_name)
        midi_number = note_index + (octave + 1) * 12
        rec.append(midi_number)
    return rec


def calculate_score(standard, user, threshold=3):
    i = 0
    score = 0
    while i < standard[-1][-1]:
        st = 0
        for note1 in standard:
            if note1[1] <= i <= note1[2]:
                st = note1[0]
                break
        if st == 0:
            score += 1
            i += 0.00001
            continue
        us = 0
        for note2 in user:
            if note2[1] <= i <= note2[2]:
                us = note2[0]
                break
        if st + threshold >= us >= st - threshold:
            score += 1
        i += 0.00001
    return score / (standard[-1][-1] / 0.00001)


def convert_data(data):
    result = []
    current_time = 0.0
    res2 = []
    for key, (lyric, pitches, durations) in data.items():
        rec1 = 0
        count =0
        for pitch, duration in zip(pitches, durations):
            start_time = current_time
            if count == 0:
                rec1 = start_time
            end_time = start_time + float(duration)
            result.append((pitch, start_time, end_time))

            current_time += float(duration)
        res2.append((lyric, rec1, current_time))
    return result, res2


def calculate_accuracy(sheet_music, user_singing, time_tolerance1, time_tolerance2, threshold):
    pitch_matches = 0
    rhythm_matches = 0
    duration_matches = 0

    for sheet_pitch, sheet_start, sheet_end in sheet_music:
        for user_pitch, user_start, user_end in user_singing:
            if abs(sheet_start - user_start) <= time_tolerance1:
                rhythm_matches += 1
                if sheet_pitch + threshold >= user_pitch >= sheet_pitch - threshold:
                    pitch_matches += 1

                sheet_duration = sheet_end - sheet_start
                user_duration = user_end - user_start
                if abs(sheet_duration - user_duration) <= time_tolerance2:
                    duration_matches += 1
                break

    total_notes = len(sheet_music)
    pitch_accuracy = (pitch_matches / total_notes) * 100 if total_notes else 0
    rhythm_accuracy = (rhythm_matches / total_notes) * 100 if total_notes else 0
    duration_accuracy = (duration_matches / total_notes) * 100 if total_notes else 0

    return round(pitch_accuracy, 2), round(rhythm_accuracy, 2), round(duration_accuracy, 2)


def visualize(stan,standard, user):
    times = []
    pitches = []
    for pitch, start, end in user:
        times.extend([start, end])
        pitches.extend([pitch, pitch])
    times2 = []
    pitches2 = []
    for pitch, start, end in standard:
        if pitch == 0:
            continue
        times2.extend([start, end])
        pitches2.extend([pitch, pitch])
    plt.figure(figsize=(8, 4))
    for lyric, start, end in stan:
        for pitch1, start1, end1 in standard:
            if start == start1:
                if lyric != "AP" and lyric != "SP":
                    plt.text(start, pitch1, lyric, fontsize=15)
                    break
                    # print(lyric, start, pitch1)

    plt.plot(times, pitches, marker='o', color='red')
    plt.plot(times2, pitches2, marker='o', color='blue')
    plt.legend(['User Singing', 'Reference'])
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (MIDI)')
    plt.grid(True)

    return plt


def analyze(audio_file, sheet_text, sheet_note, sheet_duration, threshold1, threshold2, tolerance1, tolerance2):
    output_dir = 'resources/intermediate_files'
    melody_analysis_and_save([audio_file], output_dir)

    notes_file = glob.glob(output_dir + "/*pitch.csv")
    text = sheet_text
    pro_text = []
    for i in range(len(text)):
        if text[i] == "A":
            pro_text.append(text[i] + text[i + 1])
        elif text[i] == "S":
            pro_text.append(text[i] + text[i + 1])
        elif text[i] == "P":
            continue
        else:
            pro_text.append(text[i])
    # print(pro_text)
    note = sheet_note.split(" | ")
    duration = sheet_duration.split(" | ")
    # print(len(pro_text), len(note), len(duration))
    standard = {}
    for i in range(len(pro_text)):
        char = pro_text[i]
        standard[i] = (char, note_to_midi(note[i].split(' ')) if note[i] != 'rest' else [0], duration[i].split(" "))
    # print(standard)

    user_notes = []
    with open(notes_file[0], newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            if row[0] == 'start_time_s':
                continue
            start_time_s = float(row[0])
            end_time_s = float(row[1])
            pitch_midi = int(row[2])
            user_notes.append((pitch_midi, start_time_s, end_time_s))
    user_notes = sorted(user_notes, key=lambda x: x[1])

    standard_notes,standard_lyrics = convert_data(standard)
    user_start_time = user_notes[0][1]
    standard_start_time = standard_notes[0][2]
    for i in range(len(user_notes)):
        user_notes[i] = (user_notes[i][0], user_notes[i][1] - user_start_time + standard_start_time,
                         user_notes[i][2] - user_start_time + standard_start_time)
    score = calculate_score(standard_notes, user_notes, threshold=threshold1)
    score = round(score * 100, 2)
    visualizations = visualize(standard_lyrics,standard_notes, user_notes)

    pitch_accuracy, rhythm_accuracy, duration_accuracy = calculate_accuracy(standard_notes, user_notes, time_tolerance1=tolerance1,
                                                                            time_tolerance2=tolerance2,
                                                                            threshold=threshold2)
    os.remove(notes_file[0])
    return score, visualizations, pitch_accuracy, rhythm_accuracy, duration_accuracy, len(user_notes), round(user_notes[-1][2],2), len(
        standard_notes), round(standard_notes[-1][2],2)

# if __name__ == "__main__":
#     examples = [
#         ['resources/examples/example1.wav', 'AP你要相信AP相信我们会像童话故事里AP',
#          'rest | G#3 | A#3 C4 | D#4 | D#4 F4 | rest | E4 F4 | F4 | D#4 A#3 | A#3 | A#3 | C#4 | B3 C4 | C#4 | B3 C4 | A#3 | G#3 | rest',
#          '0.14 | 0.47 | 0.1905 0.1895 | 0.41 | 0.3005 0.3895 | 0.21 | 0.2391 0.1809 | 0.32 | 0.4105 0.2095 | 0.35 | 0.43 | 0.45 | 0.2309 0.2291 | 0.48 | 0.225 0.195 | 0.29 | 0.71 | 0.14'],
#         ['resources/examples/example2.wav', 'AP半醒着AP笑着哭着都快活AP',
#          'rest | D4 | B3 | C4 D4 | rest | E4 | D4 | E4 | D4 | E4 | E4 F#4 | F4 F#4 | rest',
#          '0.165 | 0.45 | 0.53 | 0.3859 0.2441 | 0.35 | 0.38 | 0.17 | 0.32 | 0.26 | 0.33 | 0.38 0.21 | 0.3309 0.9491 | 0.125'],
#         ['resources/examples/example3.wav', 'SP一杯敬朝阳一杯敬月光AP',
#          'rest | G#3 | G#3 | G#3 | G3 | G3 G#3 | G3 | C4 | C4 | A#3 | C4 | rest',
#          '0.33 | 0.26 | 0.23 | 0.27 | 0.36 | 0.3159 0.4041 | 0.54 | 0.21 | 0.32 | 0.24 | 0.58 | 0.17']
#     ]
#     score, visualizations = analyze("../../resources/examples/example1.wav", examples[0][1],
#                                                                            examples[0][2], examples[0][3],threshold=5)
