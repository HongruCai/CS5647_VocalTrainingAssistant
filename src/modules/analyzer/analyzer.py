import whisper
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict_and_save
import mido
import pandas as pd
import glob
import os
import csv
import matplotlib.pyplot as plt
from io import BytesIO


def melody_analysis(file):
    model_output, midi_data, note_events = predict(file)
    return model_output, midi_data, note_events


def melody_analysis_and_save(file_list, output_dir):
    predict_and_save(
        file_list,
        output_dir,
        save_midi=False,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=True,
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

    for key, (lyric, pitches, durations) in data.items():
        for pitch, duration in zip(pitches, durations):
            start_time = current_time
            end_time = start_time + float(duration)
            result.append((pitch, start_time, end_time))
            current_time += float(duration)

    return result


def plot_pitch_variation(data, color):
    for pitch, start, end in data:
        plt.plot([start, end], [pitch, pitch], color=color, marker='o', linestyle='-')


def visualize(standard, user):
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
    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(times, pitches, marker='o', color='red')
    plt.plot(times2, pitches2, marker='o', color='blue')
    plt.legend(['User Singing', 'Standard Music'])
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (MIDI)')
    plt.grid(True)

    return plt


def analyze(audio_file, sheet_text, sheet_note, sheet_duration, threshold=3):
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
    start = 0
    end = 0
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
    user_start_time = user_notes[0][1]
    print(user_start_time)
    print(user_notes)
    for i in range(len(user_notes)):
        user_notes[i] = (user_notes[i][0], user_notes[i][1] - user_start_time, user_notes[i][2] - user_start_time)
    print(user_notes)
    standard_notes = convert_data(standard)
    print(standard_notes)
    score = calculate_score(standard_notes, user_notes, threshold=threshold)
    visualizations = visualize(standard_notes, user_notes)
    os.remove(notes_file[0])
    return score, visualizations

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
