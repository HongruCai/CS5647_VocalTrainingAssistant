'''
This file is used to analyze the user singing and calculate the score and accuracy.
Author: Cai Hongru
'''

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

import numpy as np
import tensorflow as tf
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

matplotlib.rcParams['font.family'] = 'SimHei'
basic_pitch_model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

'''use pitch basic to analyze the audio file and save the output to the output_dir'''
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


'''transform the note to midi number'''
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


'''calculate the score of the user singing by comparing the standard and user singing notes in the same time interval'''
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


'''convert the standard music sheet to the format of (pitch, start_time, end_time)'''
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


'''calculate the accuracy of pitch, rhythm and duration'''
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


'''visualize the standard and user singing notes'''
def visualize(stan,standard, user):
    times = []
    pitches = []
    for pitch, start, end in user:
        times.extend([start, end])
        # pitches.extend([pitch, pitch])
        pitches.extend([pitch if pitch != 0 else np.nan, pitch if pitch != 0 else np.nan])
    times2 = []
    pitches2 = []
    for pitch, start, end in standard:
        # if pitch == 0:
        #     continue
        times2.extend([start, end])
        # pitches2.extend([pitch, pitch])
        pitches2.extend([pitch if pitch != 0 else np.nan, pitch if pitch != 0 else np.nan])
    plt.figure(figsize=(8, 4))
    for lyric, start, end in stan:
        for pitch1, start1, end1 in standard:
            if start == start1:
                if lyric != "AP" and lyric != "SP":
                    plt.text(start, pitch1, lyric, fontsize=15)
                    break
                    # print(lyric, start, pitch1)

    plt.plot(times2, pitches2, marker='o', color='blue')
    plt.plot(times, pitches, marker='o', color='red')
    plt.legend(['User Singing', 'Reference'])
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (MIDI)')
    plt.grid(True)

    return plt


'''main function to analyze the user singing, calculate the score and accuracy and visualize the result'''
def analyze(audio_file, sheet_text, sheet_note, sheet_duration, threshold1, threshold2, tolerance1, tolerance2):
    output_dir = 'resources/intermediate_files'
    melody_analysis_and_save([audio_file], output_dir)

    '''extract the notes from the music sheet'''
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

    '''extract the notes from the user's output file'''
    user_notes = []
    with open(notes_file[0], newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            if row[0] == 'start_time_s':
                continue
            velocity = float(row[3])
            if velocity >= 40:
                start_time_s = float(row[0])
                end_time_s = float(row[1])
                pitch_midi = int(row[2])
                user_notes.append((pitch_midi, start_time_s, end_time_s))
    user_notes = sorted(user_notes, key=lambda x: x[1])

    '''convert to same format'''
    standard_notes,standard_lyrics = convert_data(standard)
    user_start_time = user_notes[0][1]
    standard_start_time = standard_notes[0][2]
    for i in range(len(user_notes)):
        user_notes[i] = (user_notes[i][0], user_notes[i][1] - user_start_time + standard_start_time,
                         user_notes[i][2] - user_start_time + standard_start_time)

    '''calculate the score and accuracy'''
    score = calculate_score(standard_notes, user_notes, threshold=threshold1)
    score = round(score * 100, 2)
    pitch_accuracy, rhythm_accuracy, duration_accuracy = calculate_accuracy(standard_notes, user_notes, time_tolerance1=tolerance1,
                                                                            time_tolerance2=tolerance2,
                                                                            threshold=threshold2)

    '''visualize the result'''
    visualizations = visualize(standard_lyrics,standard_notes, user_notes)

    os.remove(notes_file[0])
    return score, visualizations, pitch_accuracy, rhythm_accuracy, duration_accuracy, len(user_notes), round(user_notes[-1][2],2), len(
        standard_notes), round(standard_notes[-1][2],2)
