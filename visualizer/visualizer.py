import itertools

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip
from scipy.io.wavfile import write

left_margin_ratio, right_margin_ratio = 0.1, 0.9
dpi = 300
pixels_per_sec = 1000
height_w = 0.3
char_offset = 0.5

def generate_video(audio, input_text, input_note, input_duration):
    # Split the input strings into lists
    characters = list(input_text.replace('AP', ' ').replace('SP', ' '))
    notes = [s.split() for s in input_note.split("|")]
    durations = [[float(num) for num in s.split()] for s in input_duration.split("|")]

    flat_notes = list(itertools.chain(*notes))
    flat_durations = list(itertools.chain(*durations))

    # Define the order of notes from lowest to highest
    all_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Find the unique notes in the input string, ignoring "rest"
    unique_notes = set(n for note in notes for n in flat_notes if n != "rest")

    # Find the lowest and highest notes
    lowest_note = min(unique_notes, key=lambda note: (int(note[-1]), all_notes.index(note[:-1])))
    highest_note = max(unique_notes, key=lambda note: (int(note[-1]), all_notes.index(note[:-1])))

    lowest_octave, highest_octave = int(lowest_note[-1]), int(highest_note[-1])
    lowest_note_name, highest_note_name = lowest_note[:-1], highest_note[:-1]
    note_order = []

    for octave in range(lowest_octave, highest_octave + 1):
        for note in all_notes:
            if octave == lowest_octave and all_notes.index(note) < all_notes.index(lowest_note_name):
                continue
            if octave == highest_octave and all_notes.index(note) > all_notes.index(highest_note_name):
                break
            note_order.append(note + str(octave))

    # Reverse the note order so that higher notes are higher in the plot
    note_order = note_order[::-1]

    # Generate a piano roll-like matrix
    total_duration = sum(flat_durations)
    piano_roll = np.zeros((len(note_order), int(total_duration * pixels_per_sec)))

    # Fill in the piano roll matrix
    current_time = 0
    for note, duration in zip(flat_notes, flat_durations):
        if note != "rest":
            start_time = int(current_time * pixels_per_sec)
            end_time = int((current_time + duration) * pixels_per_sec)
            if note in note_order:
                i = note_order.index(note)
                piano_roll[i, start_time:end_time] = 1
            current_time += duration
        else:
            current_time += duration

    # Create a custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["white", "blue"])

    # Plot the piano roll
    fig, ax = plt.subplots(figsize=(total_duration, len(note_order) * height_w))  # Adjust the figure size
    ax.imshow(piano_roll, aspect='auto', cmap=cmap, interpolation='none')  # Set interpolation to 'none'
    ax.get_xaxis().set_visible(False)
    # Adjust margins
    plt.subplots_adjust(left=left_margin_ratio, right=right_margin_ratio)

    # Add the text
    current_time = 0
    for char, duration in zip(characters, durations):
        ax.text(current_time * pixels_per_sec, len(note_order) + char_offset, char, ha='center', va='bottom')
        current_time += sum(duration)

    # Label the y-axis with the note names
    ax.set_yticks(range(len(note_order)))
    ax.set_yticklabels(note_order)

    plt.savefig('resources/intermediate_files/image.png', dpi=dpi)  # Adjust the DPI for better quality

    # Load the image
    img_clip = ImageClip('resources/intermediate_files/image.png')

    # Set the duration of the image clip
    img_clip = img_clip.set_duration(total_duration)

    # Create a moving vertical bar
    bar = np.zeros((img_clip.size[1], 10, 3), dtype=np.uint8)
    bar[:, :, 0] = 255  # Red bar
    bar_clip = ImageClip(bar)
    bar_clip = bar_clip.set_duration(img_clip.duration)

    # Animate the bar
    offset = total_duration * dpi * left_margin_ratio
    v = dpi * (right_margin_ratio - left_margin_ratio)
    bar_clip = bar_clip.set_position(lambda t: (v*t + offset, 0))

    # Overlay the bar on the image
    video = CompositeVideoClip([img_clip, bar_clip])
    
    # Create an audio clip from the wav array
    write('resources/intermediate_files/audio.wav', audio[0], audio[1])
    audio = AudioFileClip('resources/intermediate_files/audio.wav')

    # Set the audio of the video clip
    video = video.set_audio(audio)

    # Get the binary representation of the video
    video.write_videofile('resources/intermediate_files/video.mp4', fps=24, audio_codec='aac')

    return 'resources/intermediate_files/video.mp4'


