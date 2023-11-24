import gradio as gr
from analyzer.analyzer import analyze
from tasinger.tasinger import TASinger
from visualizer.visualizer import generate_video

tasinger = TASinger("m4singer_diff_e2e")


'''main function to generate'''
def generate(sheet_text, sheet_note, sheet_duration, singer, mode):
    audio = tasinger.singing(singer + "-1", sheet_text, sheet_note, sheet_duration)
    if mode == "Audio":
        return audio, None
    else:
        video = generate_video(audio, sheet_text, sheet_note, sheet_duration)
        return None, video


if __name__ == "__main__":
    '''examples for testing and for users to start with'''
    examples = [
        ['resources/examples/example1.wav', 'AP可是你在敲打AP我的窗棂AP',
         'rest | G#3 | B3 | B3 C#4 | E4 | C#4 B3 | G#3 | rest | C3 | E3 | B3 G#3 | F#3 | rest',
         '0.2 | 0.38 | 0.48 | 0.41 0.72 | 0.39 | 0.5195 0.2905 | 0.5 | 0.33 | 0.4 | 0.31 | 0.565 0.265 | 1.15 | 0.24'],
        ['resources/examples/example2.wav', 'AP你要相信AP相信我们会像童话故事里AP',
         'rest | G#3 | A#3 C4 | D#4 | D#4 F4 | rest | E4 F4 | F4 | D#4 A#3 | A#3 | A#3 | C#4 | B3 C4 | C#4 | B3 C4 | A#3 | G#3 | rest',
         '0.14 | 0.47 | 0.1905 0.1895 | 0.41 | 0.3005 0.3895 | 0.21 | 0.2391 0.1809 | 0.32 | 0.4105 0.2095 | 0.35 | 0.43 | 0.45 | 0.2309 0.2291 | 0.48 | 0.225 0.195 | 0.29 | 0.71 | 0.14'],
        ['resources/examples/example3.wav', 'AP半醒着AP笑着哭着都快活AP',
         'rest | D4 | B3 | C4 D4 | rest | E4 | D4 | E4 | D4 | E4 | E4 F#4 | F4 F#4 | rest',
         '0.165 | 0.45 | 0.53 | 0.3859 0.2441 | 0.35 | 0.38 | 0.17 | 0.32 | 0.26 | 0.33 | 0.38 0.21 | 0.3309 0.9491 | 0.125'],
        ['resources/examples/example4.wav', 'SP乱石穿空AP惊涛拍岸AP',
         'rest | C#5 | D#5 | F5 D#5 | C#5 | rest | C#5 | C#5 | C#5 G#4 | G#4 | rest',
         '0.325 | 0.75 | 0.54 | 0.48 0.55 | 1.38 | 0.31 | 0.55 | 0.48 | 0.4891 0.4709 | 1.15 | 0.22']
    ]

    '''main interface'''
    vta = gr.Blocks(theme=gr.themes.Soft(), title="CS5647 Project: Vocal Training Assistant")

    with vta:
        '''title'''
        gr.Label("CS5647 Course Project: Vocal Training Assistant 📣", container=True, label="Team 9: Cai Hongru, Xiu Jingqiao, Zhou Zheng")
        '''singing analyzer'''
        gr.Label("🔎 Singing Analysis Module 🎼", show_label=False)

        '''input and output'''
        with gr.Row():
            with gr.Column(scale=1):
                '''input of audio file'''
                audio_file = gr.Audio(scale=1, type="filepath", label="Upload or Record Your Audio File Here", min_length=3)
                '''input of standard music sheet'''
                sheet_text = gr.Textbox(scale=2, lines=2, label="Input Music Sheet Texts Here")
                sheet_note = gr.Textbox(scale=2, lines=2, label="Input Music Sheet Notes Here")
                sheet_duration = gr.Textbox(scale=2, lines=2, label="Input Music Sheet Durations Here")
            with gr.Column(scale=1):
                with gr.Row():
                    '''analyzer output'''
                    ana_visualizations = gr.Plot(scale=1, label="Analysis Visualization")
                with gr.Row():
                    '''analyzer output'''
                    with gr.Column(scale=1):
                        user_total_notes = gr.Textbox(scale=1, lines=1, label="User Voice Total Notes")
                        user_total_duration = gr.Textbox(scale=1, lines=1, label="User Voice Total Duration (s)")
                        reference_total_notes = gr.Textbox(scale=1, lines=1, label="Reference Total Notes")
                        reference_total_duration = gr.Textbox(scale=1, lines=1, label="Reference Total Duration (s)")
                    with gr.Column(scale=1):
                        score = gr.Textbox(scale=1, lines=1, label="Overall Score (%)")
                        pitch_accuracy = gr.Textbox(scale=1, lines=1, label="Pitch Accuracy (%)")
                        rhythm_accuracy = gr.Textbox(scale=1, lines=1, label="Rhythm Accuracy (%)")
                        duration_accuracy = gr.Textbox(scale=1, lines=1, label="Duration Accuracy (%)")
        '''parameters for analyzer'''
        with gr.Row():
            velocity = gr.Slider(scale=1, minimum=0, maximum=100, step=5, label="Minimum Velocity to Take as Start", value=20)
            score_mode = gr.Dropdown(label="Score Mode", value="Reference Pitch", choices=["Reference Pitch", "Pitch Shift"])
            threshold1 = gr.Slider(scale=1, minimum=0, maximum=5, step=1, label="Tolerance of Pitch when Calculating Score", value=1)
        with gr.Row():
            threshold2 = gr.Slider(scale=1, minimum=0, maximum=5, step=1, label="Tolerance of Pitch when Calculating Pitch Accuracy",
                                   value=1)
            tolerance1 = gr.Slider(scale=1, minimum=0, maximum=0.5, step=0.05,
                                   label="Tolerance of Time when Calculating Rhythm Accuracy (s)",
                                   value=0.1)
            tolerance2 = gr.Slider(scale=1, minimum=0, maximum=0.5, step=0.05,
                                   label="Tolerance of Time when Calculating Duration Accuracy (s)",
                                   value=0.1)
        '''analyzer buttons'''
        with gr.Row():
            c1 = gr.ClearButton(scale=1, value="Clear Input", components=[audio_file, sheet_text, sheet_note, sheet_duration])
            c2 = gr.ClearButton(scale=1, value="Clear Analysis",
                                components=[score, ana_visualizations, pitch_accuracy, rhythm_accuracy, duration_accuracy,user_total_notes,
                                            user_total_duration,reference_total_notes,reference_total_duration])
            b1 = gr.Button(scale=1, value="Analyze My Audio")

        '''reference generation'''''
        singers = ['Tenor', 'Soprano', 'Bass', 'Alto']
        modes = ["Audio", "Video"]

        gr.Label("🎵 Reference Generation Module 🎶", show_label=False)
        '''reference generation parameters'''
        with gr.Row():
            drop1 = gr.Dropdown(label="Select a Singer", value=singers[0], choices=singers)
            drop2 = gr.Dropdown(label="Select Audio or Video to Generate", value=modes[0], choices=modes)
            b2 = gr.Button("Generate")
        '''svs output'''
        standard_audio = gr.Audio(label="Generated Reference Audio")
        '''video output'''
        audio_visualization = gr.Video(label="Generated Reference Video")
        '''Examples'''
        gr.Examples(fn=analyze, inputs=[audio_file, sheet_text, sheet_note, sheet_duration],
                    examples=examples, label="Some Examples to Start With")

        gr.Markdown(
            '''## Detailed Explanation of Parameters
            ### Singing Analysis Module
            - **Velocity**: When the volume of a certain pitch is greater than threshold, will be regarded as the beginning of the user's singing, default to 20
            - **Score Mode**: Two modes to calculate overall score, Reference Pitch means demand the user singing pitches exactly same to reference pitches
            , Pitch Shift allows pitch shift
            - **Tolerance of Pitch when Calculating Score**: abs(user_pitch-reference_pitch) <= tolerance are all valid, default to 1
            - **Tolerance of Pitch when Calculating Pitch Accuracy**: abs(single_user_pitch-single_reference_pitch) <= tolerance are all valid, default to 1
            - **Tolerance of Time when Calculating Rhythm Accuracy**: abs(user_start_time - reference_start_time) <= time_tolerance, default to 0.1
            - **Tolerance of Time when Calculating Duration Accuracy**: abs(user_duration - reference_duration) <= time_tolerance, default to 0.1
            ### Reference Generation Module
            - **Select a Singer**: Select a singer to generate reference, default to Tenor
            - **Select Audio or Video to Generate**: Select to generate audio or video, default to Audio
                    ''')
        '''button functions'''
        b1.click(analyze,
                 inputs=[audio_file, sheet_text, sheet_note, sheet_duration, threshold1, threshold2, tolerance1, tolerance2, velocity,
                         score_mode],
                 outputs=[score, ana_visualizations, pitch_accuracy, rhythm_accuracy, duration_accuracy, user_total_notes,
                          user_total_duration, reference_total_notes, reference_total_duration])
        b2.click(generate, inputs=[sheet_text, sheet_note, sheet_duration, drop1, drop2], outputs=[standard_audio, audio_visualization])

    vta.launch(share=True)
