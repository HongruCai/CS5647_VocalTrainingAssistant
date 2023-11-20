import gradio as gr
from modules.analyzer.analyzer import analyze


def generate(sheet_text, sheet_note, sheet_duration, drop1, drop2):
    if drop2 == "Audio":
        audio = generate_audio(sheet_text, sheet_note, sheet_duration, drop1)
        return audio, None
    else:
        video = generate_video(sheet_text, sheet_note, sheet_duration, drop1)
        return None, video


def generate_audio(sheet_text, sheet_note, sheet_duration, drop1):
    return "This is a test"


def generate_video(sheet_text, sheet_note, sheet_duration, drop1):
    return "This is a test"


if __name__ == "__main__":
    vts = gr.Blocks(theme='soft', mode='Vocal Training Assistant', title="CS5647 Project: Vocal Training Assistant")

    examples = [
        ['resources/examples/example1.wav', 'AP你要相信AP相信我们会像童话故事里AP',
         'rest | G#3 | A#3 C4 | D#4 | D#4 F4 | rest | E4 F4 | F4 | D#4 A#3 | A#3 | A#3 | C#4 | B3 C4 | C#4 | B3 C4 | A#3 | G#3 | rest',
         '0.14 | 0.47 | 0.1905 0.1895 | 0.41 | 0.3005 0.3895 | 0.21 | 0.2391 0.1809 | 0.32 | 0.4105 0.2095 | 0.35 | 0.43 | 0.45 | 0.2309 0.2291 | 0.48 | 0.225 0.195 | 0.29 | 0.71 | 0.14'],
        ['resources/examples/example2.wav', 'AP半醒着AP笑着哭着都快活AP',
         'rest | D4 | B3 | C4 D4 | rest | E4 | D4 | E4 | D4 | E4 | E4 F#4 | F4 F#4 | rest',
         '0.165 | 0.45 | 0.53 | 0.3859 0.2441 | 0.35 | 0.38 | 0.17 | 0.32 | 0.26 | 0.33 | 0.38 0.21 | 0.3309 0.9491 | 0.125'],
        ['resources/examples/example3.wav', 'SP一杯敬朝阳一杯敬月光AP',
         'rest | G#3 | G#3 | G#3 | G3 | G3 G#3 | G3 | C4 | C4 | A#3 | C4 | rest',
         '0.33 | 0.26 | 0.23 | 0.27 | 0.36 | 0.3159 0.4041 | 0.54 | 0.21 | 0.32 | 0.24 | 0.58 | 0.17']
    ]

    with vts:
        gr.Markdown(
            """
            # CS5647 Course Project: Vocal Training Assistant
            ## Team 9: Cai Hongru, Xiu Jingqiao, Zhou Zheng
            
            """)
        gr.Markdown(
            """
            ## Analyze Singing Performance
            """)
        with gr.Row():
            with gr.Column():
                '''input of audio file'''
                audio_file = gr.Audio(scale=1, type="filepath", label="Upload or Record Your Audio File Here")
            with gr.Column():
                '''analyzer output'''
                ana_visualizations = gr.Plot(scale=1, label="Analysis Visualization")
        with gr.Row():
            with gr.Column():
                '''input of standard music sheet'''
                sheet_text = gr.Textbox(lines=1, label="Input Music Sheet Texts Here")
                sheet_note = gr.Textbox(lines=1, label="Input Music Sheet Notes Here")
                sheet_duration = gr.Textbox(lines=1, label="Input Music Sheet Durations Here")
            with gr.Column():
                '''analyzer output'''
                score = gr.Textbox(lines=1, label="Overall Score")
                pitch_accuracy = gr.Textbox(lines=1, label="Pitch Accuracy")
                pitch_length_accuracy = gr.Textbox(lines=1, label="Rhythm and Duration Accuracy")
        with gr.Row():
            threshold = gr.Slider(minimum=0, maximum=10, step=1, label="Threshold")
            b1 = gr.Button("Analyze My Audio")
        gr.Markdown(
            """
            ## Generate Reference Audio or Video
            """)
        with gr.Row():
            drop1 = gr.Dropdown(label="Select a Singer", choices=["Male-1", "Male-2", "Female-1", "Female-2"])
            drop2 = gr.Dropdown(label="Select Audio or Video to Generate", choices=["Audio", "Video"])
            b2 = gr.Button("Generate")
        '''svs output'''
        standard_audio = gr.Audio(label="Generated Reference Audio")
        '''visualizer output'''
        audio_visualization = gr.Video(label="Generated Reference Video")
        gr.Examples(fn=analyze, inputs=[audio_file, sheet_text, sheet_note, sheet_duration],
                    outputs=[score, pitch_accuracy, pitch_length_accuracy, ana_visualizations],
                    examples=examples, label="Some Examples to Start With")

        b1.click(analyze, inputs=[audio_file, sheet_text, sheet_note, sheet_duration, threshold],
                 outputs=[score, ana_visualizations])
        b2.click(generate, inputs=[sheet_text, sheet_note, sheet_duration, drop1, drop2], outputs=[standard_audio, audio_visualization])

    vts.launch()
