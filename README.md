# CS5647 Course Project: Vocal Training Assistant

Team 9: Cai Hongru, Xiu Jingqiao, Zhou Zheng

Vocal Training Assistant is an innovative tool designed to assist in vocal training using advanced machine learning models. It offers
features for analyzing user singing performance and generating reference audio or video for practice.

## Table of Contents
- [Modules](#modules)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [License](#license)

## Modules

- Singing Analysis Module![ana.png](assets%2Fana.png)
  - Analyze your singing performance by comparing it with standard music sheets.
  - Overall score, pitch accuracy, rhythm accuracy and duration accuracy are provided.

    
- Reference Generation Module![gen.png](assets%2Fgen.png)
  - TODO
  - TODO

## Features

* **Singing Analysis**: Analyze your singing performance by comparing it with standard music sheets.
* **Audio and Video Generation**: Generate reference audio or video based on input music sheets.
* **Customization**: Choose different singers and modes for reference generation.
* **Gradio Interface**: User-friendly Gradio interface for easy interaction.


## Installation

To get started with the Vocal Training Assistant, follow these installation steps:

```bash
# Clone the repository
git clone https://github.com/HongruCai/CS5647_VocalTrainingAssistant.git

# Navigate to the project directory
cd CS5647_VocalTrainingAssistant

# Install dependencies
pip install -r requirements.txt
```
Download the checkpoint files:

Download from [Google Drive](https://drive.google.com/file/d/1DxPb9xXYvObEuVGBEokWQBqM6G5R1Gql/view?usp=drive_link)
and put them in the checkpoints folder.


## Quick Start

To quickly start using the Vocal Training Assistant, run the following command:

```bash
python main.py
```

## Usage

To use the Vocal Training Assistant, navigate through the Gradio interface. You can upload or record your audio, input music sheet details,
and choose to analyze your singing or generate reference material.


## License

We agree to the license: CC BY-NC-SA 4.0 (NonCommercial!).

This project uses the following tools or repositories:

* [gradio-app](https://www.gradio.app/)
* [basic-pitch](https://github.com/spotify/basic-pitch)
* [m4singer](https://github.com/M4Singer/M4Singer)
* TODO
