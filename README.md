# Symbolic Music Generation - Text to MIDI

**MSc Applied Artificial Intelligence - ICESI University**  
**Project: Generación Musical con IA de Texto a MIDI**

## Overview

This repository implements a text-to-MIDI generation system that creates symbolic music from natural language descriptions. The project replicates and extends the methodology from the [Text2midi paper](http://arxiv.org/abs/2412.16526), enabling the generation of MIDI files from textual captions that describe musical characteristics such as mood, genre, tempo, key, and instrumentation.

## Key Features

- **Text-to-MIDI Generation**: Transform natural language descriptions into symbolic music (MIDI format)
- **Multi-Dataset Support**: Works with SymphonyNet and MidiCaps datasets
- **Exploratory Data Analysis**: Comprehensive analysis of musical datasets including genre, mood, tempo, and instrumentation patterns
- **Model Training Pipeline**: Complete implementation of the Text2midi architecture using FlanT5 encoder and REMI+ tokenization

## Project Structure

```
symbolic-music-generation/
├── notebooks/
│   ├── eda.ipynb              # Exploratory data analysis of musical datasets
│   └── text2midi_train.ipynb  # Model training replication notebook
├── main.py                     # Entry point (placeholder)
├── pyproject.toml              # Project dependencies
└── README.md                   # This file
```

## Installation

This project requires Python 3.12 or higher. Install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
# Clone the repository
git clone https://github.com/NickEsColR/symbolic-music-generation.git
cd symbolic-music-generation

# Install dependencies with uv
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt  # If available
```

### Key Dependencies

- **PyTorch**: Deep learning framework
- **Transformers**: For FlanT5 encoder model
- **miditok**: MIDI tokenization with REMI+ support
- **music21**: Music analysis and MIDI processing
- **datasets**: Dataset loading and processing
- **pandas, matplotlib, seaborn**: Data analysis and visualization

## Usage

### 1. Exploratory Data Analysis

The `notebooks/eda.ipynb` notebook provides comprehensive analysis of two key datasets:

- **SymphonyNet Dataset**: 46,359 MIDI files with 279 million notes covering symphonic music
- **MidiCaps Dataset**: MIDI files with detailed text captions describing musical characteristics

Run the notebook to explore:
- Dataset statistics and distributions
- Genre and mood analysis
- Key signatures and time signatures
- Instrumentation patterns
- Caption analysis

### 2. Model Training

The `notebooks/text2midi_train.ipynb` notebook replicates the Text2midi training pipeline:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NickEsColR/symbolic-music-generation/blob/main/notebooks/text2midi_train.ipynb)

**Training Pipeline:**

1. **Dataset Preparation**: Preprocess SymphonyNet dataset using music21 to extract tempo, key, BPM, and instruments
2. **Pseudo-Caption Generation**: Create template-based captions for pre-training
3. **Model Architecture**:
   - Encoder: FlanT5-base for text processing
   - Tokenizer: REMI+ for MIDI tokenization
   - Decoder: Processes encoded text and tokenized MIDI
4. **Training**: Complete training loop with the prepared datasets

The notebook can be run directly in Google Colab using the badge above.

## Datasets

### SymphonyNet
- **Size**: 46,359 MIDI files
- **Duration**: 3,284 hours of music
- **Content**: Symphonic music with multiple instrument tracks
- **Reference**: [SymphonyNet Dataset](https://symphonynet.github.io/)

### MidiCaps
- **Features**: MIDI files with rich text captions
- **Metadata**: Genre, mood, tempo, key, time signature, chord progressions, instrumentation
- **Split**: 90/10 train/test partition
- **Reference**: [MidiCaps on HuggingFace](https://huggingface.co/datasets/amaai-lab/MidiCaps)

## Model Architecture

The project implements the Text2midi architecture:

- **Encoder**: Google FlanT5-base model for encoding text captions
- **MIDI Tokenization**: REMI+ (REpresentation of Music in Integer) format
- **Decoder**: Transformer-based decoder for generating MIDI sequences
- **Training**: Two-stage approach with pre-training on SymphonyNet and fine-tuning on MidiCaps

## References

- **Text2midi Paper**: [Text2midi: Generating Symbolic Music from Captions](http://arxiv.org/abs/2412.16526)
- **Pre-trained Model**: [text2midi on HuggingFace](https://huggingface.co/amaai-lab/text2midi)
- **SymphonyNet Dataset**: [https://symphonynet.github.io/](https://symphonynet.github.io/)
- **MidiCaps Dataset**: [https://huggingface.co/datasets/amaai-lab/MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps)

## Academic Context

This project is part of the MSc in Applied Artificial Intelligence program at Universidad ICESI, Colombia. It focuses on symbolic music generation using deep learning techniques to bridge natural language and musical representation.

## License

Please refer to the original Text2midi paper and dataset licenses for usage restrictions.
