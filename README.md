# Singaporean_English_ASR

This repository contains scripts, notebooks, and data for fine-tuning and evaluating speech recognition models, specifically **wav2vec2** and **Whisper**.

Objective:
- Develop a speech-to-text (STT) model that accurately transcribes English spoken with a Singaporean accent.
  The focus is on reliability, accuracy, and adaptability to Singaporean speech patterns.

Challenge Scope:
1. How do we build the most accurate and high-performing model possible?
2. How can we be absolutely sure it works reliably in real-world deployment tomorrow?

## Directory Structure

### 1. **data/**
Contains datasets for training and evaluation.

- **processed/**: Preprocessed data ready for model fine-tuning or evaluation.
- **raw/**: Raw unprocessed datasets.
- **raw_splits/**: Splits of raw data for training, validation, and testing.
- **scripts/**: Scripts for downloading dataset and data preprocessing


### 2. **evaluation/**
Notebooks for evaluating the performance of the models.

- **wav2vec2_evaluation.ipynb**: Evaluation of the fine-tuned wav2vec2 model.
- **whisper_evaluation.ipynb**: Evaluation of the fine-tuned Whisper model.


### 3. **output/**
Stores outputs from training and evaluation.

- **evaluations/**:
  - **wav2vec2/**: Evaluation results for wav2vec2
  - **whisper/**: Evaluation results for Whisper.
- **models/**:
  - **wav2vec2/**: Saved wav2vec2 model checkpoints.
  - **whisper/**: Saved Whisper model checkpoints.

### 4. **training/**
Notebooks and resources for fine-tuning models.

- **wav2vec2/**:
  - **finetune-wav2vec2-sample.ipynb**: Sample notebook demonstrating wav2vec2 fine-tuning on a small dataset.
  - **finetune-wav2vec2.ipynb**: Full-scale notebook for fine-tuning wav2vec2.
  - **vocab.json**: Vocabulary file for wav2vec2.

- **whisper/**: Resources for Whisper fine-tuning.


### 5. **inference/**
Python Scripts to infer sample audio file in both Whisper and Wav2Vec2


## Getting Started

### Prerequisites
1. Python 3.8+
2. Install dependencies using the provided `requirements.txt` file:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use .venv\Scripts\activate
    pip install -r requirements.txt
    ```

## Usage
1. Fine-tuning
Navigate to the training/ directory and open the respective notebook for the model you want to fine-tune (e.g., finetune-wav2vec2.ipynb).

2. Evaluation
Navigate to the evaluation/ directory and open the evaluation notebook for the model (e.g., wav2vec2_evaluation.ipynb).

3. Outputs
All fine-tuned models are saved in the output/models/ directory.
Evaluation results (e.g., WER, CER) are stored in the output/evaluations/ directory.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Hugging Face Transformers for wav2vec2 and Whisper implementations.
OpenAI and Meta for providing state-of-the-art speech models.
