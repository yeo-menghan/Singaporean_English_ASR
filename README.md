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

## Preliminary Results
### Key Findings
- The fine-tuned model achieved a **21.05% Word Error Rate (WER)**, compared to **122.81%** for the base model.
- **Character Error Rate (CER)** improved from **123.61%** to **6.94%**.
- **Inference time** decreased by **58.5%** (from **0.423s** to **0.176s** per sample).
- The fine-tuned model handled Singaporean English accents significantly better.


## Test data used:
<details>
<summary>Test Data, Click to expand</summary>

```json
[
  {
    "text": "trading halt has turned into a suspension pending the release of an announcement.",
    "path": "../processed\\test\\SP0873-CH00-SE01-RC755.flac"
  },
  {
    "text": "the shop keeps its costs low by buying its supplies in bulk.",
    "path": "../processed\\test\\SP0873-CH00-SE01-RC451.flac"
  },
  {
    "text": "regardless, as free parking on sunday and in general becomes increasingly endangered, so might our delicate social fabric.",
    "path": "../processed\\test\\SP0402-CH00-SE00-RC209.flac"
  },
  {
    "text": "many countries require a passport to be valid for at least six months as part of their entry requirements.",
    "path": "../processed\\test\\SP0835-CH00-SE00-RC397.flac"
  },
  {
    "text": "previously, there was no restriction.",
    "path": "../processed\\test\\SP0566-CH00-SE00-RC113.flac"
  },
  {
    "text": "most youths today think that their future is uncertain.",
    "path": "../processed\\test\\SP0681-CH00-SE00-RC095.flac"
  },
  {
    "text": "how else to do so other than being dapper?",
    "path": "../processed\\test\\SP0838-CH00-SE01-RC732.flac"
  },
  {
    "text": "it will improve your experience the more it knows you.",
    "path": "../processed\\test\\SP0681-CH00-SE00-RC397.flac"
  },
  {
    "text": "set a stop-loss order of 50% for my current order.",
    "path": "../processed\\test\\SP0566-CH00-SE00-RC213.flac"
  },
  {
    "text": "a smile can often lift up a weary spirit.",
    "path": "../processed\\test\\SP0873-CH00-SE01-RC401.flac"
  }
]
```
</details>

## Metric Definitions

| Metric                  | Definition                                                   |
|-------------------------|-------------------------------------------------------------|
| Word Error Rate (WER)   | Measures the percentage of words that were incorrectly recognized. |
| Character Error Rate (CER) | Measures the percentage of characters that were incorrectly recognized. |
| Inference Time          | The time taken to process and transcribe audio samples.     |

## Model Performance Comparison

| Model                   | WER     | CER     | Inference Time |
|-------------------------|---------|---------|-----------------|
| Whisper Base            | 122.81  | 123.61  | 0.423           |
| Whisper Base Fine-Tuned | 21.05   | 6.94    | 0.176           |
| **Improvement**         | 101.75  | 116.67  | 0.247 (58.5%)   |

### Error Analysis
- The base model performed poorly on strongly accented Singaporean English, often transcribing text in Malay or producing repetitive patterns.
- The fine-tuned model significantly improved accuracy but still faced challenges with certain domain-specific terms and rare words.

### Cases Where Fine-tuned Model Performed Better
| Audio Reference                       | Base Model (WER) | Fine-Tuned (WER) | Improvement     |
|---------------------------------------|-------------------|-------------------|------------------|
| SP0681-CH00-SE00-RC095.flac           | 500.00%           | 22.22%            | 477.78%          |
| SP0835-CH00-SE00-RC397.flac           | 226.32%           | 5.26%             | 221.05%          |

### Cases Where Fine-tuned Model Performed Worse
| Audio Reference                       | Base Model (WER) | Fine-Tuned (WER) | Improvement     |
|---------------------------------------|-------------------|-------------------|------------------|
| SP0873-CH00-SE01-RC401.flac           | 44.44%            | 77.78%            | -33.33%          |
| SP0873-CH00-SE01-RC451.flac           | 16.67%            | 33.33%            | -16.67%          |

### Edge Case Analysis
- **Accent Variations**: The fine-tuned model showed decent improvement in handling strongly accented Singaporean English, which the base model often misinterpreted.
- **Domain-Specific Terminology**: Significant improvement was observed in financial and technical terms.
- **Background Noise and Audio Quality**: This remains an area for potential further improvement.

## Note
Please contact the author to retrieve the following items:
- Model checkpoints for Whisper and Wav2Vec2 (under `output/models`)
- Data used in the original test of the scripts under `data/processed`, `data/raw`, `data/raw_splits`

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Hugging Face Transformers for wav2vec2 and Whisper implementations.
OpenAI and Meta for providing state-of-the-art speech models.
