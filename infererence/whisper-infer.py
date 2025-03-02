import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

# Load the fine-tuned model and processor
model_name_or_path = "../output/models/whisper/best_model"
processor = WhisperProcessor.from_pretrained(model_name_or_path)

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and move to appropriate device
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
model = model.to(device)

# Set the model to evaluation mode
model.eval()

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text using the Whisper model.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Transcription of the audio.
    """
    # Load audio
    from datasets import load_dataset
    audio = load_dataset("audiofolder", data_dir=os.path.dirname(audio_path), 
                         split="train", drop_labels=True)
    audio = audio[0]["audio"]

    # Process audio
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    input_features = input_features.to(device)

    # Generate transcription - handle the language detection issue from PR #28687
    with torch.no_grad():
        model.config.use_cache = True  
        # Fix for the language detection issue - specify English explicitly
        model.generation_config.language = "en"
        model.generation_config.task = "transcribe"
        
        generated_ids = model.generate(input_features=input_features)

    # Decode transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return transcription

if __name__ == "__main__":
    sample_audio = "./sample/sample.flac"  
    print(f"Transcribing audio: {sample_audio}")
    transcription = transcribe_audio(sample_audio)
    print(f"Transcription: {transcription}")