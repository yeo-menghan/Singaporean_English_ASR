import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import os

# Load the fine-tuned model and processor
model_name_or_path = "../output/models/wav2vec2/best_model" 
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and move to appropriate device
model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path)
model = model.to(device)

# Set the model to evaluation mode
model.eval()

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text using the Wav2Vec2 model.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Transcription of the audio.
    """
    # Load audio
    try:
        # Method 1: Using soundfile directly
        speech_array, sampling_rate = sf.read(audio_path)
        
        # If stereo, convert to mono by taking the mean of channels
        if len(speech_array.shape) > 1:
            speech_array = speech_array.mean(axis=1)
    except Exception as e:
        print(f"Error loading audio with soundfile: {e}")
        # Method 2: Fall back to datasets if soundfile fails
        from datasets import load_dataset
        audio = load_dataset("audiofolder", data_dir=os.path.dirname(audio_path), 
                             split="train", drop_labels=True)
        audio = audio[0]["audio"]
        speech_array = audio["array"]
        sampling_rate = audio["sampling_rate"]
    
    # Process audio
    inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Generate transcription
    with torch.no_grad():
        logits = model(**inputs).logits

    # Decode predictions
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

if __name__ == "__main__":
    sample_audio = "./sample/sample.flac"  
    print(f"Transcribing audio: {sample_audio}")
    transcription = transcribe_audio(sample_audio)
    print(f"Transcription: {transcription}")