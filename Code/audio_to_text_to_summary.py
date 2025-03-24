import whisper
import subprocess
import argparse
import os
from transformers import pipeline, AutoTokenizer

ALLOWED_EXTENSIONS = (".mp3", ".mp4", ".wav", ".flac")

def transcribe_audio(audio_path):
    if not audio_path.lower().endswith(ALLOWED_EXTENSIONS):
        print("Error: Unsupported audio format. Use MP3, MP4, WAV, or FLAC.")
        return

    model = whisper.load_model("base")

    # Convert audio to WAV if not already in WAV format
    converted_audio = "converted_audio.wav"
    
    if not audio_path.lower().endswith(".wav"):
        try:
            subprocess.run(
                ["ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", "-y", converted_audio],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            print("Error: FFmpeg failed to process the audio file.")
            return
    else:
        converted_audio = audio_path

    result = model.transcribe(converted_audio)

    print("\nTranscribed Text: ", result["text"])

    with open("transcription.txt", "w", encoding="utf-8") as file:
        file.write(result["text"])


def chunk_text(text, tokenizer, max_length=512, overlap=50):
    """Splits text into tokenized chunks of max_length tokens with an overlap."""
    tokens = tokenizer(text, truncation=False, padding=False)["input_ids"]  # Tokenize full text
    chunks = []
    
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]  # Slice tokens within limit
        chunks.append(chunk)
    
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]  # Convert back to text
    

def summarize_text(transcript_file):
    model_name = "facebook/bart-large-cnn"  # Better summarization model
    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()

    # Tokenize and split into chunks
    chunks = chunk_text(transcript, tokenizer, max_length=512, overlap=50)

    summaries = []
    for chunk in chunks:
        input_length = len(chunk.split())  # Count words
        max_length = max(int(input_length * 0.6), 30)  # 60% of input length, minimum 30 words
        
        # Ensure max_length does not exceed input_length - 1
        if max_length >= input_length:
            max_length = input_length - 1  

        min_length = min(30, input_length)  # Ensure min_length does not exceed input_length
        

        summary = summarizer(chunk, max_length=input_length - 1, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    # Merge chunk summaries into a final summary
    final_summary = " ".join(summaries)
    
    return final_summary





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio to text using Whisper.")
    parser.add_argument("audio_file", type=str, help="Path to the input audio file")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print("Error: File does not exist!")
    else:
        transcribe_audio(args.audio_file)
        summary = summarize_text("transcription.txt")
        print("\n**Meeting Summary**\n", summary)

        # Save summary to file
        with open("Second_summary.txt", "w", encoding="utf-8") as f:
            f.write("**Meeting Summary**\n\n")
            f.write(summary)
