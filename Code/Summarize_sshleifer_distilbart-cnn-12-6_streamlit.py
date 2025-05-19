import whisper
import subprocess
import tempfile
import os
import ffmpeg
import pathlib
from transformers import pipeline, AutoTokenizer
import streamlit as st


allowed_extensions = ["mp3", "mp4", "wav"]
allowed_mime_types = ["audio/mpeg", "audio/wav", "video/mp4"]
ALLOWED_EXTEN = (".mp3", ".mp4", ".wav", ".flac")

# Load model and tokenizer once
@st.cache_resource

    
def transcribe_audio(audio_path):
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=audio_path.name[-4:]) as tmp_input:
        tmp_input.write(audio_path.read())
        input_audio_path = tmp_input.name

    # Convert to WAV if necessary
    if not audio_path.name.endswith(".wav"):
        #Creates a temporary file with .wav extension.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_output:
            converted_audio_path = tmp_output.name
        
        #Uses ffmpeg to convert the uploaded file to:
        #16 kHz sample rate (ar=16000)
        #Mono channel (ac=1)
        
        try:
            #subprocess.run(
            #    ["ffmpeg", "-i", input_audio_path, "-ar", "16000", "-ac", "1", "-y", converted_audio_path],
            #    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            #)
            (
                ffmpeg.input(input_audio_path).output(converted_audio_path, ar=16000, ac=1).run(quiet=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            st.error("FFmpeg failed to process the audio file.")
            os.unlink(input_audio_path)
            os.unlink(converted_audio_path)
            st.stop()
    else:
        converted_audio_path = input_audio_path
        
    model = whisper.load_model("base")
    result = model.transcribe(converted_audio_path)
            
    os.unlink(input_audio_path)
    if input_audio_path != converted_audio_path:
        os.unlink(converted_audio_path)
    
    return result


    

def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    #model_name = "facebook/bart-large-cnn"
    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return summarizer, tokenizer

def chunk_text(text, tokenizer, max_length=512, overlap=50):
    tokens = tokenizer(text, truncation=False, padding=False)["input_ids"]
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(chunk)
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def summarize_text(text, summarizer, tokenizer):
    chunks = chunk_text(text, tokenizer, max_length=512, overlap=50)
    summaries = []
    #for chunk in chunks:
    #    summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
    #    summaries.append(summary[0]['summary_text'])
    #return " ".join(summaries)

    for chunk in chunks:
        input_length = len(chunk.split())  # Count words
        max_length = max(int(input_length * 0.6), 30)  # 60% of input length, minimum 30 words
        
        # Ensure max_length does not exceed input_length - 1
        if max_length >= input_length:
            max_length = input_length - 1  

        min_length = min(30, input_length)  # Ensure min_length does not exceed input_length
        

        summary = summarizer(chunk, max_length=input_length - 1, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

# Initialize session state variables if not already
if "last_input_text" not in st.session_state:
    st.session_state["last_input_text"] = ""
if "last_summary" not in st.session_state:
    st.session_state["last_summary"] = ""


# Streamlit UI
st.title("📄 Text Summarizer using GenAI")
st.write("Choose an option to summarize your content (limited to 500 words).")
st.write(
    "This is not a full GenAI application for industry-scale text summarization. "
    "It demonstrates how to use a transformer-based model specialized for summarization, "
    "specifically the distilled BART model `sshleifer/distilbart-cnn-12-6`. "
    "Feel free to visit my [GitHub](https://github.com/spatelsuy/Text-Audio-Summarization-GenAI) for the code and more details."
)



option = st.radio("Choose input method:", ("Upload TXT File", "Enter Text Manually", "Audio to Text Summary"))

input_text = ""


if option == "Audio to Text Summary":
    uploaded_audio_file = st.file_uploader("Upload a file in MP3, MP4, or WAV format", type=ALLOWED_EXTEN)
    if uploaded_audio_file is not None:
        st.write(f"Uploaded file type: {uploaded_audio_file.type} : {uploaded_audio_file.name}")
        if uploaded_audio_file.type in allowed_mime_types:
            if st.button("Summarize audio"):
                audio_text = transcribe_audio(uploaded_audio_file)
                st.session_state["audio_text"] = audio_text["text"]  # Save to session state
        else:
            st.error("Invalid file type. Please upload only MP3, MP4, or WAV files.")
            input_text = ""
                
        if "audio_text" in st.session_state:
            st.write("Correct if any before clicking 'Summarize Text' button below")
            st.text_area("Transcribed Audio", value=st.session_state["audio_text"], height=300)
            if st.button("Summarize It"):
                input_text = st.session_state["audio_text"]


if option == "Upload TXT File":
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        word_count = len(text.split())
        if word_count < 100 or word_count > 500:
            st.error(f"File contains {word_count} words. Please enter word count between 100 and 500.")
        else:
            input_text = text

elif option == "Enter Text Manually":
    text = st.text_area("Enter text (≤500 words):", height=300)
    if st.button("Summarize Text"):
        with st.spinner("Summarizing the text..."):
            word_count = len(text.split())
            if word_count == 0:
                st.warning("Please enter some text.")
            elif word_count < 100 or word_count > 500:
                st.error(f"Input has {word_count} words. Please enter word count between 100 and 500.")
            else:
                input_text = text

            
# Summarize only if we have input_text
if input_text:
    if input_text == st.session_state["last_input_text"]:
        # Same input as before, show stored summary
        st.success("Using cached summary:")
        st.write(st.session_state["last_summary"])
    else:
        # New input, generate summary
        with st.spinner("Summarizing..."):
            summarizer, tokenizer = load_model()
            summary = summarize_text(input_text, summarizer, tokenizer)
            st.session_state["last_input_text"] = input_text
            st.session_state["last_summary"] = summary
            st.success("Summary generated:")
            st.write(summary)
            
    if "audio_text" in st.session_state:
        del st.session_state["audio_text"]
