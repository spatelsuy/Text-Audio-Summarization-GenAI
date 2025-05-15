import streamlit as st
from transformers import pipeline, AutoTokenizer
import tempfile

# Load model and tokenizer once
@st.cache_resource
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
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

# Streamlit UI
st.title("📄 Text Summarizer using GenAI")
st.write("Choose an option to summarize your content (limited to 1000 words).")

option = st.radio("Choose input method:", ("Upload TXT File", "Enter Text Manually"))

summarizer, tokenizer = load_model()
input_text = ""
input_text_local = ""

if option == "Upload TXT File":
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        word_count = len(text.split())
        if word_count > 500:
            st.error(f"File contains {word_count} words. Please upload a file with 1000 or fewer words.")
        else:
            input_text = text

elif option == "Enter Text Manually":
    text = st.text_area("Enter text (≤1000 words):", height=300)
    if st.button("Summarize Text"):
        with st.spinner("Summarizing the text..."):
            word_count = len(text.split())
            if word_count == 0:
                st.warning("Please enter some text.")
            elif word_count > 500:
                st.error(f"Input has {word_count} words. Please reduce to 1000 or fewer.")
            else:
                input_text = text
                summary = summarize_text(input_text, summarizer, tokenizer)
                st.success("✅ Summary generated:")
                st.write(summary)
            

# Show summary
if input_text and option == "Upload TXT File":
    if st.button("Summarize File"):
        with st.spinner("Summarizing the text from file..."):
            summary = summarize_text(input_text, summarizer, tokenizer)
            st.success("✅ Summary generated:")
            st.write(summary) 