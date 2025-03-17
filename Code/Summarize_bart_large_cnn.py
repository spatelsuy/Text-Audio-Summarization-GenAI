from transformers import pipeline, AutoTokenizer

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
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    # Merge chunk summaries into a final summary
    final_summary = " ".join(summaries)
    
    return final_summary

# Run summarization
summary = summarize_text("Second.txt")
print("📌 **Meeting Summary** 📌\n", summary)

# Save summary to file
with open("Second_summary.txt", "w", encoding="utf-8") as f:
    f.write("📌 **Meeting Summary** 📌\n\n")
    f.write(summary)