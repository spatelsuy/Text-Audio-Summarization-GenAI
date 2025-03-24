from transformers import pipeline

def summarize_text(transcript_file):
    summarizer = pipeline("summarization", model="t5-small")

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()

    summary = summarizer(transcript, max_length=72, min_length=50, do_sample=False)
    return summary[0]['summary_text']

summary = summarize_text("Meeting_Audio.txt")
print("Meeting Summary:\n", summary)

with open("Meeting_Minutes.txt", "w", encoding="utf-8") as f:
    f.write("**Meeting Minutes**\n\n")
    f.write(summary)
