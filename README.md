# Text-Audio-Summarization-GenAI
A Generative AI-powered meeting summarization tool that efficiently extracts key insights from audio and text. Built using state-of-the-art models from Hugging Face, this solution streamlines meeting notes, making them concise and actionable. While designed for meetings, it can summarize any content effectively.


## 📌 Text & Audio Summarization with Generative AI  

This repository provides Python scripts for summarizing text and audio using **Generative AI** models. It enables automatic summarization of long transcripts and audio recordings for meeting minutes and efficient information retrieval.  

### 📝 Features  
✅ Summarizes large text documents  
✅ Converts speech to text and extracts key insights  
✅ Uses state-of-the-art AI models (IBM Granite, BART, Whisper)  
✅ Supports handling large inputs via chunking  

### 🛠️ Installation  
The installation steps cover everything from setup to execution so that you can build your summarizer Gen AI tool on Windows and run it locally without relying on external APIs.

<ins>**Install Python 3.8+**</ins>  
During installation, check "Add Python to PATH".  
After installation check version of Python from command prompt. It will also ensure PATH setup  
```
python --version
```
<ins>**Install Git**</ins>  
[Install Git](https://git-scm.com/downloads)

<ins>**Install FFmpeg (May require for Whisper)**</ins>  
Download FFmpeg for Windows from this [link](https://ffmpeg.org/download.html)  
Extract the files and add the bin folder to your system's PATH.  

>**FFmpeg** is an open-source software library used for handling multimedia files and streams. It provides tools to convert audio and video formats to any other supported format.  
>**Whisper** is an open-source speech recognition system developed by OpenAI. It is designed to transcribe spoken language into text. It supports multiple languages and various audio formats. If you have an audio format that is not supported by Whisper, you should use FFmpeg to convert the audio to Whisper supported format.  

```
ffmpeg -version
```   

<ins>***Setup Python Virtual Environment***</ins>  
Why Do You Need a Virtual Environment?  
>A Python virtual environment is an isolated environment that allows you to install project-specific Python libraries and dependencies without affecting other Python projects.  

From command prompt run following. We are using name **meeting-ai-env** for the virtual environment
```
python -m venv meeting-ai-env
```

Activate the virtual environment (after activating you can can see the environment name at the beginning)  
```  
meeting-ai-env\Scripts\activate
```

<ins>***Install Required Python Packages for the project***</ins>  
Run this command inside the activated virtual environment to ensure it is for the project only  
```
pip install torch torchaudio transformers openai-whisper vosk librosa numpy pydub
```

>**Torch (PyTorch)** It is an open-source machine learning framework developed by Meta (Facebook). It provides deep learning capabilities for AI models. It is required by **Whisper** and Hugging Face **Transformers**.
Note: Torch and PyTorch are both deep learning frameworks. Tourch is written in C and Lua. It is no longer actively developed since 2019 and hence **PyTorch**. It is a reimplementation of Torch in Python.  

>**Torchaudio** is an audio processing library that extends PyTorch. It works well with Whisper and helps in loading, transforming, and analyzing audio data.  

>**Transformers (Hugging Face)** is a library for Natural Language Processing (NLP). It enables text summarization using pre-trained AI models like GPT (Generative Pre-trained Transformer), T5 (Text-to-Text Transfer Transformer), BERT (Bidirectional Encoder Representations from Transformers), and BART (Bidirectional and Auto-Regressive Transformers).  




