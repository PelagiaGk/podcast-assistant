---
title: Podcast Assistant
sdk: static
sdk_version: 5.0.1
python_version: "3.10"
models:
  - snakers4/silero-vad
  - openai/whisper-base
  - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  - pyannote/speaker-diarization-2.1
app_file: app.py
pinned: false
---

# 🎙️ Podcast Assistant

An automated AI-powered assistant for podcast creators to effortlessly identify, extract, and analyze promotion-worthy clips from their full-length episodes. It can currently handle both video & audio-only files.
This project is deployed and hosted live as a **Hugging Face Space**. It is structured to sync seamlessly from GitHub directly to Hugging Face via GitHub Actions.

---

## 🚀 Live Application

The application is fully hosted and ready to use without any local setup:
👉 **[Access the Podcast Assistant on Hugging Face Spaces](https://pelagia-gk-podcast-assistant.hf.space)** (https://pelagia-gk-podcast-assistant.hf.space)

---

## 🛠 Features

Smart Clipping: Automatically cuts at natural sentence boundaries for a professional, finished feel.
Semantic Prioritization: Uses AI to select clips that best represent the overall topic of your file.
Quality Filtering: Automatically discards clips with low transcription confidence or excessive repetitive filler.
Polished Transitions: Automatically applies short audio fade-ins and fade-outs to prevent jarring cuts.

---

## 🏗 Pipeline Architecture

This app runs a multi-stage inference pipeline:
VAD: Silero VAD filters the audio to keep only meaningful speech.
Transcription: faster-whisper performs high-speed, accurate transcription.
Analysis: A SentenceTransformer model embeds the transcript to calculate thematic relevance.
Scoring & Export: Candidates are scored and sliced using optimized ffmpeg and moviepy commands.

---

## 🔒 Privacy & Data Policy

Temporary Storage: Your files are processed in a secure, temporary environment.
Automated Cleanup: To ensure the space remains performant and to respect disk limitations, all session data (including uploaded files and generated clips) is automatically purged.
Non-Persistent: No data is stored permanently. Once your session is cleared, all related files are permanently deleted from the host.

---

## 👥 Contributing & Reporting Issues

This repository contains the public source code for the Podcast Assistant. While the project is intended exclusively for deployment on Hugging Face Spaces rather than local machine execution, **contributions, improvements, and bug reporting are highly encouraged!**

### Found a Bug or Have a Feature Request?
Since the application runs in a production cloud environment, environment discrepancies should be minimal. However, if you notice unexpected behavior, processing errors, or have UI enhancement ideas:
1. Navigate to the **Issues** tab in this GitHub repository.
2. Check if a similar issue has already been reported.
3. Open a new issue with a clear description of the behavior, steps to reproduce it, and any error logs visible from the Hugging Face console.

### Pull Requests
If you want to contribute fixes, optimize the processing pipeline, or update the Gradio interface:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/improvement`).
3. Commit your changes with descriptive messages.
4. Push your branch and open a **Pull Request** against the main branch.

*Note: Merged changes will automatically sync and deploy to the live Hugging Face Space.*

---

## 🔒 Security & Environment Variables

If you are maintaining a fork or setting up your own sync mirror, ensure the following environment secrets are securely configured within your GitHub Actions and Hugging Face Space settings:
* `HF_TOKEN`: Your Hugging Face User Access Token (with Write permissions) to allow GitHub Actions to push changes.
* Any necessary API keys or user tokens required by the `pyannote` models (as they require accepting user terms on Hugging Face before first use).

---

## 📝 License

This project is open-source. Please check the `LICENSE` file in the root directory for specific terms.
