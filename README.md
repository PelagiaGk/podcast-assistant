---
title: Podcast Assistant
sdk: gradio
sdk_version: 5.0.1
python_version: "3.10"
models:
  - pyannote/segmentation-3.0
  - pyannote/speaker-diarization-3.1
  - distil-large-v3
  - valhalla/distilbart-mnli-12-1
  - all-MiniLM-L6-v2
app_file: app.py
pinned: false
---

# 🎙️ Podcast Assistant

An automated AI-powered assistant for podcast creators to effortlessly identify, extract, and analyze promotion-worthy clips from their full-length episodes. It can currently handle both video & audio only files.

This project is deployed and hosted live as a **Hugging Face Space**. It is structured to sync seamlessly from GitHub directly to Hugging Face via GitHub Actions.

---

## 🚀 Live Application

The application is fully hosted and ready to use without any local setup:
👉 **[Access the Podcast Assistant on Hugging Face Spaces](https://pelagia-gk-podcast-assistant.hf.space)** (https://pelagia-gk-podcast-assistant.hf.space)

---

## 🛠️ System Architecture & Models

The Podcast Assistant leverages specialized, state-of-the-art open-source machine learning models to process audio pipelines efficiently:

* **Audio Segmentation & Diarization:**
    * `pyannote/segmentation-3.0`: Handles voice activity detection and structural audio segmentation.
    * `pyannote/speaker-diarization-3.1`: Identifies "who spoke when," separating multiple hosts and guests within an episode.
* **Automatic Speech Recognition (ASR):**
    * `distil-large-v3`: A highly optimized, distilled version of OpenAI's Whisper model used for rapid and accurate speech-to-text transcription.
* **Zero-Shot Classification:**
    * `valhalla/distilbart-mnli-12-1`: Evaluates transcribed dialogue segments for specific promotional hooks, emotional spikes, or thematic highlights using Natural Language Inference (NLI).
* **Text Embeddings:**
    * `all-MiniLM-L6-v2`: Maps text segments into a vector space for semantic search and relevance ranking of extracted clips.

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
2. Create a feature branch (`git checkout -b feature/amazing-improvement`).
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
