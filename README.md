# 🎙️ Podcast Assistant - A free tool for content creators that supports multiple languages

A free automated AI-powered assistant for podcast creators to effortlessly identify, extract, and analyze promotion-worthy clips from their full-length episodes. It can currently handle both video and audio-only files, and multiple languages.
### Link: https://podcast-assistant.made-in.app/

---

## Models

  - snakers4/silero-vad
  - openai/whisper-medium
  - sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  - pyannote/speaker-diarization-2.1

---

## 🛠 Features

- Smart Clipping: Cuts at natural sentence boundaries for a professional, finished feel.
- Semantic Prioritization: Uses AI to select clips that best represent the overall topic of your file.
- Quality Filtering: Automatically discards clips with low transcription confidence or excessive repetitive filler.
- Polished Transitions: Automatically applies short audio fade-ins and fade-outs to prevent jarring cuts.

---

## 🏗 Pipeline Architecture

This app runs a multi-stage inference pipeline:
- VAD: Silero VAD filters the audio to keep only meaningful speech.
- Transcription: faster-whisper performs high-speed, accurate transcription.
- Analysis: A SentenceTransformer model embeds the transcript to calculate thematic relevance.
- Scoring & Export: Candidates are scored and sliced using optimized ffmpeg and moviepy commands.

---

## 🔒 Privacy & Data Policy

- Temporary Storage: Your files are processed in a secure, temporary environment.
- Automated Cleanup: To ensure the space remains performant and to respect disk limitations, all session data (including uploaded files and generated clips) is automatically purged.
- Non-Persistent: No data is stored permanently. Once your session is cleared, all related files are permanently deleted from the host.

---

## 👥 Reporting Issues

This repository contains the public source code for the Podcast Assistant. While the project is intended exclusively for deployment rather than local machine execution, **contributions, improvements, and bug reporting are highly encouraged!**

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

*Note: Merged changes will automatically sync and deploy.*

---

## 🔒 Security & Environment Variables

* If you are maintaining a fork or setting up your own sync mirror, ensure the following environment secrets are securely configured within your GitHub Actions and Hugging Face Space settings:
  1. `HF_TOKEN`: Your Hugging Face User Access Token (with Write permissions) to allow GitHub Actions to push changes.
  2. Any necessary API keys or user tokens required by the `pyannote` models (as they require accepting user terms on Hugging Face before first use). 
  --> Visit the pyannote/speaker-diarization-2.1 model page and click "Agree and access repository", as well as the pyannote/segmentation model page.
* Hardware Requirements: This project needs at least 2 vCPUs and 4GB of RAM to run effectively.
* To run locally you need to run these in your terminal:
   ### Option A) Docker (using the terminal of your choice)
    1. To connect to your Docker, run on your terminal:
       docker build or docker run
    2. Build the local image:
       docker build -t podcast-assistant-local .
    3. Run the container, linking the local .env file:
       docker run -p 7860:7860 --env-file .env podcast-assistant-local
  ### Option B) Open the terminal of your choice (VS Code, Git bash, GitHub Codespaces etc)
    1. Create a virtual environment:
       python -m venv venv
       source venv/bin/activate
    2. Install the packages directly:
       pip install -r requirements.txt
    3. Run the app:
       python app.py

---

## 📝 License

This project is open-source. Please check the `LICENSE` file in the root directory for specific terms.
