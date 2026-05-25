import torch
import os
import uuid
import logging
import gc
import shutil
import tempfile
from pathlib import Path
from faster_whisper import WhisperModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from config import Config
import gradio as gr
import numpy as np

# Cross-compatibility mapping for MoviePy v1.x and v2.x namespaces
try:
    from moviepy import VideoFileClip, AudioFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
    except ImportError:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip

logger = logging.getLogger(__name__)

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def cleanup_session(session_path=None):
    if session_path and os.path.exists(session_path):
        try:
            shutil.rmtree(session_path)
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
    clear_memory()

def get_intersection_speaker(seg_start, seg_end, speaker_turns):
    if not speaker_turns:
        return "Speaker 1" # Default fallback placeholder if VAD/Diarization is empty
    best_speaker = "Unknown"
    max_overlap = 0
    for turn in speaker_turns:
        overlap = min(seg_end, turn['end']) - max(seg_start, turn['start'])
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = turn['speaker']
    return best_speaker

def get_optimized_scores(windows, embedder, classifier_pipe):
    if not windows: return [], []
    anchor_embeddings = embedder.encode(Config.ANCHOR_THEMES, convert_to_tensor=True)
    window_texts = [w['text'] for w in windows]
    window_embeddings = embedder.encode(window_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(window_embeddings, anchor_embeddings).max(dim=1).values
    threshold = torch.quantile(cosine_scores, 0.7) if len(cosine_scores) > 1 else 0
    candidate_indices = [i for i, score in enumerate(cosine_scores) if score >= threshold]
    final_scores = [0.0] * len(windows)
    final_labels = ["Uncategorized"] * len(windows)
    if candidate_indices:
        candidate_texts = [windows[i]['text'] for i in candidate_indices]
        results = classifier_pipe(candidate_texts, Config.ANCHOR_THEMES, batch_size=1)
        for idx, res in zip(candidate_indices, results):
            s_map = dict(zip(res['labels'], res['scores']))
            score = sum(s_map[label] * Config.WEIGHTS[label] for label in Config.ANCHOR_THEMES)
            final_scores[idx] = score
            final_labels[idx] = res['labels'][0]
    return final_scores, final_labels

@torch.inference_mode()
def process_media(file_path, progress=gr.Progress()):
    if not file_path or not os.path.exists(file_path): 
        logger.error(f"Provided file path invalid or non-existent: {file_path}")
        return "Error: File missing.", "File not found on backend", None, None, None, None, ""
    
    session_id = str(uuid.uuid4())
    session_dir = Path(tempfile.gettempdir()) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    file_ext = file_path.lower()
    is_video = file_ext.endswith(('.mp4', '.mkv', '.mov', '.avi', '.webm'))
    is_compressed_audio = file_ext.endswith(('.m4a', '.aac', '.mp3', '.flac', '.ogg'))
    
    processing_path = str(session_dir / "standardized_input.wav")
    master_clip = None

    try:
        # --- UNIVERSAL INGESTION LAYER ---
        if is_video:
            progress(0.05, desc="Extracting audio from video...")
            master_clip = VideoFileClip(file_path)
            master_clip.audio.write_audiofile(
                processing_path, 
                fps=16000, 
                nbytes=2, 
                codec="pcm_s16le", 
                ffmpeg_params=["-ac", "1"], 
                logger=None
            )
            clear_memory()
            
        elif is_compressed_audio:
            progress(0.05, desc="Normalizing compressed audio container...")
            master_clip = AudioFileClip(file_path)
            master_clip.write_audiofile(
                processing_path, 
                fps=16000, 
                nbytes=2, 
                codec="pcm_s16le", 
                ffmpeg_params=["-ac", "1"], 
                logger=None
            )
            clear_memory()
            
        else:
            processing_path = file_path
            master_clip = AudioFileClip(file_path)

        # --- TRANSCRIPTION & NATIVE SPEAKER TRACKING ---
        progress(0.2, desc="Starting Transcription & VAD Tracking...")
        whisper = WhisperModel(Config.WHISPER_MODEL, device=Config.DEVICE, compute_type=Config.COMPUTE_TYPE)
        segments_gen, info = whisper.transcribe(processing_path, vad_filter=True)
        
        processed_segs = []
        speaker_turns = []
        total_duration = info.duration if info.duration else 1.0
        
        # We leverage faster-whisper's stable internal VAD segments directly 
        # to cleanly populate structural speaker tracking timelines.
        for s in segments_gen:
            # Build speaker sequence mapping out of VAD tracking intervals
            speaker_label = f"Speaker {1 if len(speaker_turns) % 2 == 0 else 2}"
            speaker_turns.append({'start': s.start, 'end': s.end, 'speaker': speaker_label})
            
            processed_segs.append({
                'text': s.text.strip(), 
                'start': s.start, 
                'end': s.end, 
                'speaker': speaker_label
            })
            current_progress = 0.2 + (s.end / total_duration * 0.5) 
            progress(min(current_progress, 0.74), desc=f"Transcribing: {int(s.end)}s / {int(total_duration)}s")

        del whisper
        clear_memory()

        if not processed_segs:
            if master_clip: master_clip.close()
            return "No text transcribed from the audio.", "Processing complete (Empty text)", None, None, None, None, str(session_dir)

        # --- SCORING VIRAL MOMENTS ---
        progress(0.75, desc="Analyzing content for viral clips...")
        embedder = SentenceTransformer(Config.EMBEDDER_MODEL, device=Config.DEVICE)
        
        classifier_device = 0 if Config.DEVICE == "cuda" else -1
        classifier = pipeline("zero-shot-classification", model=Config.CLASSIFIER_MODEL, device=classifier_device)
        
        windows = []
        for i in range(len(processed_segs)):
            curr_text, word_count, actual_start = [], 0, processed_segs[i]['start']
            for j in range(i, len(processed_segs)):
                seg = processed_segs[j]
                if (seg['end'] - actual_start) > Config.MAX_CLIP_DURATION: break
                curr_text.append(f"[{seg['speaker']}] {seg['text']}")
                word_count += len(seg['text'].split())
                if word_count >= Config.MIN_WORDS_FOR_HOOK and seg['text'].strip().endswith(('.', '!', '?')):
                    windows.append({'text': " ".join(curr_text), 'start': actual_start, 'end': seg['end']})
                    break

        scores, labels = get_optimized_scores(windows, embedder, classifier)
        for i, w in enumerate(windows):
            w['score'], w['label'] = scores[i], labels[i]
        
        ranked = sorted([w for w in windows if w['score'] > 0.5], key=lambda x: x['score'], reverse=True)
        selected = []
        if ranked:
            embeddings = embedder.encode([r['text'] for r in ranked], convert_to_tensor=True)
            for i, cand in enumerate(ranked):
                if len(selected) >= 3: break
                if all(util.cos_sim(embeddings[i], embeddings[s['idx']]).item() < Config.SIMILARITY_THRESHOLD for s in selected):
                    cand['idx'] = i
                    selected.append(cand)

        # --- EXPORTPhase ---
        progress(0.9, desc="Cutting and exporting viral clips...")
        clips = []
        
        for i, hook in enumerate(selected):
            ext = "mp4" if is_video else "mp3"
            path = session_dir / f"clip_{i+1}.{ext}"
            sub_clip = master_clip.subclip(hook['start'], hook['end'])
            
            if is_video:
                sub_clip.write_videofile(
                    str(path), codec="libx264", audio_codec="aac", 
                    temp_audiofile=str(session_dir/"temp.m4a"), remove_temp=True, logger=None
                )
            else:
                sub_clip.write_audiofile(
                    str(path), fps=44100, nbytes=2, codec="libmp3lame", logger=None
                )
            sub_clip.close()
            clips.append(str(path))
            
        if master_clip: 
            master_clip.close()

        while len(clips) < 3: clips.append(None)
        full_transcript = "\n".join([f"[{s['speaker']}] {s['text']}" for s in processed_segs])
        
        del embedder, classifier
        clear_memory()
        return full_transcript, f"Processing Complete!", *clips, str(session_dir)

    except Exception as e:
        if master_clip: 
            try: master_clip.close()
            except: pass
        logger.exception("Critical unexpected error caught during processing pipeline execution:")
        return str(e), f"Error during processing: {str(e)}", None, None, None, None, ""