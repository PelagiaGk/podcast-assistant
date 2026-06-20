import os
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TORCH_HOME"] = "/tmp/torch_cache"

import torch
import uuid
import logging
import gc
import shutil
import tempfile
import subprocess
from pathlib import Path
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, util
from config import Config
import gradio as gr
import numpy as np

try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        from moviepy.video.io.VideoFileClip import VideoFileClip

logger = logging.getLogger(__name__)

def safe_slice(clip, start_time, end_time):
    """Slices a MoviePy Video clip safely using explicit version compatibility checks."""
    if hasattr(clip, "subcut"):
        return clip.subcut(start_time, end_time)
    elif hasattr(clip, "subclip"):
        return clip.subclip(start_time, end_time)
    else:
        return clip

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_optimized_scores(windows, embedder):
    if not windows: return [], [], []
    
    anchor_embeddings = embedder.encode(Config.ANCHOR_THEMES, convert_to_tensor=True)
    window_texts = [w['text'] for w in windows]
    window_embeddings = embedder.encode(window_texts, convert_to_tensor=True)
    
    similarity_matrix = util.cos_sim(window_embeddings, anchor_embeddings)
    
    final_scores = []
    final_labels = []
    
    for i, window in enumerate(windows):
        row_scores = similarity_matrix[i]
        s_map = {label: row_scores[j].item() for j, label in enumerate(Config.ANCHOR_THEMES)}
        
        score = sum(s_map[label] * Config.WEIGHTS[label] for label in Config.ANCHOR_THEMES)
        
        best_theme_idx = torch.argmax(row_scores).item()
        best_label = Config.ANCHOR_THEMES[best_theme_idx]
        
        final_scores.append(score)
        final_labels.append(best_label)
        
    return final_scores, final_labels, window_embeddings

@torch.inference_mode()
def process_media(file_path, progress=gr.Progress()):
    if not file_path or not os.path.exists(file_path): 
        logger.error(f"Provided file path invalid or non-existent: {file_path}")
        return "Error\nFile selection missing or not found.", None, None, None, "", ""
    
    session_id = str(uuid.uuid4())
    #Save directly into a path that can be discarded, or let Gradio track it
    session_dir = Path(tempfile.gettempdir()) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    file_ext = file_path.lower()
    is_video = file_ext.endswith(('.mp4', '.mkv', '.mov', '.avi', '.webm'))
    is_compressed_audio = file_ext.endswith(('.m4a', '.aac', '.mp3', '.flac', '.ogg'))
    
    processing_path = str(session_dir / "standardized_input.wav")
    master_clip = None

    try:
        #native FFmpeg to extract audio
        if is_video or is_compressed_audio:
            progress(0.05, desc="Extracting/Normalizing audio via FFmpeg...")
            cmd = [
                "ffmpeg", "-y", "-i", str(file_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                processing_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            processing_path = file_path

        #Standard Whisper pipeline
        progress(0.2, desc="Starting Transcription...")
        base_model = WhisperModel(Config.WHISPER_MODEL, device=Config.DEVICE, compute_type=Config.COMPUTE_TYPE)
        
        segments_gen, info = base_model.transcribe(
            processing_path, 
            vad_filter=True, 
            vad_parameters=dict(min_silence_duration_ms=700)
        )
        
        processed_segs = []
        total_duration = info.duration if info.duration else 1.0

        for s in segments_gen:
            speaker_label = f"Speaker {1 if len(processed_segs) % 2 == 0 else 2}"
            processed_segs.append({
                'text': s.text.strip(), 
                'start': s.start, 
                'end': s.end, 
                'speaker': speaker_label
            })
            
            current_progress = min(0.2 + (s.end / total_duration * 0.54), 0.74) 
            progress(current_progress, desc=f"Transcribing: {int(s.end)}s / {int(total_duration)}s")

        del base_model
        clear_memory()

        if not processed_segs:
            return "Error\nNo dialogue transcribed.", None, None, None, str(session_dir), str(session_dir)

        #Semantic Window Search
        progress(0.75, desc="Analyzing content for viral clips...")
        embedder = SentenceTransformer(Config.EMBEDDER_MODEL, device=Config.DEVICE)
        
        windows = []
        for i in range(len(processed_segs)):
            curr_text, word_count = [], 0
            actual_start = float(processed_segs[i]['start'])
            
            for j in range(i, len(processed_segs)):
                seg = processed_segs[j]
                precise_end = float(seg['end'])
                
                if (precise_end - actual_start) > Config.MAX_CLIP_DURATION: 
                    break
                    
                curr_text.append(f"[{seg['speaker']}] {seg['text']}")
                word_count += len(seg['text'].split())
                
                if word_count >= Config.MIN_WORDS_FOR_HOOK and seg['text'].strip().endswith(('.', '!', '?')):
                    windows.append({
                        'text': " ".join(curr_text), 
                        'start': max(0.0, actual_start - 0.4), 
                        'end': precise_end + 0.6,
                        'original_index': len(windows) 
                    })
        
        scores, labels, window_embeddings = get_optimized_scores(windows, embedder)
        for i, w in enumerate(windows):
            w['score'], w['label'] = scores[i], labels[i]

        score_cutoff = getattr(Config, 'SCORE_THRESHOLD', 0.35)
        ranked = sorted([w for w in windows if w['score'] > score_cutoff], key=lambda x: x['score'], reverse=True)
        
        selected = []
        selected_indices = []
        
        for cand in ranked:
            if len(selected) >= 3: 
                break
                
            time_overlap = False
            for sel in selected:
                if max(cand['start'], sel['start']) < min(cand['end'], sel['end']):
                    time_overlap = True
                    break
            
            if time_overlap:
                continue

            if len(selected) == 0:
                selected.append(cand)
                selected_indices.append(cand['original_index'])
            else:
                #Re-use precomputed embeddings
                cand_emb = window_embeddings[cand['original_index']]
                sel_embs = torch.stack([window_embeddings[idx] for idx in selected_indices])
                
                sim_scores = util.cos_sim(cand_emb, sel_embs)
                similarity_limit = getattr(Config, 'SIMILARITY_THRESHOLD', 0.65)
                if torch.max(sim_scores).item() < similarity_limit:
                    selected.append(cand)
                    selected_indices.append(cand['original_index'])
                        
        del embedder
        clear_memory()

        #Export Loop
        progress(0.9, desc="Cutting and exporting viral clips...")
        clips = []
        
        if is_video:
            master_clip = VideoFileClip(str(file_path))
            for i, hook in enumerate(selected):
                path = session_dir / f"clip_{i+1}.mp4"
                sub_clip = safe_slice(master_clip, hook['start'], hook['end'])
                sub_clip.write_videofile(
                    str(path), codec="libx264", audio_codec="aac", 
                    logger=None, verbose=False
                )
                sub_clip.close()
                clips.append(str(path))
            master_clip.close()
        else:
            for i, hook in enumerate(selected):
                path = session_dir / f"clip_{i+1}.mp3"
                duration = hook['end'] - hook['start']
                
                cmd = [
                    "ffmpeg", "-y", "-ss", str(hook['start']), "-t", str(duration),
                    "-i", str(processing_path), "-acodec", "libmp3lame", "-b:a", "192k",
                    str(path)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                clips.append(str(path))

        clip1 = clips[0] if len(clips) > 0 else None
        clip2 = clips[1] if len(clips) > 1 else None
        clip3 = clips[2] if len(clips) > 2 else None

        status_summary = f"Processing Complete!\nSuccessfully extracted {len(clips)} clip(s)."
        return status_summary, clip1, clip2, clip3, str(session_dir), str(session_dir)

    except Exception as e:
        if master_clip: 
            try: master_clip.close()
            except: pass
        logger.exception("Critical unexpected error caught during processing pipeline execution:")
        return f"Pipeline Execution Failed\n**Reason:** {str(e)}", None, None, None, str(session_dir), str(session_dir)