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
from faster_whisper import WhisperModel, BatchedInferencePipeline
from sentence_transformers import SentenceTransformer, util
from config import Config
import gradio as gr
import numpy as np

try:
    from moviepy import VideoFileClip, AudioFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
    except ImportError:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip

logger = logging.getLogger(__name__)

def safe_slice(clip, start_time, end_time):
    """Slices a MoviePy Video clip safely using explicit version compatibility checks."""
    if hasattr(clip, "subcut"):
        return clip.subcut(start_time, end_time)  #Modern MoviePy v2.x
    elif hasattr(clip, "subclip"):
        return clip.subclip(start_time, end_time)  #Legacy MoviePy v1.x
    else:
        return clip

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
        return "Speaker 1"
    best_speaker = "Unknown"
    max_overlap = 0
    for turn in speaker_turns:
        overlap = min(seg_end, turn['end']) - max(seg_start, turn['start'])
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = turn['speaker']
    return best_speaker

def get_optimized_scores(windows, embedder):
    if not windows: return [], []
    
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
        
    return final_scores, final_labels

@torch.inference_mode()
def process_media(file_path, progress=gr.Progress()):
    if not file_path or not os.path.exists(file_path): 
        logger.error(f"Provided file path invalid or non-existent: {file_path}")
        return "Error\nFile selection missing or not found on the backend host.", None, None, None, "", ""
    
    session_id = str(uuid.uuid4())
    session_dir = Path(tempfile.gettempdir()) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    file_ext = file_path.lower()
    is_video = file_ext.endswith(('.mp4', '.mkv', '.mov', '.avi', '.webm'))
    is_compressed_audio = file_ext.endswith(('.m4a', '.aac', '.mp3', '.flac', '.ogg'))
    
    processing_path = str(session_dir / "standardized_input.wav")
    master_clip = None

    try:
        if is_video:
            progress(0.05, desc="Extracting audio from video...")
            master_clip = VideoFileClip(file_path)
            master_clip.audio.write_audiofile(
                processing_path, fps=16000, nbytes=2, codec="pcm_s16le", ffmpeg_params=["-ac", "1"], logger=None
            )
            master_clip.close()
            clear_memory()
            
        elif is_compressed_audio:
            progress(0.05, desc="Normalizing compressed audio container...")
            master_clip = AudioFileClip(file_path)
            master_clip.write_audiofile(
                processing_path, fps=16000, nbytes=2, codec="pcm_s16le", ffmpeg_params=["-ac", "1"], logger=None
            )
            master_clip.close()
            clear_memory()
            
        else:
            processing_path = file_path

        #Batched Whisper Pipeline on CPU/GPU with Word Timestamps
        progress(0.2, desc="Starting Transcription...")
        base_model = WhisperModel(Config.WHISPER_MODEL, device=Config.DEVICE, compute_type=Config.COMPUTE_TYPE)
        batched_model = BatchedInferencePipeline(base_model)
        
        segments_gen, info = batched_model.transcribe(processing_path, vad_filter=True, batch_size=16, word_timestamps=True)
        
        processed_segs = []
        speaker_turns = []
        total_duration = info.duration if info.duration else 1.0

        for s in segments_gen:
            speaker_label = f"Speaker {1 if len(speaker_turns) % 2 == 0 else 2}"
            speaker_turns.append({'start': s.start, 'end': s.end, 'speaker': speaker_label})
            
            #Keep track of individual words for precise cutting borders
            words_list = []
            if hasattr(s, 'words') and s.words:
                words_list = [{'word': w.word, 'start': w.start, 'end': w.end} for w in s.words]
            
            processed_segs.append({
                'text': s.text.strip(), 
                'start': s.start, 
                'end': s.end, 
                'speaker': speaker_label,
                'words': words_list  #Store the granular word data
            })
            
            current_progress = min(0.2 + (s.end / total_duration * 0.54), 0.74) 
            pct = int(current_progress * 100)
            progress(current_progress, desc=f"Transcribing ({pct}%): {int(s.end)}s / {int(total_duration)}s")

        del batched_model, base_model
        clear_memory()

        if not processed_segs:
            return "Error\nNo dialogue transcribed from the media source.", None, None, None, str(session_dir), str(session_dir)

        #Semantic Window Search
        progress(0.75, desc="Analyzing content for viral clips...")
        embedder = SentenceTransformer(Config.EMBEDDER_MODEL, device=Config.DEVICE)
        
        windows = []
        for i in range(len(processed_segs)):
            #Establish absolute initial fallback start
            fallback_start = processed_segs[i]['start']
            #If word data exists, anchor perfectly to the first word spoken in this phrase block
            if processed_segs[i]['words']:
                actual_start = processed_segs[i]['words'][0]['start']
            else:
                actual_start = fallback_start

            curr_text, word_count = [], 0
            for j in range(i, len(processed_segs)):
                seg = processed_segs[j]
                
                #Use the segment's precise final word end if available
                precise_end = seg['words'][-1]['end'] if seg['words'] else seg['end']
                
                if (precise_end - actual_start) > Config.MAX_CLIP_DURATION: 
                    break
                    
                curr_text.append(f"[{seg['speaker']}] {seg['text']}")
                word_count += len(seg['text'].split())
                
                if word_count >= Config.MIN_WORDS_FOR_HOOK and seg['text'].strip().endswith(('.', '!', '?')):
                    windows.append({
                        'text': " ".join(curr_text), 
                        'start': max(0, actual_start - 0.2), 
                        'end': precise_end + 0.1             
                    })

        scores, labels = get_optimized_scores(windows, embedder)
        for i, w in enumerate(windows):
            w['score'], w['label'] = scores[i], labels[i]

        #Filter and rank items with a valid score
        score_cutoff = getattr(Config, 'SCORE_THRESHOLD', 0.35)
        ranked = sorted([w for w in windows if w['score'] > score_cutoff], key=lambda x: x['score'], reverse=True)
        
        selected = []
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
            else:
                cand_emb = embedder.encode(cand['text'], convert_to_tensor=True)
                sel_embs = embedder.encode([s['text'] for s in selected], convert_to_tensor=True)
                
                sim_scores = util.cos_sim(cand_emb, sel_embs)
                similarity_limit = getattr(Config, 'SIMILARITY_THRESHOLD', 0.65)
                if torch.max(sim_scores).item() < similarity_limit:
                    selected.append(cand)
                        
        del embedder
        clear_memory()

        #Export Loop
        progress(0.9, desc="Cutting and exporting viral clips...")
        clips = []
        
        if is_video:
            master_clip = VideoFileClip(file_path)
            for i, hook in enumerate(selected):
                path = session_dir / f"clip_{i+1}.mp4"
                sub_clip = safe_slice(master_clip, hook['start'], hook['end'])
                sub_clip.write_videofile(
                    str(path), codec="libx264", audio_codec="aac", 
                    temp_audiofile=str(session_dir / "temp.m4a"), remove_temp=True, logger=None
                )
                sub_clip.close()
                clips.append(str(path))
            master_clip.close()
        else:
            for i, hook in enumerate(selected):
                path = session_dir / f"clip_{i+1}.mp3"
                duration = hook['end'] - hook['start']
                
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(hook['start']),
                    "-t", str(duration),
                    "-i", str(processing_path),
                    "-acodec", "libmp3lame",
                    "-b:a", "192k",
                    str(path)
                ]
                
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                clips.append(str(path))

        clip1 = clips[0] if len(clips) > 0 else None
        clip2 = clips[1] if len(clips) > 1 else None
        clip3 = clips[2] if len(clips) > 2 else None

        status_summary = f"Processing Complete!\nSuccessfully extracted {len(clips)} highly relevant viral clip(s)."
        return status_summary, clip1, clip2, clip3, str(session_dir), str(session_dir)

    except Exception as e:
        if master_clip: 
            try: master_clip.close()
            except: pass
        logger.exception("Critical unexpected error caught during processing pipeline execution:")
        error_md = f"Pipeline Execution Failed\n**Reason:** {str(e)}"
        return error_md, None, None, None, str(session_dir), str(session_dir)