import os
import gc
import re
import time
import uuid
import shutil
import logging
import tempfile
import subprocess
import torch
import gradio as gr
from pathlib import Path
from faster_whisper import WhisperModel, BatchedInferencePipeline
from sentence_transformers import SentenceTransformer, util
from config import Config
import soundfile as sf

try:
    from moviepy import VideoFileClip, AudioFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
    except ImportError:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip

_vad_model = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Multilingual punctuation support
SENTENCE_ENDINGS = (
    '.', '!', '?', ';', #Latin/Cyrillic
    '。', '！', '？', #Chinese/Japanese/Korean
    '؟', '۔', #Arabic/Urdu
    '।', '॥', #Indic
    '։', '՜', '՞' #Armenian
)

def clear_memory():
    """Aggressively flushes RAM and VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def cleanup_session(session_path):
    """Safely removes session folder and cleans memory."""
    if session_path and os.path.exists(session_path):
        try:
            shutil.rmtree(session_path)
        except Exception as e:
            logging.error(f"Cleanup failed: {e}")
    gc.collect() 
    return None, None, None, "", ""

def cleanup_stale_sessions(max_age_hours=1):
    """Prevents storage exhaustion in public deployments by purging old sessions."""
    temp_dir = Path(tempfile.gettempdir())
    current_time = time.time()
    
    for session_dir in temp_dir.glob("session_*"):
        if session_dir.is_dir():
            dir_age = current_time - session_dir.stat().st_mtime
            if dir_age > (max_age_hours * 3600):
                try:
                    shutil.rmtree(session_dir)
                    logger.info(f"Purged stale session: {session_dir}")
                except Exception as e:
                    logger.error(f"Error deleting session {session_dir}: {e}")

def safe_slice(clip, start_time, end_time):
    """Slices a MoviePy clip safely using explicit version compatibility checks."""
    if hasattr(clip, "subcut"):
        return clip.subcut(start_time, end_time)
    elif hasattr(clip, "subclip"):
        return clip.subclip(start_time, end_time)
    else:
        return clip

def _ends_on_sentence(text: str) -> bool:
    """Checks if text ends on a boundary, ignoring trailing whitespace or quotes."""
    cleaned = re.sub(r'[\'"\s]+$', '', text)
    return cleaned.endswith(SENTENCE_ENDINGS)

def get_optimized_scores(windows, embedder, full_text):
    """
    Scores segments based on semantic relevance and speech density.
    This ignores raw audio volume, preventing music from skewing results.
    """
    if not windows: return []
    
    full_doc_embedding = embedder.encode(full_text, convert_to_tensor=True)
    window_texts = [w['text'] for w in windows]
    window_embeddings = embedder.encode(window_texts, convert_to_tensor=True)
    
    semantic_scores = [util.cos_sim(w_emb, full_doc_embedding).item() for w_emb in window_embeddings]
    
    densities = [len(w['text']) / max(1.0, w['end'] - w['start']) for w in windows]
    max_density = max(densities) if densities else 1.0
    normalized_densities = [d / max_density for d in densities]

    return [(sem * 0.7) + (den * 0.3) for sem, den in zip(semantic_scores, normalized_densities)]

def build_windows(processed_segs, min_dur, ideal_max, hard_max):
    windows = []
    n = len(processed_segs)
    for i in range(n):
        anchor_start = processed_segs[i]['start']
        candidates = []
        for j in range(i, n):
            current_dur = processed_segs[j]['end'] - anchor_start
            if current_dur > hard_max: break
            if current_dur >= min_dur:
                candidates.append({'index': j, 'duration': current_dur})
        
        if not candidates: continue
        best_cut = min(candidates, key=lambda x: abs(x['duration'] - ideal_max))
        idx = best_cut['index']
        windows.append({
            'text': " ".join([processed_segs[k]['text'] for k in range(i, idx+1)]),
            'start': anchor_start,
            'end': processed_segs[idx]['end'],
        })
    return windows

def get_speech_timestamps_from_file(wav_path):
    global _vad_model
    if _vad_model is None:
        _vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                                           model='silero_vad', 
                                           force_reload=False)
    (get_speech_timestamps, _, read_audio, _, _) = utils
    wav = read_audio(str(wav_path))
    return get_speech_timestamps(wav, _vad_model, sampling_rate=16000, threshold=0.5)

@torch.inference_mode()
def process_media(file_path, progress=gr.Progress()):
    if not file_path or not os.path.exists(file_path):
        return "Error\nFile not found.", None, None, None, "", ""

    cleanup_stale_sessions()

    session_id = str(uuid.uuid4())
    session_dir = Path(tempfile.gettempdir()) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    transcription_ready_audio = str(session_dir / "transcribe_low_res.wav")
    
    is_video = file_path.lower().endswith(('.mp4', '.mkv', '.mov', '.avi', '.webm'))
    try:
        if is_video:
            master_clip = VideoFileClip(file_path)
            master_clip.audio.write_audiofile(transcription_ready_audio, fps=16000, nbytes=2, codec="pcm_s16le", ffmpeg_params=["-ac", "1"], logger=None)
            master_clip.close()
            master_clip = None 
        else:
            transcription_ready_audio = file_path
    except Exception as e:
        return f"Extraction Error: {str(e)}", None, None, None, "", ""

    if not os.path.exists(transcription_ready_audio):
        return "Error: Audio file creation failed.", None, None, None, "", ""

    speech_intervals = get_speech_timestamps_from_file(transcription_ready_audio)

    #Transcription
    base_model = WhisperModel(Config.WHISPER_MODEL, device=Config.DEVICE, compute_type=Config.COMPUTE_TYPE)
    batched_model = BatchedInferencePipeline(base_model)

    segments_gen, info = batched_model.transcribe(
        transcription_ready_audio, 
        vad_filter=True, 
        batch_size=16,
        condition_on_previous_text=False 
    )

    processed_segs = []
    for s in segments_gen:
        is_valid_speech = any(s.start < interval['end'] and s.end > interval['start'] for interval in speech_intervals)
        
        if not is_valid_speech or s.no_speech_prob > 0.35 or s.avg_logprob < -1.0:
            continue
            
        clean_text = re.sub(r'\[.*?\]|\(.*?\)|\♪', '', s.text).strip()
        if not clean_text or clean_text.lower() in ["thank you", "bye", "subscribe"]:
            continue
            
        processed_segs.append({'text': clean_text, 'start': s.start, 'end': s.end, 'speaker': "Speaker"})

    #Analysis
    embedder = SentenceTransformer(Config.EMBEDDER_MODEL, device=Config.DEVICE)
    windows = build_windows(processed_segs, getattr(Config, 'MIN_CLIP_DURATION', 30.0), getattr(Config, 'MAX_CLIP_DURATION', 60.0), 90.0)
    
    full_transcript = " ".join([s['text'] for s in processed_segs])
    scores = get_optimized_scores(windows, embedder, full_transcript)
    for i, w in enumerate(windows): w['score'] = scores[i]

    ranked = sorted([w for w in windows if w['score'] > 0.4], key=lambda x: x['score'], reverse=True)
    selected, max_overlap_pct = [], 0.25
    
    for cand in ranked:
        if len(selected) >= 3: break
        overlap = any((min(cand['end'], sel['end']) - max(cand['start'], sel['start'])) > 0 for sel in selected)
        if not overlap: selected.append(cand)

    selected = sorted(selected, key=lambda x: x['start'])

    #Export
    clips = []
    try:
        if is_video:
            master_clip = VideoFileClip(file_path)
            for i, hook in enumerate(selected):
                path = session_dir / f"clip_{i+1}.mp4"
                sub_clip = safe_slice(master_clip, max(0.0, hook['start'] - 0.15), min(hook['end'] + 0.1, master_clip.duration))
                sub_clip.write_videofile(
                    str(path), codec="libx264", audio_codec="aac", audio_bitrate="320k", audio_fps=48000, preset="fast", logger=None
                )
                sub_clip.close()
                clips.append(str(path))
            master_clip.close()
        else:
            for i, hook in enumerate(selected):
                path = session_dir / f"clip_{i+1}.mp3"
                start, end = max(0.0, hook['start'] - 0.15), min(hook['end'] + 0.1, info.duration)
                fade_start = max(0.0, (end - start) - 0.1)
                cmd = ["ffmpeg", "-y", "-ss", str(start), "-t", str(end-start), "-i", str(file_path), 
                       "-filter_complex", f"afade=t=in:st=0:d=0.15,afade=t=out:st={fade_start}:d=0.1", 
                       "-acodec", "libmp3lame", "-b:a", "320k", "-ar", "48000", str(path)]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                clips.append(str(path))
    except Exception as e:
        return f"Export Error: {str(e)}", None, None, None, str(session_dir), str(session_dir)

    return "Processing Complete!", *[clips[i] if i < len(clips) else None for i in range(3)], str(session_dir), str(session_dir)      

except Exception as e:
    logger.exception("Pipeline failed")
    return f"Error: {str(e)}", None, None, None, str(session_dir), str(session_dir)
finally:
    if master_clip: master_clip.close()
    clear_memory()