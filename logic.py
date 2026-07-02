import os
import gc
import uuid
import shutil
import logging
import tempfile
import subprocess
import wave
from pathlib import Path
import numpy as np
import torch
import gradio as gr
from faster_whisper import WhisperModel, BatchedInferencePipeline
from sentence_transformers import SentenceTransformer, util
from config import Config

try:
    from moviepy import VideoFileClip, AudioFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
    except ImportError:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip

logger = logging.getLogger(__name__)

SENTENCE_ENDINGS = (
    '.', '!', '?', ';', #Latin/Cyrillic
    '。', '！', '？', #Chinese/Japanese/Korean
    '؟', '۔', #Arabic/Urdu
    '।', '॥', #Indic
    '։', '՜', '՞'  #Armenian
)

def safe_slice(clip, start_time, end_time):
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

def cleanup_session(session_path=None):
    if session_path and os.path.exists(session_path):
        try:
            shutil.rmtree(session_path)
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
    clear_memory()
    
def get_window_energy(wav_path, start_time, end_time):
    """Calculates RMS audio energy to identify enthusiastic peaks."""
    try:
        with wave.open(wav_path, 'rb') as wf:
            sr = wf.getframerate()
            start_frame = int(start_time * sr)
            end_frame = int(end_time * sr)
            wf.setpos(start_frame)
            frames = wf.readframes(end_frame - start_frame)
            audio_data = np.frombuffer(frames, dtype=np.int16)
            return np.sqrt(np.mean(audio_data.astype(np.float64)**2)) if len(audio_data) > 0 else 0.0
    except Exception:
        return 0.0

def _ends_on_sentence(text: str) -> bool:
    if not text: return False
    return text.strip().endswith(SENTENCE_ENDINGS)

def build_windows(processed_segs, min_dur, ideal_max, hard_max):
    windows = []
    n = len(processed_segs)
    i = 0
    while i < n:
        anchor_start = processed_segs[i]['start']
        candidates = []
        for j in range(i, n):
            current_dur = processed_segs[j]['end'] - anchor_start
            if min_dur <= current_dur <= hard_max:
                candidates.append({
                    'index': j,
                    'duration': current_dur,
                    'is_sent': _ends_on_sentence(processed_segs[j]['text']),
                    'gap': processed_segs[j+1]['start'] - processed_segs[j]['end'] if j+1 < n else 999.0
                })
        
        if not candidates: break
        
        sentence_ends = [c for c in candidates if c['is_sent']]
        best_cut = min(sentence_ends, key=lambda x: abs(x['duration'] - ideal_max)) if sentence_ends \
                   else max(candidates, key=lambda x: x['gap'])
        
        idx = best_cut['index']
        windows.append({
            'text': " ".join([f"[{processed_segs[k]['speaker']}] {processed_segs[k]['text']}" for k in range(i, idx+1)]),
            'start': anchor_start,
            'end': processed_segs[idx]['end'],
        })
        i = idx + 1
    return windows

def get_optimized_scores(windows, embedder, full_text, wav_path):
    if not windows: return []
    full_doc_embedding = embedder.encode(full_text, convert_to_tensor=True)
    window_texts = [w['text'] for w in windows]
    window_embeddings = embedder.encode(window_texts, convert_to_tensor=True)
    
    semantic_scores = [util.cos_sim(w_emb, full_doc_embedding).item() for w_emb in window_embeddings]
    energies = [get_window_energy(wav_path, w['start'], w['end']) for w in windows]
    max_energy = max(energies) if energies else 1.0
    normalized_energies = [e / max_energy for e in energies]

    return [(sem * 0.6) + (eng * 0.4) for sem, eng in zip(semantic_scores, normalized_energies)]

@torch.inference_mode()
def process_media(file_path, progress=gr.Progress()):
    session_id = str(uuid.uuid4())
    session_dir = Path(tempfile.gettempdir()) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    master_clip = None

    try:
        if is_video:
            progress(0.05, desc="Extracting audio for AI pipeline...")
            master_clip = VideoFileClip(file_path)
            master_clip.audio.write_audiofile(
                transcription_ready_audio, fps=16000, nbytes=2, codec="pcm_s16le",
                ffmpeg_params=["-ac", "1"], logger=None
            )
            master_clip.close()
            clear_memory()

        elif is_compressed_audio:
            progress(0.05, desc="Normalizing audio container for AI pipeline...")
            master_clip = AudioFileClip(file_path)
            master_clip.write_audiofile(
                transcription_ready_audio, fps=16000, nbytes=2, codec="pcm_s16le",
                ffmpeg_params=["-ac", "1"], logger=None
            )
            master_clip.close()
            clear_memory()

        else:
            transcription_ready_audio = file_path

        #Transcription
        progress(0.2, desc="Starting Transcription...")
        base_model = WhisperModel(Config.WHISPER_MODEL, device=Config.DEVICE, compute_type=Config.COMPUTE_TYPE)
        batched_model = BatchedInferencePipeline(base_model)
        segments_gen, info = batched_model.transcribe(
            transcription_ready_audio, vad_filter=True, batch_size=16
        )

        processed_segs = []
        speaker_turns = []
        total_duration = info.duration if info.duration else 1.0

        for s in segments_gen:
            speaker_label = f"Speaker {1 if len(speaker_turns) % 2 == 0 else 2}"
            speaker_turns.append({'start': s.start, 'end': s.end, 'speaker': speaker_label})
            processed_segs.append({
                'text': s.text.strip(),
                'start': s.start,
                'end': s.end,
                'speaker': speaker_label,
            })
            current_progress = min(0.2 + (s.end / total_duration * 0.54), 0.74)
            pct = int(current_progress * 100)
            progress(current_progress, desc=f"Transcribing ({pct}%): {int(s.end)}s / {int(total_duration)}s")

        del batched_model, base_model
        clear_memory()

        if not processed_segs:
            return "Error\nNo dialogue transcribed from the media source.", None, None, None, str(session_dir), str(session_dir)

        #Semantic Analysis 
        progress(0.75, desc="Analyzing content for promotional clips...")
        embedder = SentenceTransformer(Config.EMBEDDER_MODEL, device=Config.DEVICE)

        min_dur = getattr(Config, 'MIN_CLIP_DURATION', 30.0) 
        ideal_max = getattr(Config, 'MAX_CLIP_DURATION', 60.0)
        hard_max = 90.0                                       
        max_overlap_pct = 0.25 

        windows = build_windows(processed_segs, min_dur, ideal_max, hard_max)

        if not windows:
            logger.warning("No windows generated — falling back to raw segment boundaries.")
            windows = [{
                'text': f"[{s['speaker']}] {s['text']}",
                'start': s['start'],
                'end': s['end'],
            } for s in processed_segs]

        full_transcript = " ".join([s['text'] for s in processed_segs])

        scores = get_optimized_scores(windows, embedder, full_transcript, transcription_ready_audio)
        
        for i, w in enumerate(windows):
            w['score'] = scores[i]
            w['label'] = "High Energy Core" 

        ranked = sorted(
            [w for w in windows if w['score'] > 0.4], 
            key=lambda x: x['score'],
            reverse=True,
        )

        selected = []
        if ranked:
            for cand in ranked:
                if len(selected) >= 3:
                    break

                cand_dur = cand['end'] - cand['start']
                time_overlap = False

                for sel in selected:
                    sel_dur = sel['end'] - sel['start']
                    overlap_time = min(cand['end'], sel['end']) - max(cand['start'], sel['start'])

                    if overlap_time > 0:
                        cand_overlap_ratio = overlap_time / cand_dur
                        sel_overlap_ratio = overlap_time / sel_dur
                        if cand_overlap_ratio > max_overlap_pct or sel_overlap_ratio > max_overlap_pct:
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
                    if torch.max(sim_scores).item() < Config.SIMILARITY_THRESHOLD:
                        selected.append(cand)

        del embedder
        clear_memory()

        #Export Loops 
        progress(0.9, desc="Cutting and exporting viral clips...")
        clips = []

        if is_video:
            master_clip = VideoFileClip(file_path)
            for i, hook in enumerate(selected):
                path = session_dir / f"clip_{i+1}.mp4"
                
                padded_start = max(0.0, hook['start'] + 0.15)
                padded_end = min(hook['end'] + 0.4, master_clip.duration or total_duration)
                
                sub_clip = safe_slice(master_clip, padded_start, padded_end)
                
                sub_clip.write_videofile(
                    str(path), codec="libx264", audio_codec="aac",
                    audio_bitrate="320k",
                    temp_audiofile=str(session_dir / f"temp_audio_{i}.m4a"),
                    remove_temp=True, logger=None,
                )
                sub_clip.close()
                clips.append(str(path))
            master_clip.close()
            
        else:
            for i, hook in enumerate(selected):
                path = session_dir / f"clip_{i+1}.mp3"
                
                padded_start = max(0.0, hook['start'] - 0.15)
                padded_end = min(hook['end'] + 0.4, total_duration)
                export_duration = padded_end - padded_start
                
                fade_start = max(0.0, export_duration - 0.4)

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(padded_start),
                    "-t",  str(export_duration),
                    "-i",  str(file_path), 
                    "-filter_complex", f"afade=t=in:st=0:d=0.15,afade=t=out:st={fade_start}:d=0.4", 
                    "-acodec", "libmp3lame",
                    "-b:a", "320k",    
                    str(path),
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                clips.append(str(path))

        clip1 = clips[0] if len(clips) > 0 else None
        clip2 = clips[1] if len(clips) > 1 else None
        clip3 = clips[2] if len(clips) > 2 else None

        status_summary = (
            f"Processing Complete!\n"
            f"Successfully extracted {len(clips)} highly relevant viral clip(s)."
        )
        return status_summary, clip1, clip2, clip3, str(session_dir), str(session_dir)

    except Exception as e:
        logger.exception("Pipeline failed")
        return f"Error: {str(e)}", None, None, None, None, None
        
    finally:
        if master_clip:
            try: master_clip.close()
            except: pass
        clear_memory()
    