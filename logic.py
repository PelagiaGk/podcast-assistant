import os
import gc
import re
import time
import uuid
import shutil
import logging
import tempfile
import threading
import subprocess
import torch
import gradio as gr
from pathlib import Path
from faster_whisper import WhisperModel, BatchedInferencePipeline
from sentence_transformers import SentenceTransformer, util
from config import Config
import random

try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        from moviepy.video.io.VideoFileClip import VideoFileClip

_vad_model = None
_vad_utils = None

_whisper_pipeline = None
_embedder_model = None

_inference_lock = threading.Lock()

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

def load_vad_model():
    """Loads the Silero VAD model and its utility functions."""
    global _vad_model, _vad_utils
    if _vad_model is not None and _vad_utils is not None:
        return

    from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
    use_onnx = getattr(Config, 'VAD_USE_ONNX', False)
    _vad_model = load_silero_vad(onnx=use_onnx)
    _vad_utils = (get_speech_timestamps, None, read_audio, None, None)

def get_whisper_pipeline():
    """Loads and caches the faster-whisper model. Loaded once per
    process instead of once per request."""
    global _whisper_pipeline
    if _whisper_pipeline is None:
        base_model = WhisperModel(Config.WHISPER_MODEL, device=Config.DEVICE, compute_type=Config.COMPUTE_TYPE)
        if getattr(Config, 'USE_BATCHED_INFERENCE', True):
            _whisper_pipeline = BatchedInferencePipeline(base_model)
        else:
            _whisper_pipeline = base_model
    return _whisper_pipeline

def get_embedder_model():
    """Loads and caches the sentence embedder."""
    global _embedder_model
    if _embedder_model is None:
        _embedder_model = SentenceTransformer(Config.EMBEDDER_MODEL, device=Config.DEVICE)
    return _embedder_model

def warm_up_models():
    """Loads VAD, Whisper, and the embedder once, up front."""
    with _inference_lock:
        load_vad_model()
        get_whisper_pipeline()
        get_embedder_model()
    logger.info("Models warmed up and ready.")

def clear_memory():
    """Flushes RAM and VRAM (of transient tensors — the cached
    models themselves are intentionally kept alive)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def cleanup_session(session_path):
    """Removes session folder and cleans memory."""
    if session_path and os.path.exists(session_path):
        try:
            shutil.rmtree(session_path)
        except Exception as e:
            logging.error(f"Cleanup failed: {e}")
    gc.collect()
    return None, None, None, "", ""

def cleanup_stale_sessions(max_age_hours=1):
    """Background execution."""
    try:
        temp_dir = Path(tempfile.gettempdir())
        current_time = time.time()
        
        for session_dir in temp_dir.glob("session_*"):
            if session_dir.is_dir():
                dir_age = current_time - session_dir.stat().st_mtime
                if dir_age > (max_age_hours * 3600):
                    try:
                        shutil.rmtree(session_dir)
                    except Exception as e:
                        logging.debug(f"Could not delete {session_dir}: {e}")
    except Exception as e:
        logging.error(f"Cleanup thread error: {e}")

def safe_slice(clip, start_time, end_time):
    """Slices a MoviePy clip using explicit version compatibility checks."""
    if hasattr(clip, "subcut"):
        return clip.subcut(start_time, end_time)
    elif hasattr(clip, "subclip"):
        return clip.subclip(start_time, end_time)
    else:
        return clip

def apply_audio_fade(clip, fade_in=0.15, fade_out=0.2):
    """Short audio fade in/out so exported clips don't start or end
    on a jarring hard cut. Handles both MoviePy v1 and v2 APIs."""
    try:
        from moviepy import afx
        return clip.with_effects([afx.AudioFadeIn(fade_in), afx.AudioFadeOut(fade_out)])
    except ImportError:
        try:
            return clip.audio_fadein(fade_in).audio_fadeout(fade_out)
        except AttributeError:
            return clip

def _ends_on_sentence(text: str) -> bool:
    """Checks if text ends on a sentence boundary, ignoring trailing whitespace or quotes."""
    cleaned = re.sub(r'[\'"\s]+$', '', text)
    return cleaned.endswith(SENTENCE_ENDINGS)

def _speech_overlap_ratio(seg_start, seg_end, speech_intervals):
    """Fraction of a Whisper segment's duration that Silero VAD independently
    confirmed as speech."""
    seg_dur = max(1e-6, seg_end - seg_start)
    covered = 0.0
    for interval in speech_intervals:
        covered += max(0.0, min(seg_end, interval['end']) - max(seg_start, interval['start']))
    return covered / seg_dur

_REPEAT_RE = re.compile(r'\b(\w+(?:\s+\w+){0,2}?)\b(?:\s+\1\b){2,}', re.IGNORECASE)

def _looks_repetitive(text: str) -> bool:
    """Flags likely hallucinated/song-lyric text: the same word or short
    phrase repeated back-to-back three or more times."""
    return bool(_REPEAT_RE.search(text))

def _starts_naturally(processed_segs, i, min_gap=0.5):
    """A clip may only begin at segment i if it's a natural entry point:
    the very first segment overall, right after a sentence boundary, or
    preceded by a real pause(topic change /breath break)."""
    if i == 0:
        return True
    prev = processed_segs[i - 1]
    if _ends_on_sentence(prev['text']):
        return True
    return (processed_segs[i]['start'] - prev['end']) >= min_gap

def _window_confidence(processed_segs, i, idx):
    """Transcription-confidence for the segments spanning a window,
    from signals Whisper: avg_logprob near 0 and a low
    no_speech_prob."""
    segs = processed_segs[i:idx + 1]
    if not segs:
        return 0.0
    logprob_component = sum(max(0.0, min(1.0, s['avg_logprob'] + 1.0)) for s in segs) / len(segs)
    no_speech_component = sum(1.0 - s['no_speech_prob'] for s in segs) / len(segs)
    return (logprob_component + no_speech_component) / 2.0

def _overlap_ratio(a, b):
    """Fraction of the shorter clip's duration that overlaps with the other clip.
    0.0 = no overlap, 1.0 = one fully contains the other."""
    intersection = min(a['end'], b['end']) - max(a['start'], b['start'])
    if intersection <= 0:
        return 0.0
    shorter = min(a['end'] - a['start'], b['end'] - b['start'])
    if shorter <= 0:
        return 1.0
    return intersection / shorter

def get_optimized_scores(windows, embedder, full_text):
    """
    Scores windows on three factors: semantic relevance to the full
    transcript, speech density (ignoring raw audio volume), and transcription confidence.
    """
    if not windows:
        return []

    full_doc_embedding = embedder.encode(full_text, convert_to_tensor=True)
    window_texts = [w['text'] for w in windows]
    window_embeddings = embedder.encode(window_texts, convert_to_tensor=True)

    semantic_scores = [util.cos_sim(w_emb, full_doc_embedding).item() for w_emb in window_embeddings]

    densities = [len(w['text']) / max(1.0, w['end'] - w['start']) for w in windows]
    max_density = max(densities) if densities else 1.0
    max_density = max_density if max_density > 0 else 1.0
    normalized_densities = [d / max_density for d in densities]

    confidences = [w.get('confidence', 1.0) for w in windows]

    sem_w = getattr(Config, 'SCORE_SEMANTIC_WEIGHT', 0.5)
    den_w = getattr(Config, 'SCORE_DENSITY_WEIGHT', 0.2)
    conf_w = getattr(Config, 'SCORE_CONFIDENCE_WEIGHT', 0.3)

    return [
        (sem * sem_w) + (den * den_w) + (conf * conf_w)
        for sem, den, conf in zip(semantic_scores, normalized_densities, confidences)
    ]

def build_windows(processed_segs, min_dur, ideal_max, hard_max, min_gap=0.5, require_natural_start=True):
    windows = []
    n = len(processed_segs)
    for i in range(n):
        if require_natural_start and not _starts_naturally(processed_segs, i, min_gap):
            continue
        anchor_start = processed_segs[i]['start']
        candidates = []
        for j in range(i, n):
            current_dur = processed_segs[j]['end'] - anchor_start
            if current_dur > hard_max:
                break
            if current_dur >= min_dur:
                candidates.append({'index': j, 'duration': current_dur})

        if not candidates:
            continue

        sentence_candidates = [
            c for c in candidates if _ends_on_sentence(processed_segs[c['index']]['text'])
        ]
        pool = sentence_candidates if sentence_candidates else candidates

        best_cut = min(pool, key=lambda x: abs(x['duration'] - ideal_max))
        idx = best_cut['index']
        windows.append({
            'text': " ".join([processed_segs[k]['text'] for k in range(i, idx + 1)]),
            'start': anchor_start,
            'end': processed_segs[idx]['end'],
            'confidence': _window_confidence(processed_segs, i, idx),
        })
    return windows

def get_speech_timestamps_from_file(wav_path):
    load_vad_model()
    if _vad_model is None or _vad_utils is None:
        raise RuntimeError("VAD model failed to load.")

    get_speech_timestamps, _, read_audio, _, _ = _vad_utils
    wav = read_audio(str(wav_path))
    return get_speech_timestamps(
        wav, _vad_model, sampling_rate=16000,
        threshold=getattr(Config, 'VAD_THRESHOLD', 0.6),
        return_seconds=True
    )

@torch.inference_mode()
def process_media(file_path, progress=gr.Progress()):
    if not file_path or not os.path.exists(file_path):
        return "Error\nFile not found.", None, None, None, "", ""

    is_video = file_path.lower().endswith(('.mp4', '.mkv', '.mov', '.avi', '.webm'))

    session_id = str(uuid.uuid4())
    session_dir = Path(tempfile.gettempdir()) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    transcription_ready_audio = str(session_dir / "transcribe_low_res.wav")
    master_clip = None

    if random.random() < 0.1:
        cleanup_thread = threading.Thread(
            target=cleanup_stale_sessions, 
            args=(1,),
            daemon=True
        )
        cleanup_thread.start()

    with _inference_lock:
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(file_path),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                transcription_ready_audio
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            speech_intervals = get_speech_timestamps_from_file(transcription_ready_audio)

            #Transcription
            batched_model = get_whisper_pipeline()

            transcribe_kwargs = dict(vad_filter=True, condition_on_previous_text=False)
            if isinstance(batched_model, BatchedInferencePipeline):
                transcribe_kwargs['batch_size'] = 16

            segments_gen, info = batched_model.transcribe(transcription_ready_audio, **transcribe_kwargs)

            min_speech_overlap = getattr(Config, 'MIN_SPEECH_OVERLAP', 0.6)
            max_compression_ratio = getattr(Config, 'MAX_COMPRESSION_RATIO', 2.4)

            processed_segs = []
            for s in segments_gen:
                speech_ratio = _speech_overlap_ratio(s.start, s.end, speech_intervals)
                if speech_ratio < min_speech_overlap:
                    continue
                if s.no_speech_prob > 0.35 or s.avg_logprob < -1.0 or s.compression_ratio > max_compression_ratio:
                    continue
                clean_text = re.sub(r'\[.*?\]|\(.*?\)|\♪', '', s.text).strip()
                if not clean_text or clean_text.lower() in ["thank you", "bye", "subscribe"]:
                    continue
                if _looks_repetitive(clean_text):
                    continue
                processed_segs.append({
                    'text': clean_text, 'start': s.start, 'end': s.end, 'speaker': "Speaker",
                    'avg_logprob': s.avg_logprob, 'no_speech_prob': s.no_speech_prob,
                })

            #Analysis
            embedder = get_embedder_model()
            window_args = (
                processed_segs,
                getattr(Config, 'MIN_CLIP_DURATION', 30.0),
                getattr(Config, 'MAX_CLIP_DURATION', 60.0),
                getattr(Config, 'MAX_HARD_DURATION', 90.0),
            )
            windows = build_windows(*window_args, min_gap=getattr(Config, 'MIN_NATURAL_GAP', 0.5))
            if not windows:
                windows = build_windows(*window_args, require_natural_start=False)

            full_transcript = " ".join([s['text'] for s in processed_segs])
            scores = get_optimized_scores(windows, embedder, full_transcript)
            for i, w in enumerate(windows):
                w['score'] = scores[i]

            min_score = getattr(Config, 'MIN_WINDOW_SCORE', 0.5)
            min_confidence = getattr(Config, 'MIN_WINDOW_CONFIDENCE', 0.65)
            ranked = sorted(
                [w for w in windows if w['score'] > min_score and w['confidence'] >= min_confidence],
                key=lambda x: x['score'], reverse=True
            )
            max_overlap_pct = getattr(Config, 'MAX_OVERLAP_PCT', 0.25)
            selected = []

            for cand in ranked:
                if len(selected) >= 3:
                    break
                overlap = any(_overlap_ratio(cand, sel) > max_overlap_pct for sel in selected)
                if not overlap:
                    selected.append(cand)

            selected = sorted(selected, key=lambda x: x['start'])

            if not selected:
                return (
                    "No segment in this file met the quality bar. Try a different "
                    "file. ",
                    None, None, None, str(session_dir), str(session_dir)
                )

            #Export
            clips = []
            fade_in = getattr(Config, 'FADE_IN_DURATION', 0.15)
            fade_out = getattr(Config, 'FADE_OUT_DURATION', 0.2)
            if is_video:
                master_clip = VideoFileClip(file_path)
                for i, hook in enumerate(selected):
                    path = session_dir / f"clip_{i+1}.mp4"
                    sub_clip = safe_slice(
                        master_clip,
                        max(0.0, hook['start'] - fade_in),
                        min(hook['end'] + fade_out, master_clip.duration)
                    )
                    sub_clip = apply_audio_fade(sub_clip, fade_in, fade_out)
                    sub_clip.write_videofile(
                        str(path), codec="libx264", audio_codec="aac",
                        audio_bitrate="320k", audio_fps=48000, preset="fast", logger=None
                    )
                    sub_clip.close()
                    clips.append(str(path))
            else:
                for i, hook in enumerate(selected):
                    path = session_dir / f"clip_{i+1}.mp3"
                    start, end = max(0.0, hook['start'] - fade_in), min(hook['end'] + fade_out, info.duration)
                    fade_start = max(0.0, (end - start) - fade_out)
                    cmd = [
                        "ffmpeg", "-y", "-ss", str(start), "-t", str(end - start), "-i", str(file_path),
                        "-af", f"afade=t=in:st=0:d={fade_in},afade=t=out:st={fade_start}:d={fade_out}",
                        "-acodec", "libmp3lame", "-b:a", "320k", "-ar", "48000", str(path)
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    clips.append(str(path))

            return (
                "Processing Complete!",
                *[clips[i] if i < len(clips) else None for i in range(3)],
                str(session_dir), str(session_dir)
            )

        except subprocess.CalledProcessError:
            logger.exception("ffmpeg command failed")
            return "Error: Could not process the media file.", None, None, None, str(session_dir), str(session_dir)
        except Exception as e:
            logger.exception("Pipeline failed")
            return f"Error: {str(e)}", None, None, None, str(session_dir), str(session_dir)
        finally:
            if master_clip is not None:
                master_clip.close()
            clear_memory()