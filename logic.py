import torch
import os
import uuid
import logging
import gc
import shutil
import tempfile
from pathlib import Path
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.effects import normalize
from transformers import pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
from sentence_transformers import SentenceTransformer, util
from config import Config
import gradio as gr

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy.video.io.VideoFileClip import VideoFileClip

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
    
    gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio")
    if os.path.exists(gradio_tmp):
        shutil.rmtree(gradio_tmp, ignore_errors=True)
    clear_memory()

def get_intersection_speaker(seg_start, seg_end, speaker_turns):
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
    if not file_path: return [None]*7
    
    session_id = str(uuid.uuid4())
    session_dir = Path(tempfile.gettempdir()) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    is_video = file_path.lower().endswith(('.mp4', '.mkv', '.mov', '.avi', '.webm'))
    processing_path = file_path

    try:
        #Video to Audio extraction (Podcast Optimized)
        if is_video:
            progress(0.05, desc="Extracting audio (Optimizing for RAM)...")
            video = VideoFileClip(file_path)
            processing_path = str(session_dir / "extracted_audio.wav")
            #Downsampling to 16kHz Mono for long podcasts
            video.audio.write_audiofile(
                processing_path, 
                fps=16000, 
                nbytes=2, 
                buffersize=2000, 
                ffmpeg_params=["-ac", "1"], 
                logger=None
            )
            video.close()
            clear_memory()

        #Diarization
        progress(0.1, desc="Identifying Speakers (This may take a while for long files...)")
        diar_pipe = DiarizationPipeline.from_pretrained(Config.DIARIZATION_MODEL, use_auth_token=Config.HF_TOKEN)
        diar_map = diar_pipe(processing_path)
        speaker_turns = [{'start': t.start, 'end': t.end, 'speaker': s} for t, _, s in diar_map.itertracks(yield_label=True)]
        del diar_pipe 
        clear_memory()

        #Transcription (Iterative for long files)
        progress(0.3, desc="Starting Transcription...")
        whisper = WhisperModel(Config.WHISPER_MODEL, device=Config.DEVICE, compute_type=Config.COMPUTE_TYPE)
        # Using a generator to update progress for long podcasts
        segments_gen, info = whisper.transcribe(processing_path, vad_filter=True)
        
        processed_segs = []
        #total_duration helps calculate percentage
        total_duration = info.duration
        
        for s in segments_gen:
            processed_segs.append({
                'text': s.text.strip(), 
                'start': s.start, 
                'end': s.end, 
                'speaker': get_intersection_speaker(s.start, s.end, speaker_turns)
            })
            #Update progress bar every segment based on time elapsed
            current_progress = 0.3 + (s.end / total_duration * 0.4) 
            progress(current_progress, desc=f"Transcribing: {int(s.end)}s / {int(total_duration)}s")

        del whisper
        clear_memory()

        #Scoring Viral Moments
        progress(0.75, desc="Analyzing content for viral hooks...")
        embedder = SentenceTransformer(Config.EMBEDDER_MODEL, device=Config.DEVICE)
        classifier = pipeline("zero-shot-classification", model=Config.CLASSIFIER_MODEL, device=-1)
        
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

        #Exporting (Heavy step for video)
        progress(0.9, desc="Cutting and exporting viral clips...")
        clips = []
        
        if is_video:
            video_full = VideoFileClip(file_path)
            for i, hook in enumerate(selected):
                path = session_dir / f"clip_{i+1}.mp4"
                video_full.subclip(hook['start'], hook['end']).write_videofile(
                    str(path), codec="libx264", audio_codec="aac", temp_audiofile=str(session_dir/"temp.m4a"), 
                    remove_temp=True, logger=None
                )
                clips.append(str(path))
            video_full.close()
        else:
            audio = AudioSegment.from_file(file_path)
            normalized_audio = normalize(audio)
            for i, hook in enumerate(selected):
                path = session_dir / f"hook_{i+1}.mp3"
                normalized_audio[int(hook['start']*1000):int(hook['end']*1000)].fade_in(200).fade_out(200).export(str(path), format="mp3")
                clips.append(str(path))

        while len(clips) < 3: clips.append(None)
        full_transcript = "\n".join([f"[{s['speaker']}] {s['text']}" for s in processed_segs])
        
        del embedder, classifier
        clear_memory()
        return full_transcript, f"Processing Complete!", *clips, str(session_dir)

    except Exception as e:
        logger.error(f"Critical System Error: {e}")
        return str(e), "Error during processing", None, None, None, None, ""
