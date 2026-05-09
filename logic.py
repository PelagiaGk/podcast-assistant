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

# Resource Management
def clear_memory():
    """Aggressively clears RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def cleanup_session(session_path):
    """Deletes temporary files and Gradio's internal cache."""
    if session_path and os.path.exists(session_path):
        try:
            shutil.rmtree(session_path)
            logger.info(f"Session directory deleted: {session_path}")
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
    
    gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio")
    if os.path.exists(gradio_tmp):
        shutil.rmtree(gradio_tmp, ignore_errors=True)
        
    clear_memory()
    return [None, "Session Scrubbed. Files deleted.", None, None, None, None, ""]

#Audio Logic 

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
def process_audio(audio_path, progress=gr.Progress()):
    if not audio_path: return [None]*7
    
    session_id = str(uuid.uuid4())
    session_dir = Path(tempfile.gettempdir()) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        #Diarization
        progress(0.1, desc="Step 1: Identifying Speakers")
        diar_pipe = DiarizationPipeline.from_pretrained(Config.DIARIZATION_MODEL, use_auth_token=Config.HF_TOKEN)
        diar_map = diar_pipe(audio_path)
        speaker_turns = [{'start': t.start, 'end': t.end, 'speaker': s} for t, _, s in diar_map.itertracks(yield_label=True)]
        
        del diar_pipe 
        clear_memory()

        #Transcription
        progress(0.4, desc="Step 2: Transcribing Audio")
        whisper = WhisperModel(Config.WHISPER_MODEL, device=Config.DEVICE, compute_type=Config.COMPUTE_TYPE)
        segments, _ = whisper.transcribe(audio_path, vad_filter=True)
        
        processed_segs = []
        for s in segments:
            spk = get_intersection_speaker(s.start, s.end, speaker_turns)
            processed_segs.append({'text': s.text.strip(), 'start': s.start, 'end': s.end, 'speaker': spk})
        
        del whisper
        clear_memory()

        #Scoring
        progress(0.7, desc="Step 3: Finding Viral Moments")
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

        #Exporting
        progress(0.9, desc="Step 4: Finalizing Clips...")
        audio = AudioSegment.from_file(audio_path)
        normalized_audio = normalize(audio)
        
        clips = []
        for i, hook in enumerate(selected):
            start_ms, end_ms = int(hook['start'] * 1000), int(hook['end'] * 1000)
            path = session_dir / f"hook_{i+1}.mp3"
            normalized_audio[start_ms:end_ms].fade_in(200).fade_out(200).export(str(path), format="mp3")
            clips.append(str(path))

        while len(clips) < 3: clips.append(None)
        
        full_transcript = "\n".join([f"[{s['speaker']}] {s['text']}" for s in processed_segs])
        
        del embedder, classifier, audio, normalized_audio
        clear_memory()
        
        return full_transcript, f"✅ Clips Ready!", *clips, str(session_dir)

    except Exception as e:
        logger.error(f"Critical System Error: {e}")
        return str(e), "❌ Processing Error", None, None, None, None, ""
