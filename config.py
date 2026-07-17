#Configuration & Logging
from dotenv import load_dotenv
import logging
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Podcast Assistant")

class Config:
    HF_TOKEN = os.getenv("HF_TOKEN")
    DEVICE = "cpu"  
    COMPUTE_TYPE = "int8" #Saves 50% RAM on CPU 
    VAD_USE_ONNX = True

    #Model Selection
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
    EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization@2.1")

    MAX_FILE_SIZE_MB = 500

    MIN_CLIP_DURATION = 30.0
    MAX_CLIP_DURATION = 60.0
    MAX_HARD_DURATION = 90.0

    VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.mov', '.avi', '.webm')
    AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac')
    
    WHISPER_VAD_FILTER = True

    USE_BATCHED_INFERENCE = True
    VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.7"))
    MIN_SPEECH_OVERLAP = float(os.getenv("MIN_SPEECH_OVERLAP", "0.75"))

    MAX_COMPRESSION_RATIO = 2.4

    SCORE_SEMANTIC_WEIGHT = float(os.getenv("SCORE_SEMANTIC_WEIGHT", "0.5"))
    SCORE_DENSITY_WEIGHT = float(os.getenv("SCORE_DENSITY_WEIGHT", "0.2"))
    SCORE_CONFIDENCE_WEIGHT = float(os.getenv("SCORE_CONFIDENCE_WEIGHT", "0.3"))

    MIN_NATURAL_GAP = 0.5

    MIN_WINDOW_SCORE = 0.5
    MIN_WINDOW_CONFIDENCE = 0.65
    MAX_OVERLAP_PCT = 0.25

    @classmethod
    def validate_env(cls):
        """Checks for required secrets before running."""
        if not cls.HF_TOKEN:
            raise EnvironmentError(
                "Env Error"
            )
