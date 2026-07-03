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
    WHISPER_MODEL = "base" 
    EMBEDDER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DIARIZATION_MODEL = "pyannote/speaker-diarization@2.1"

    MAX_FILE_SIZE_MB = 500
    MAX_CLIP_DURATION = 120
    MIN_WORDS_FOR_HOOK = 30 
    SIMILARITY_THRESHOLD = 0.45 
    
    VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.mov', '.avi', '.webm')
    AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac')
    
    WHISPER_VAD_FILTER = True

    WEIGHTS = {
        "profound insight": 5.0,
        "actionable advice": 4.5,
        "emotional storytelling": 4.0,
        "controversial opinion": 3.5,
        "casual small talk": -5.0
    }
    ANCHOR_THEMES = list(WEIGHTS.keys())

    @classmethod
    def validate_env(cls):
        """Checks for required secrets before running."""
        if not cls.HF_TOKEN:
            raise EnvironmentError(
                "Env Error"
            )
