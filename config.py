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
    
    #Model Selection
    WHISPER_MODEL = "distil-large-v3" 
    CLASSIFIER_MODEL = "valhalla/distilbart-mnli-12-1" 
    EMBEDDER_MODEL = "all-MiniLM-L6-v2"
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
    
    MIN_WORDS_FOR_HOOK = 30 
    MAX_CLIP_DURATION = 90 
    SIMILARITY_THRESHOLD = 0.45 
    
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
