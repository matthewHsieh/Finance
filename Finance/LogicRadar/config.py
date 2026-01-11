import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Explicitly look for .env in the same directory as this config.py file
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    
    # LLM Settings (Support for Local/Ollama or 'terminal')
    # Default to OpenAI if not set
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1") 
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "terminal") # Default to terminal for manual check
    
    # Data Paths
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    
    # Analysis Settings
    CORRELATION_THRESHOLD = 0.8
    MIN_DATA_POINTS = 100
    LOOKBACK_WINDOW = 252 # 1 Trading Year

    @staticmethod
    def check_keys():
        """Helper to ensure keys are present"""
        missing = []
        
        # FRED is always required for data
        if not Config.FRED_API_KEY:
            missing.append("FRED_API_KEY")

        # LLM Check
        if Config.LLM_PROVIDER == 'terminal':
            # No keys needed for terminal mode
            pass 
        elif "openai.com" in Config.LLM_BASE_URL and not Config.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
            
        if missing:
            print(f"⚠️ Warning: Missing API Keys: {', '.join(missing)}")
            print(f"Checked .env at: {env_path}")
            print("Please ensure your .env file is correct.")
