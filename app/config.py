import os
from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "dev")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pth")
API_KEY = os.getenv("API_KEY", "")
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "yourdockerhubusername/house-seg-lab")
