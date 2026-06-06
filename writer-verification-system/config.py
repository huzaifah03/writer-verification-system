import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = os.environ.get("DEBUG", "True") == "True"

    # File uploads
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

    # Database
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "database", "writer_verification.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Model
    MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "writer_verification_model.pth")

    # Image preprocessing
    IMAGE_SIZE = (224, 224)  # ResNet50 input size

    # Similarity threshold — ROC-optimal from evaluation (Youden J, hybrid mode)
    # Scores ABOVE this → Same Writer, BELOW → Different Writer
    SIMILARITY_THRESHOLD = 0.764
