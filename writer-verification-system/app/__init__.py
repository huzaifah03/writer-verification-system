from flask import Flask
from config import Config
from database.db import init_db
import os


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Ensure upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Initialize database
    init_db(app)

    # Register routes
    from app.routes import main
    app.register_blueprint(main)

    return app
