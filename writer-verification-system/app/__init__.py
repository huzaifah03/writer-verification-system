from flask import Flask
from flask_login import LoginManager
from config import Config
from database.db import init_db
import os

login_manager = LoginManager()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Ensure upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Initialize database
    init_db(app)

    # Ensure at least one admin exists — promote oldest teacher if none do
    with app.app_context():
        from database.db import Teacher, db as _db
        if Teacher.query.count() > 0 and Teacher.query.filter_by(is_admin=True).count() == 0:
            first = Teacher.query.order_by(Teacher.created_at).first()
            first.is_admin = True
            _db.session.commit()
            app.logger.info(f"Promoted '{first.email}' to admin (no admin existed).")

    # Mark any batch left in "running" state by a prior crashed process as failed
    with app.app_context():
        from database.db import Batch, db as _db
        orphans = Batch.query.filter_by(status="running").all()
        for b in orphans:
            b.status = "failed"
        if orphans:
            _db.session.commit()
            app.logger.warning(f"Marked {len(orphans)} orphan batches as failed.")

    # Initialize Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    from database.db import Teacher

    @login_manager.user_loader
    def load_user(user_id):
        return Teacher.query.get(int(user_id))

    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)

    from app.auth import auth_bp
    app.register_blueprint(auth_bp)

    from app.batch_routes import batch_bp, api_bp
    app.register_blueprint(batch_bp)
    app.register_blueprint(api_bp)

    from app.admin import admin_bp
    app.register_blueprint(admin_bp)

    return app
