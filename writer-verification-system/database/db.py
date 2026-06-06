"""
Database models and initialization.
Uses SQLite via Flask-SQLAlchemy.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class Assignment(db.Model):
    """Stores uploaded assignment image metadata."""
    __tablename__ = "assignments"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Assignment {self.original_name}>"


class VerificationResult(db.Model):
    """Stores the result of each writer verification comparison."""
    __tablename__ = "verification_results"

    id = db.Column(db.Integer, primary_key=True)

    # The two assignments being compared
    assignment1_id = db.Column(db.Integer, db.ForeignKey("assignments.id"), nullable=False)
    assignment2_id = db.Column(db.Integer, db.ForeignKey("assignments.id"), nullable=False)

    # Similarity score between 0 and 1
    similarity_score = db.Column(db.Float, nullable=False)

    # Final decision: "Same Writer" or "Different Writer"
    decision = db.Column(db.String(50), nullable=False)

    # Plagiarism risk level: "Low", "Medium", "High"
    risk_level = db.Column(db.String(20), nullable=False)

    verified_at = db.Column(db.DateTime, default=datetime.utcnow)

    assignment1 = db.relationship("Assignment", foreign_keys=[assignment1_id])
    assignment2 = db.relationship("Assignment", foreign_keys=[assignment2_id])

    def __repr__(self):
        return f"<Result {self.decision} | Score: {self.similarity_score:.2f}>"


# ---------------------------------------------------------------------------
# Auth models
# ---------------------------------------------------------------------------

class Teacher(UserMixin, db.Model):
    """Registered teacher account. Only seeded teachers can log in."""
    __tablename__ = "teachers"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    batches = db.relationship("Batch", backref="teacher", lazy=True)

    def set_password(self, plain: str) -> None:
        self.password_hash = generate_password_hash(plain)

    def check_password(self, plain: str) -> bool:
        return check_password_hash(self.password_hash, plain)

    def __repr__(self):
        return f"<Teacher {self.email}>"


# ---------------------------------------------------------------------------
# Batch models
# ---------------------------------------------------------------------------

class Batch(db.Model):
    """A batch verification job submitted by a teacher."""
    __tablename__ = "batches"

    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey("teachers.id"), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), default="pending")   # pending|running|completed|failed
    threshold = db.Column(db.Float, default=0.75)
    total_samples = db.Column(db.Integer, default=0)
    total_pairs = db.Column(db.Integer, default=0)
    flagged_pairs = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    samples = db.relationship("BatchSample", backref="batch", cascade="all, delete-orphan")
    pairs = db.relationship("BatchPair", backref="batch", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Batch {self.name} [{self.status}]>"


class BatchSample(db.Model):
    """A single uploaded handwriting image within a batch."""
    __tablename__ = "batch_samples"

    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey("batches.id"), nullable=False)
    filename = db.Column(db.String(255), nullable=False)           # UUID storage name
    original_filename = db.Column(db.String(255), nullable=False)  # original upload name
    file_path = db.Column(db.String(512), nullable=False)
    file_size_bytes = db.Column(db.Integer, default=0)
    upload_order = db.Column(db.Integer, default=0)

    def __repr__(self):
        return f"<BatchSample {self.original_filename}>"


class BatchPair(db.Model):
    """A computed similarity pair within a batch."""
    __tablename__ = "batch_pairs"

    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey("batches.id"), nullable=False)
    sample1_id = db.Column(db.Integer, db.ForeignKey("batch_samples.id"), nullable=False)
    sample2_id = db.Column(db.Integer, db.ForeignKey("batch_samples.id"), nullable=False)
    similarity_score = db.Column(db.Float, nullable=True)
    decision = db.Column(db.String(50), nullable=True)
    risk_level = db.Column(db.String(20), nullable=True)
    computed_at = db.Column(db.DateTime, nullable=True)

    sample1 = db.relationship("BatchSample", foreign_keys=[sample1_id])
    sample2 = db.relationship("BatchSample", foreign_keys=[sample2_id])

    def __repr__(self):
        return f"<BatchPair {self.id} [{self.decision}]>"


def init_db(app):
    """Initialize the database with the Flask app."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        # Safe migration: add is_admin column if this is an existing DB that predates it
        from sqlalchemy import inspect, text
        insp = inspect(db.engine)
        cols = [c["name"] for c in insp.get_columns("teachers")]
        if "is_admin" not in cols:
            db.session.execute(
                text("ALTER TABLE teachers ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT 0")
            )
            db.session.commit()
