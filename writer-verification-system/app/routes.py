"""
Flask Routes
-------------
Handles:
    GET  /          → Upload page (index)
    POST /verify    → Run writer verification on two uploaded images
    GET  /history   → View past verification results
    GET  /result/<id> → View a single result
"""

import os
import uuid
from flask import Blueprint, request, jsonify, render_template, current_app
from werkzeug.utils import secure_filename
from database.db import db, Assignment, VerificationResult
from model.predict import verify_writers

main = Blueprint("main", __name__)


def allowed_file(filename: str) -> bool:
    allowed = current_app.config["ALLOWED_EXTENSIONS"]
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def save_upload(file) -> tuple[str, str]:
    """
    Save an uploaded file to the uploads folder with a unique name.

    Returns:
        (unique_filename, absolute_path)
    """
    original_name = secure_filename(file.filename)
    ext = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)
    return unique_name, save_path


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@main.route("/")
def index():
    """Render the upload page."""
    return render_template("index.html")


@main.route("/history")
def history():
    """Render past verification results."""
    results = VerificationResult.query.order_by(VerificationResult.verified_at.desc()).all()
    return render_template("history.html", results=results)


@main.route("/result/<int:result_id>")
def result_detail(result_id):
    """Render a single verification result page."""
    result = VerificationResult.query.get_or_404(result_id)
    return render_template("result.html", result=result)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@main.route("/verify", methods=["POST"])
def verify():
    """
    POST /verify
    Accepts two image files (assignment1, assignment2).
    Runs the writer verification pipeline and returns JSON results.
    Also saves the result to the database.
    """
    # Validate files are present
    if "assignment1" not in request.files or "assignment2" not in request.files:
        return jsonify({"error": "Please upload both assignment images."}), 400

    file1 = request.files["assignment1"]
    file2 = request.files["assignment2"]

    if file1.filename == "" or file2.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file1.filename) or not allowed_file(file2.filename):
        return jsonify({"error": "Only PNG and JPG images are allowed."}), 400

    try:
        # Save uploads
        name1, path1 = save_upload(file1)
        name2, path2 = save_upload(file2)

        # Save assignment records to DB
        a1 = Assignment(filename=name1, original_name=secure_filename(file1.filename))
        a2 = Assignment(filename=name2, original_name=secure_filename(file2.filename))
        db.session.add_all([a1, a2])
        db.session.flush()  # Get IDs before commit

        # Run verification pipeline
        result = verify_writers(path1, path2)

        # Save verification result to DB
        vr = VerificationResult(
            assignment1_id=a1.id,
            assignment2_id=a2.id,
            similarity_score=result["similarity_score"],
            decision=result["decision"],
            risk_level=result["risk_level"],
        )
        db.session.add(vr)
        db.session.commit()

        return jsonify({
            "result_id": vr.id,
            **result
        }), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Verification error: {e}")
        return jsonify({"error": "An error occurred during verification. Please try again."}), 500
