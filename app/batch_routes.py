"""
Batch upload, results, and API routes.

Blueprints:
  batch_bp  url_prefix="/batches"   — UI routes
  api_bp    url_prefix="/api"       — JSON endpoints
"""

import csv
import io
import os
import uuid

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, current_app, make_response, jsonify,
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from database.db import db, Batch, BatchPair, BatchSample
from app.batch_processor import generate_pair_records, start_batch_processing

batch_bp = Blueprint("batch", __name__, url_prefix="/batches")
api_bp   = Blueprint("api",   __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# Helpers (shared by upload and validation)
# ---------------------------------------------------------------------------

def _file_size(f) -> int:
    """Seek-based file size without consuming the stream."""
    f.seek(0, 2)
    size = f.tell()
    f.seek(0)
    return size


def _validate_files(files, allowed_exts, max_size) -> str | None:
    """Return an error string, or None if all files are valid."""
    if len(files) < 2 or len(files) > 50:
        return f"Upload between 2 and 50 files (got {len(files)})."
    for f in files:
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in allowed_exts:
            return (
                f"'{f.filename}' has a disallowed extension. "
                f"Allowed: {', '.join(sorted(allowed_exts))}."
            )
        if _file_size(f) > max_size:
            mb = max_size // (1024 * 1024)
            return f"'{f.filename}' exceeds the {mb} MB per-file limit."
    return None


def _batch_progress(batch_id: int, total_pairs: int) -> tuple[int, float]:
    """Return (pairs_completed, progress_percent) for a batch."""
    completed = BatchPair.query.filter(
        BatchPair.batch_id == batch_id,
        BatchPair.similarity_score.isnot(None),
    ).count()
    pct = round(completed / total_pairs * 100, 1) if total_pairs > 0 else 0.0
    return completed, pct


# ---------------------------------------------------------------------------
# batch_bp — /batches  (list)
# ---------------------------------------------------------------------------

@batch_bp.route("/")
@login_required
def batch_list():
    batches = (
        Batch.query
        .filter_by(teacher_id=current_user.id)
        .order_by(Batch.created_at.desc())
        .all()
    )
    return render_template("batch_list.html", batches=batches)


# ---------------------------------------------------------------------------
# batch_bp — /batches/new  (upload form)
# NOTE: this block is preserved exactly from Step B2 — do not modify.
# ---------------------------------------------------------------------------

@batch_bp.route("/new", methods=["GET", "POST"])
@login_required
def batch_new():
    if request.method == "GET":
        return render_template("batch_new.html")

    # --- Validate name ---
    name = request.form.get("name", "").strip()
    if not name or len(name) > 200:
        flash("Batch name is required and must be 200 characters or fewer.", "error")
        return render_template("batch_new.html"), 400

    # --- Validate threshold ---
    try:
        threshold = float(request.form.get("threshold", 0.75))
    except (ValueError, TypeError):
        threshold = 0.75
    if not (0.0 <= threshold <= 1.0):
        flash("Threshold must be between 0.0 and 1.0.", "error")
        return render_template("batch_new.html"), 400

    # --- Validate files ---
    files = [f for f in request.files.getlist("samples") if f.filename]
    error = _validate_files(
        files,
        current_app.config["ALLOWED_EXTENSIONS"],
        current_app.config["MAX_CONTENT_LENGTH"],
    )
    if error:
        flash(error, "error")
        return render_template("batch_new.html"), 400

    # --- Persist batch ---
    batch = Batch(
        teacher_id=current_user.id,
        name=name,
        threshold=threshold,
        status="pending",
        total_samples=len(files),
    )
    db.session.add(batch)
    db.session.flush()  # populate batch.id before saving files

    # --- Save files and create BatchSample rows ---
    batch_dir = os.path.join(
        current_app.config["UPLOAD_FOLDER"], "batches", str(batch.id)
    )
    os.makedirs(batch_dir, exist_ok=True)

    samples = []
    for idx, f in enumerate(files):
        ext = f.filename.rsplit(".", 1)[-1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        abs_path = os.path.join(batch_dir, unique_name)
        size = _file_size(f)
        f.save(abs_path)

        sample = BatchSample(
            batch_id=batch.id,
            filename=unique_name,
            original_filename=secure_filename(f.filename),
            file_path=abs_path,
            file_size_bytes=size,
            upload_order=idx,
        )
        db.session.add(sample)
        samples.append(sample)

    db.session.flush()  # populate sample IDs before generating pairs

    # --- Generate all-vs-all pairs ---
    pair_records = generate_pair_records(batch.id, [s.id for s in samples])
    db.session.add_all(pair_records)
    batch.total_pairs = len(pair_records)
    db.session.commit()

    # --- Start background processing ---
    start_batch_processing(current_app._get_current_object(), batch.id)

    return redirect(url_for("batch.batch_detail", batch_id=batch.id))


# ---------------------------------------------------------------------------
# batch_bp — /batches/<id>  (detail / results)
# ---------------------------------------------------------------------------

@batch_bp.route("/<int:batch_id>")
@login_required
def batch_detail(batch_id):
    batch = Batch.query.get_or_404(batch_id)
    if batch.teacher_id != current_user.id:
        return "Forbidden", 403

    samples = (
        BatchSample.query
        .filter_by(batch_id=batch_id)
        .order_by(BatchSample.upload_order)
        .all()
    )

    flagged_pairs = (
        BatchPair.query
        .filter(
            BatchPair.batch_id == batch_id,
            BatchPair.similarity_score >= batch.threshold,
        )
        .order_by(BatchPair.similarity_score.desc())
        .all()
    )

    total_completed, progress_percent = _batch_progress(batch_id, batch.total_pairs)

    return render_template(
        "batch_detail.html",
        batch=batch,
        samples=samples,
        flagged_pairs=flagged_pairs,
        total_completed=total_completed,
        progress_percent=progress_percent,
    )


# ---------------------------------------------------------------------------
# batch_bp — /batches/<id>/export.csv
# ---------------------------------------------------------------------------

@batch_bp.route("/<int:batch_id>/export.csv")
@login_required
def batch_export_csv(batch_id):
    batch = Batch.query.get_or_404(batch_id)
    if batch.teacher_id != current_user.id:
        return "Forbidden", 403

    pairs = (
        BatchPair.query
        .filter(BatchPair.batch_id == batch_id)
        .order_by(BatchPair.similarity_score.desc())
        .all()
    )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Sample1", "Sample2", "Score", "Decision", "Risk"])
    for p in pairs:
        writer.writerow([
            p.sample1.original_filename,
            p.sample2.original_filename,
            p.similarity_score,
            p.decision,
            p.risk_level,
        ])

    response = make_response(buf.getvalue())
    response.headers["Content-Type"] = "text/csv"
    response.headers["Content-Disposition"] = (
        f"attachment; filename=batch_{batch_id}_results.csv"
    )
    return response


# ---------------------------------------------------------------------------
# api_bp — /api/batches/<id>/status  (JSON polling endpoint)
# ---------------------------------------------------------------------------

@api_bp.route("/batches/<int:batch_id>/status")
@login_required
def batch_status_api(batch_id):
    batch = Batch.query.get_or_404(batch_id)
    if batch.teacher_id != current_user.id:
        return jsonify({"error": "Forbidden"}), 403

    pairs_completed, progress_percent = _batch_progress(batch_id, batch.total_pairs)

    flagged_pairs = BatchPair.query.filter(
        BatchPair.batch_id == batch_id,
        BatchPair.similarity_score >= batch.threshold,
    ).count()

    return jsonify({
        "batch_id":        batch.id,
        "status":          batch.status,
        "total_samples":   batch.total_samples,
        "total_pairs":     batch.total_pairs,
        "pairs_completed": pairs_completed,
        "flagged_pairs":   flagged_pairs,
        "threshold":       batch.threshold,
        "progress_percent": progress_percent,
    })