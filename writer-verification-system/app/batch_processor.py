"""
Batch processing helpers — pair generation and background runner.
"""

import itertools
import threading
from datetime import datetime

from database.db import db, Batch, BatchPair, BatchSample

_processing_lock = threading.Lock()


def generate_pair_records(batch_id: int, sample_ids: list) -> list:
    """
    Return a list of un-committed BatchPair objects covering every unique
    unordered pair from sample_ids.  n samples → n*(n-1)/2 pairs.
    All score/decision/risk fields are left as None (uncomputed).
    """
    pairs = []
    for s1_id, s2_id in itertools.combinations(sample_ids, 2):
        pairs.append(BatchPair(
            batch_id=batch_id,
            sample1_id=s1_id,
            sample2_id=s2_id,
        ))
    return pairs


def process_batch(app, batch_id: int) -> None:
    """
    Run the all-vs-all verification pipeline for one batch.
    Intended to run inside a daemon thread — opens its own app context.

    Each pair is committed individually so partial progress survives a crash.
    On any exception: marks the batch as "failed", logs, then re-raises.
    """
    from model.predict import verify_writers

    with app.app_context():
        try:
            batch = Batch.query.get(batch_id)
            batch.status = "running"
            db.session.commit()

            pending_pairs = (
                BatchPair.query
                .filter_by(batch_id=batch_id)
                .filter(BatchPair.similarity_score.is_(None))
                .all()
            )

            for pair in pending_pairs:
                s1 = BatchSample.query.get(pair.sample1_id)
                s2 = BatchSample.query.get(pair.sample2_id)
                result = verify_writers(s1.file_path, s2.file_path)
                pair.similarity_score = result["similarity_score"]
                pair.decision = result["decision"]
                pair.risk_level = result["risk_level"]
                pair.computed_at = datetime.utcnow()
                db.session.commit()

            batch = Batch.query.get(batch_id)
            batch.flagged_pairs = (
                BatchPair.query
                .filter(
                    BatchPair.batch_id == batch_id,
                    BatchPair.similarity_score >= batch.threshold,
                )
                .count()
            )
            batch.status = "completed"
            batch.completed_at = datetime.utcnow()
            db.session.commit()

        except Exception:
            try:
                batch = Batch.query.get(batch_id)
                batch.status = "failed"
                db.session.commit()
            except Exception:
                db.session.rollback()
            app.logger.exception(f"Batch {batch_id} processing failed.")
            raise


def start_batch_processing(app, batch_id: int) -> None:
    """
    Enforce a single-runner constraint via a module-level lock, then launch
    a daemon thread targeting process_batch.

    Must be called from within an active Flask request context (e.g. from a
    route handler) — uses the existing DB session for the status check and
    the initial "running" write while the lock is held.

    If another batch is already running, this batch is left as "pending" and
    no thread is started.
    """
    with _processing_lock:
        running = Batch.query.filter_by(status="running").first()
        if running is not None:
            app.logger.info(
                f"Batch {batch_id} left as pending — "
                f"batch {running.id} is already running."
            )
            return

        batch = Batch.query.get(batch_id)
        batch.status = "running"
        db.session.commit()

    thread = threading.Thread(
        target=process_batch,
        args=(app, batch_id),
        daemon=True,
    )
    thread.start()