from functools import wraps
from flask import Blueprint, render_template, redirect, url_for, flash, request, abort
from flask_login import login_required, current_user
from database.db import db, Teacher

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated


@admin_bp.route("/teachers")
@login_required
@admin_required
def teacher_list():
    teachers = Teacher.query.order_by(Teacher.created_at).all()
    return render_template("admin_teachers.html", teachers=teachers)


@admin_bp.route("/teachers/new", methods=["POST"])
@login_required
@admin_required
def teacher_new():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()
    is_admin = request.form.get("is_admin") == "on"

    if not name or not email or not password:
        flash("Name, email, and password are all required.", "error")
        return redirect(url_for("admin.teacher_list"))

    if Teacher.query.filter_by(email=email).first():
        flash(f"A teacher with email {email} already exists.", "error")
        return redirect(url_for("admin.teacher_list"))

    teacher = Teacher(name=name, email=email, is_admin=is_admin)
    teacher.set_password(password)
    db.session.add(teacher)
    db.session.commit()
    flash(f"Teacher '{name}' added successfully.", "success")
    return redirect(url_for("admin.teacher_list"))


@admin_bp.route("/teachers/<int:teacher_id>/delete", methods=["POST"])
@login_required
@admin_required
def teacher_delete(teacher_id):
    teacher = Teacher.query.get_or_404(teacher_id)
    if teacher.id == current_user.id:
        flash("You cannot delete your own account.", "error")
        return redirect(url_for("admin.teacher_list"))
    db.session.delete(teacher)
    db.session.commit()
    flash(f"Teacher '{teacher.name}' deleted.", "success")
    return redirect(url_for("admin.teacher_list"))


@admin_bp.route("/teachers/<int:teacher_id>/reset", methods=["POST"])
@login_required
@admin_required
def teacher_reset(teacher_id):
    teacher = Teacher.query.get_or_404(teacher_id)
    new_password = request.form.get("new_password", "").strip()
    if not new_password:
        flash("New password cannot be empty.", "error")
        return redirect(url_for("admin.teacher_list"))
    teacher.set_password(new_password)
    db.session.commit()
    flash(
        f"Password for '{teacher.name}' has been reset. "
        f"New password: {new_password}",
        "success"
    )
    return redirect(url_for("admin.teacher_list"))
