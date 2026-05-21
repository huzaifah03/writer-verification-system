"""
Authentication blueprint — login / logout.
No public signup. Accounts are created via scripts/seed_teacher.py.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from database.db import Teacher

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        teacher = Teacher.query.filter_by(email=email).first()
        if teacher and teacher.check_password(password):
            login_user(teacher)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("main.index"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))