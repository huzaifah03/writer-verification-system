"""
Seed a single demo teacher account into the database.
Run once before first use:
    venv\\Scripts\\python.exe scripts\\seed_teacher.py

If a teacher already exists the script exits without making any changes.
"""

import sys
import os
import secrets
import string

# Ensure the project root is on sys.path regardless of where the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from database.db import db, Teacher


def seed():
    app = create_app()
    with app.app_context():
        if Teacher.query.count() > 0:
            print("Teacher account already exists.")
            return

        alphabet = string.ascii_letters + string.digits
        password = "".join(secrets.choice(alphabet) for _ in range(12))

        teacher = Teacher(email="teacher@iqra.edu.pk", name="Demo Teacher")
        teacher.set_password(password)
        db.session.add(teacher)
        db.session.commit()

        print("=" * 44)
        print("  Demo Teacher Account Created")
        print("=" * 44)
        print(f"  Email   : teacher@iqra.edu.pk")
        print(f"  Password: {password}")
        print("=" * 44)
        print("  Save these credentials — the password")
        print("  cannot be recovered from the database.")
        print("=" * 44)


if __name__ == "__main__":
    seed()