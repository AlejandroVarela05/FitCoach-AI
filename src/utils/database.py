# database.py


# This module manages persistent storage for FitCoach AI using SQLite and SQLAlchemy.
# It defines the database schema (UserProfile, TrainingSession, ExerciseRecord)
# and provides a DatabaseManager class to easily create, read, and update records.
#
# PURPOSE:
#   - Store user profile information (goal, fitness level, training days, streak).
#   - Log every training session with duration, reps, and average posture score.
#   - Keep detailed exercise‑level records including errors and targets.
#   - Enable future analysis of progress and trends.
#
# COURSE CONNECTION:
#   While not tied to a specific lecture, this module supports the overall system
#   architecture by adding data persistence. It is a practical implementation of
#   software engineering best practices (separation of concerns, ORM usage).
#
# DECISIONS:
#   - I chose SQLite because it is lightweight, file‑based, and requires no server.
#   - I use SQLAlchemy ORM to abstract database operations and make the code more readable.
#   - The DatabaseManager class centralises all database interactions, making it easy
#     to swap the backend or mock the database in tests.
#   - I automatically create the tables on initialisation so the system is ready to use
#     without extra setup steps.



import os
import sys
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import DATABASE_PATH

# I create the base class for all ORM models.
Base = declarative_base()


class UserProfile(Base):
    # This table stores the user's profile and high‑level training preferences.
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, default="User")
    goal = Column(String)           # e.g. "lose_weight", "gain_muscle"
    level = Column(String)          # "beginner", "intermediate", "advanced"
    days_per_week = Column(Integer)
    weeks_total = Column(Integer)
    week_current = Column(Integer, default=1)
    streak_days = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TrainingSession(Base):
    # Each row represents one completed (or partially completed) workout session.
    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, default=1)
    date = Column(DateTime, default=datetime.utcnow)
    duration_seconds = Column(Float, default=0)
    exercises_completed = Column(Integer, default=0)
    total_reps = Column(Integer, default=0)
    avg_posture_score = Column(Float, default=0.0)   # 0.0 to 1.0
    session_completed = Column(Boolean, default=False)


class ExerciseRecord(Base):
    # Detailed records for each exercise performed within a session.
    __tablename__ = "exercise_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer)
    exercise_name = Column(String)
    reps_completed = Column(Integer, default=0)
    reps_target = Column(Integer, default=0)
    posture_score = Column(Float, default=0.0)
    errors_knee = Column(Integer, default=0)
    errors_back = Column(Integer, default=0)
    errors_depth = Column(Integer, default=0)
    errors_rhythm = Column(Integer, default=0)
    errors_neck = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    # This class provides a high‑level API for all database operations.

    def __init__(self, db_path=None):
        # I use the default database path from the configuration if none is provided.
        if db_path is None:
            db_path = DATABASE_PATH

        # I create the SQLAlchemy engine for SQLite.
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)

        # I ensure all tables exist in the database.
        Base.metadata.create_all(self.engine)

        # I create a session factory for creating new database sessions.
        self.Session = sessionmaker(bind=self.engine)

    def get_or_create_profile(self, **kwargs):
        # I retrieve the existing user profile or create a new one if none exists.
        session = self.Session()
        profile = session.query(UserProfile).first()
        if profile is None:
            profile = UserProfile(**kwargs)
            session.add(profile)
            session.commit()
        session.close()
        return profile

    def update_profile(self, **kwargs):
        # I update the user profile with the given keyword arguments.
        session = self.Session()
        profile = session.query(UserProfile).first()
        if profile:
            for key, value in kwargs.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            session.commit()
        session.close()

    def save_session(self, duration, exercises_done, total_reps, avg_score, completed):
        # I save a new training session and return its auto‑generated ID.
        session = self.Session()
        training = TrainingSession(
            duration_seconds=duration,
            exercises_completed=exercises_done,
            total_reps=total_reps,
            avg_posture_score=avg_score,
            session_completed=completed,
        )
        session.add(training)
        session.commit()
        session_id = training.id
        session.close()
        return session_id

    def save_exercise_record(self, session_id, exercise_name, reps, target,
                              score, errors):
        # I save detailed metrics for a single exercise performed in a session.
        db_session = self.Session()
        record = ExerciseRecord(
            session_id=session_id,
            exercise_name=exercise_name,
            reps_completed=reps,
            reps_target=target,
            posture_score=score,
            errors_knee=errors.get("knee_error", 0),
            errors_back=errors.get("back_error", 0),
            errors_depth=errors.get("depth_error", 0),
            errors_rhythm=errors.get("rhythm_error", 0),
            errors_neck=errors.get("neck_error", 0),
        )
        db_session.add(record)
        db_session.commit()
        db_session.close()

    def get_all_sessions(self):
        # I return a list of all training sessions, ordered by date (most recent first).
        session = self.Session()
        sessions = session.query(TrainingSession).order_by(
            TrainingSession.date.desc()
        ).all()
        session.close()
        return sessions

    def get_exercise_history(self, exercise_name=None):
        # I return exercise records, optionally filtered by exercise name.
        session = self.Session()
        query = session.query(ExerciseRecord)
        if exercise_name:
            query = query.filter(ExerciseRecord.exercise_name == exercise_name)
        records = query.order_by(ExerciseRecord.timestamp.desc()).all()
        session.close()
        return records

    def get_streak(self):
        # I retrieve the current streak (consecutive days of training) from the user profile.
        session = self.Session()
        profile = session.query(UserProfile).first()
        streak = profile.streak_days if profile else 0
        session.close()
        return streak


if __name__ == "__main__":
    # I run a quick self‑test to verify the database functionality.
    print("Testing database...")

    import tempfile
    test_db = os.path.join(tempfile.gettempdir(), "fitcoach_test.db")
    db = DatabaseManager(db_path=test_db)

    # Create a test profile.
    profile = db.get_or_create_profile(
        name="Test User",
        goal="lose_weight",
        level="beginner",
        days_per_week=3,
        weeks_total=8
    )
    print(f"  Profile created: {profile.name}, goal={profile.goal}")

    # Save a dummy training session.
    sid = db.save_session(
        duration=1200,
        exercises_done=4,
        total_reps=48,
        avg_score=0.85,
        completed=True
    )
    print(f"  Session saved with ID: {sid}")

    # Save an exercise record for that session.
    db.save_exercise_record(
        session_id=sid,
        exercise_name="squat",
        reps=12,
        target=12,
        score=0.9,
        errors={"knee_error": 2, "back_error": 1}
    )
    print("  Exercise record saved")

    # Verify that we can retrieve the session.
    sessions = db.get_all_sessions()
    print(f"  Total sessions in DB: {len(sessions)}")

    # Clean up the test database file.
    os.remove(test_db)
    print("Database tests passed!")