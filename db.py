import sqlite3
import pandas as pd
import datetime

DATABASE = "community_feedback.db"


def get_db_connection():
    """Establish and return a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # This allows dictionary-like access to rows
    return conn


def init_db():
    """Initialize the database by creating the feedback table if it doesn't exist."""
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_text TEXT,
            followers INTEGER,
            followings INTEGER,
            verified BOOLEAN,
            predicted_score REAL,
            human_label INTEGER,
            timestamp TEXT,
            features TEXT
        )
    ''')
    conn.commit()
    conn.close()


def get_crowd_feedback_data():
    """Retrieve all feedback records from the database and return as a DataFrame."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()
    return df


def add_feedback(post_text, followers, followings, verified, predicted_score, human_label, features):
    """Insert a new feedback record into the database."""
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO feedback (post_text, followers, followings, verified, predicted_score, human_label, timestamp, features)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        post_text,
        followers,
        followings,
        verified,
        predicted_score,
        human_label,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        str(features)
    ))
    conn.commit()
    conn.close()


# Initialize the database when this module is imported
init_db()
