# test_db.py
import sqlite3
from db import get_db_connection, init_db


def test_database():
    # Initialize database (creates table if not exists)
    init_db()

    # Get a connection and create a cursor
    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert a test record
    test_record = ("Test post", 100, 300, False, 0.75, 1, "2025-04-10 12:00:00", "[0.1, 0.2, 0.3]")
    cursor.execute("""
        INSERT INTO feedback (post_text, followers, followings, verified, predicted_score, human_label, timestamp, features)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, test_record)
    conn.commit()

    # Query the record back
    cursor.execute("SELECT * FROM feedback WHERE post_text = 'Test post'")
    rows = cursor.fetchall()

    # Print the rows to verify insertion
    print("Test Record(s):")
    for row in rows:
        print(dict(row))

    # Clean up: delete the test record
    cursor.execute("DELETE FROM feedback WHERE post_text = 'Test post'")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    test_database()
