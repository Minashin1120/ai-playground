import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, '.env'))

db_url = os.getenv('DATABASE_URL')
engine = create_engine(db_url)

def check_threads():
    with engine.connect() as conn:
        print("Checking for threads with NULL title...")
        res = conn.execute(text("SELECT id, user_id, title FROM thread WHERE title IS NULL")).fetchall()
        print(f"Found {len(res)} threads with NULL title.")
        for row in res:
            print(row)

if __name__ == "__main__":
    check_threads()
