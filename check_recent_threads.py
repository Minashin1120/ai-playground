import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, '.env'))

db_url = os.getenv('DATABASE_URL')
engine = create_engine(db_url)

def check_recent_threads():
    with engine.connect() as conn:
        print("Checking recent thread updates...")
        res = conn.execute(text("SELECT id, public_id, user_id, updated_at, custom_instruction FROM thread ORDER BY updated_at DESC LIMIT 5")).fetchall()
        for row in res:
            print(row)

if __name__ == "__main__":
    check_recent_threads()
