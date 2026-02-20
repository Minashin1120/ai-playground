import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, '.env'))

db_url = os.getenv('DATABASE_URL')
engine = create_engine(db_url)

def check_db_health():
    with engine.connect() as conn:
        print("--- Testing connection ---")
        try:
            res = conn.execute(text("SELECT 1")).fetchone()
            print("Connection OK")
        except Exception as e:
            print("Connection failed: " + str(e))
            return

        print("--- Checking for long running queries ---")
        try:
            res = conn.execute(text("SHOW FULL PROCESSLIST")).fetchall()
            for row in res:
                if row[4] != 'Sleep':
                    print(str(row))
        except Exception as e:
            print("Failed to show processlist: " + str(e))

        print("--- Checking Thread table count ---")
        try:
            res = conn.execute(text("SELECT COUNT(*) FROM thread")).fetchone()
            print("Total threads: " + str(res[0]))
        except Exception as e:
            print("Failed to count threads: " + str(e))

if __name__ == "__main__":
    check_db_health()