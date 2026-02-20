import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, '.env'))

db_url = os.getenv('DATABASE_URL')
engine = create_engine(db_url)

def check_locks():
    with engine.connect() as conn:
        print("--- Process List ---")
        try:
            res = conn.execute(text("SHOW FULL PROCESSLIST")).fetchall()
            for row in res:
                print(row)
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n--- InnoDB Status ---")
        try:
            res = conn.execute(text("SHOW ENGINE INNODB STATUS")).fetchall()
            for row in res:
                print(row[2])
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_locks()