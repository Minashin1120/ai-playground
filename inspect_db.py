import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, '.env'))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
db = SQLAlchemy(app)

def check_db():
    with app.app_context():
        for table in ['thread', 'message', 'user']:
            print(f"--- {table} table ---")
            try:
                res = db.session.execute(text(f"DESCRIBE {table}")).fetchall()
                for row in res:
                    print(row)
            except Exception as e:
                print(f"Error: {e}")
            print("")

if __name__ == "__main__":
    check_db()