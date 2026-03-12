import os
import sys
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from dotenv import load_dotenv

# アプリケーションのディレクトリ構造に合わせる
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, '.env'))

app = Flask(__name__)
# 修正: DATABASE_URLのプレフィックスをmysql+pymysqlに固定
db_url = os.getenv('DATABASE_URL')
if db_url and db_url.startswith('mysql://'):
    db_url = db_url.replace('mysql://', 'mysql+pymysql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
db = SQLAlchemy(app)

def fix_db():
    with app.app_context():
        # default_enable_url_context
        result = db.session.execute(text("SHOW COLUMNS FROM user LIKE 'default_enable_url_context'")).fetchone()
        if not result:
            print("Column 'default_enable_url_context' not found. Adding it...")
            try:
                db.session.execute(text("ALTER TABLE user ADD COLUMN default_enable_url_context BOOLEAN DEFAULT FALSE AFTER default_enable_search"))
                db.session.commit()
                print("Successfully added 'default_enable_url_context' column.")
            except Exception as e:
                print(f"Error adding column: {e}")
                db.session.rollback()
        else:
            print("Column 'default_enable_url_context' already exists.")

        # last_enable_url_context
        result = db.session.execute(text("SHOW COLUMNS FROM user LIKE 'last_enable_url_context'")).fetchone()
        if not result:
            print("Column 'last_enable_url_context' not found. Adding it...")
            try:
                db.session.execute(text("ALTER TABLE user ADD COLUMN last_enable_url_context BOOLEAN DEFAULT FALSE AFTER last_enable_search"))
                db.session.commit()
                print("Successfully added 'last_enable_url_context' column.")
            except Exception as e:
                print(f"Error adding column: {e}")
                db.session.rollback()
        else:
            print("Column 'last_enable_url_context' already exists.")

if __name__ == "__main__":
    fix_db()
