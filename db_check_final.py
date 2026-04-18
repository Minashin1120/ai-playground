import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from cryptography.fernet import Fernet

base_dir = '/home/ai-chat-minashin1120/app'
load_dotenv(os.path.join(base_dir, '.env'))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    system_prompt = db.Column(db.Text)
    system_prompt_enabled = db.Column(db.Boolean)
    enable_e2ee = db.Column(db.Boolean)

KEY_FILE = os.path.join(base_dir, 'secret.key')
cipher = None
if os.path.exists(KEY_FILE):
    with open(KEY_FILE, 'rb') as kf:
        cipher = Fernet(kf.read().strip())

def decrypt_val(val):
    if not val or not cipher: return val
    try: return cipher.decrypt(val.encode()).decode()
    except Exception as e: return f"[Decryption Error: {e}]"

with app.app_context():
    u = User.query.filter_by(username='minashin1120').first()
    if u:
        print(f"--- DB State for {u.username} ---")
        print(f"ID: {u.id}")
        print(f"E2EE: {u.enable_e2ee}")
        print(f"SP Enabled: {u.system_prompt_enabled}")
        print(f"Raw SP: {u.system_prompt}")
        print(f"Decrypted SP: {decrypt_val(u.system_prompt)}")
    else:
        print("User minashin1120 not found")
