import os
import json
import time
import logging
import base64
import mimetypes
import redis
import shutil
import glob
import requests
import tiktoken
import subprocess
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from rq import Queue
from datetime import datetime
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for, make_response, flash, send_from_directory, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import or_, exc, text
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import pypdf
from cryptography.fernet import Fernet

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'instance/uploads')
app.config['CHANGELOG_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static/changelogs')
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024
app.config['MAINTENANCE_MODE'] = os.path.exists(os.path.join(os.path.dirname(__file__), 'maintenance.lock'))

# Redis DB 10
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/10')
redis_conn = redis.from_url(REDIS_URL)
task_queue = Queue('ai_chat_queue', connection=redis_conn)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Encryption ---
KEY_FILE = os.path.join(os.path.dirname(__file__), 'secret.key')
cipher = None
try:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, 'rb') as kf: cipher = Fernet(kf.read().strip())
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as kf: kf.write(key)
        cipher = Fernet(key)
except Exception as e:
    logger.error(f'Encryption setup failed: {e}')

def encrypt_val(val):
    if not val or not cipher: return val
    try: return cipher.encrypt(val.encode()).decode()
    except: return val

def decrypt_val(val):
    if not val or not cipher: return val
    try: return cipher.decrypt(val.encode()).decode()
    except: return val

@app.before_request
def check_maintenance():
    if app.config['MAINTENANCE_MODE'] and request.endpoint not in ['static', 'login', 'logout']:
        if not current_user.is_authenticated or current_user.username != 'minashin1120':
            return render_template('maintenance.html'), 503

@app.after_request
def add_security_headers(response):
    csp = "default-src * 'unsafe-inline' 'unsafe-eval' data: blob: https://cdn.jsdelivr.net https://cdnjs.cloudflare.com;"
    response.headers['Content-Security-Policy'] = csp
    return response

# --- Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password_hash = db.Column(db.String(255))
    system_prompt = db.Column(db.Text, default="")
    openai_api_key = db.Column(db.Text, nullable=True)
    gemini_api_key = db.Column(db.Text, nullable=True)
    xai_api_key = db.Column(db.Text, nullable=True)
    is_setup_completed = db.Column(db.Boolean, default=False)
    enable_e2ee = db.Column(db.Boolean, default=False)
    
    threads = db.relationship('Thread', backref='user', lazy=True, cascade="all, delete-orphan")
    gems = db.relationship('Gem', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Thread(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default="New Chat")
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='thread', cascade="all, delete-orphan", lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(db.Integer, db.ForeignKey('thread.id'), nullable=False)
    role = db.Column(db.String(20))
    content = db.Column(db.Text)
    model = db.Column(db.String(50))
    image_url = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    tokens = db.Column(db.Integer, default=0)
    thought_data = db.Column(db.Text)
    is_encrypted = db.Column(db.Boolean, default=False)
    thought_signature = db.Column(db.Text, nullable=True)

class Gem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    instruction = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))

def get_key_for_user(user, name):
    user_key_field = name.lower()
    user_key = getattr(user, user_key_field, None)
    if user_key and user_key.strip():
        decrypted = decrypt_val(user_key.strip())
        if decrypted: return decrypted
    if user.username == 'minashin1120':
        sys_key = os.getenv(name)
        if sys_key and "placeholder" not in sys_key: return sys_key
    return None

def verify_turnstile(token):
    secret = os.getenv('TURNSTILE_SECRET_KEY')
    if not secret: return True
    if not token: return False
    try:
        res = requests.post('https://challenges.cloudflare.com/turnstile/v0/siteverify', data={'secret': secret, 'response': token}, timeout=5)
        return res.json().get('success', False)
    except: return False

def count_tokens(text, model="gpt-4"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except: return len(text) // 4

# --- Worker Task ---
def background_chat_task(job_id, thread_id, model_key, message_text, img_list, options, api_keys, user_id, user_config):
    with app.app_context():
        channel = f"ai_chat:channel:{job_id}"
        r = redis.from_url(REDIS_URL)
        def publish_chunk(dt, d): r.publish(channel, json.dumps({"type": dt, "data": d}))
        
        try:
            # 1. History Construction with Signatures
            all_msgs = Message.query.filter_by(thread_id=thread_id).order_by(Message.timestamp).all()
            history = []
            
            for m in all_msgs[:-1]:
                cnt = decrypt_val(m.content) if m.is_encrypted else m.content
                sig = m.thought_signature
                history.append({
                    'role': m.role, 
                    'content': cnt, 
                    'image_url': m.image_url,
                    'signature': sig
                })

            is_gemini = 'gemini' in model_key or 'nano' in model_key
            is_grok = 'grok' in model_key
            
            req_key = api_keys.get('gemini') if is_gemini else (api_keys.get('xai') if is_grok else api_keys.get('openai'))
            if not req_key: publish_chunk("error", "API Key missing."); return

            gemini_client = genai.Client(api_key=req_key, http_options={'api_version': 'v1alpha'}) if is_gemini else None
            openai_client = OpenAI(api_key=req_key) if not is_gemini and not is_grok else None
            xai_client_std = OpenAI(api_key=req_key, base_url="https://api.x.ai/v1") if is_grok else None

            # File Loading
            loaded_files = []
            for fname in img_list:
                info = {'name': fname, 'text': None, 'bytes': None, 'mime': None, 'path': os.path.join(app.config['UPLOAD_FOLDER'], fname)}
                try:
                    if os.path.exists(info['path']):
                        info['mime'] = mimetypes.guess_type(info['path'])[0] or 'application/octet-stream'
                        if fname.lower().endswith('.pdf'):
                            info['mime'] = 'application/pdf'
                            try:
                                reader = pypdf.PdfReader(info['path'])
                                extracted = ""
                                for page in reader.pages: extracted += page.extract_text() + "\n"
                                info['text'] = extracted[:50000]
                            except: pass
                        
                        is_img = fname.endswith(('.webp','.png','.jpg','.jpeg','.gif','.mp4'))
                        if not is_img and not info['text']:
                            try:
                                with open(info['path'], 'r', encoding='utf-8', errors='ignore') as f: info['text'] = f.read()
                            except: pass
                        
                        if not info['text']:
                            with open(info['path'], 'rb') as f: info['bytes'] = f.read()
                        loaded_files.append(info)
                except: pass

            full_res, thought_accumulated, generated_images = "", "", []
            final_signature = None

            if is_gemini:
                real_model = model_key
                # FIX: Logic Update for correct IDs
                if "gemini-3-flash" in model_key: real_model = "gemini-3-flash-preview"
                elif "3.0" in model_key and "pro" in model_key: real_model = "gemini-3.0-pro-preview"
                elif "2.5" in model_key: real_model = "gemini-2.5-flash"
                elif "nano-banana-pro" in model_key: real_model = "gemini-3.0-pro-image-preview"
                elif "nano-banana" in model_key: real_model = "gemini-2.5-flash-image"

                config_params = {'temperature': 0.7}
                
                if "nano" not in model_key:
                    if options.get('enable_thinking'):
                        budget = 1024
                        lvl = options.get('thinking_level', 'low')
                        if lvl == 'medium': budget = 4096
                        elif lvl == 'high': budget = 8192
                        elif lvl == 'minimal': budget = 1024
                        config_params['thinking_config'] = types.ThinkingConfig(include_thoughts=True, thinking_budget=budget)
                    
                    if options.get('enable_search'): config_params['tools'] = [types.Tool(google_search=types.GoogleSearch())]
                    if options.get('system_prompt'): config_params['system_instruction'] = options.get('system_prompt')
                    if options.get('safety_setting') == 'none':
                        config_params['safety_settings'] = [
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
                        ]

                # --- Construct History with Signatures ---
                contents = []
                for m in history:
                    parts = []
                    sig_bytes = None
                    if m.get('signature'):
                        try:
                            sig_bytes = base64.b64decode(m['signature'])
                        except: pass

                    if m['content']:
                        parts.append(types.Part(text=m['content']))

                    if m['image_url']:
                        try:
                            for h_img in json.loads(m['image_url']):
                                h_path = os.path.join(app.config['UPLOAD_FOLDER'], h_img)
                                if os.path.exists(h_path): 
                                    mime_type = mimetypes.guess_type(h_path)[0] or 'image/webp'
                                    with open(h_path, 'rb') as f: 
                                        p = types.Part.from_bytes(data=f.read(), mime_type=mime_type)
                                        if sig_bytes and m['role'] == 'assistant':
                                            p.thought_signature = sig_bytes
                                        parts.append(p)
                        except: pass
                    
                    if not parts and sig_bytes and m['role'] == 'assistant':
                         p = types.Part(text="")
                         p.thought_signature = sig_bytes
                         parts.append(p)

                    if parts:
                        contents.append(types.Content(role='model' if m['role'] == 'assistant' else 'user', parts=parts))

                curr_parts = [types.Part(text=message_text)]
                for fi in loaded_files:
                    if fi['text']: curr_parts.append(types.Part(text=f"\n\nFile: {fi['name']}\n{fi['text']}"))
                    elif fi['bytes']: curr_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime'] if fi['mime']!='application/octet-stream' else 'image/webp'))
                contents.append(types.Content(role='user', parts=curr_parts))

                stream = gemini_client.models.generate_content_stream(model=real_model, contents=contents, config=types.GenerateContentConfig(**config_params))
                
                for chunk in stream:
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        for candidate in chunk.candidates:
                            if hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'thought_signature') and part.thought_signature:
                                        final_signature = base64.b64encode(part.thought_signature).decode('utf-8')

                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        try:
                                            user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                                            if not os.path.exists(user_dir): os.makedirs(user_dir, exist_ok=True)
                                            fn = f"gen_{int(time.time())}_{len(generated_images)}.png"
                                            Image.open(BytesIO(part.inline_data.data)).save(os.path.join(user_dir, fn))
                                            db_path = f"{user_id}/{fn}"
                                            generated_images.append(db_path)
                                            img_md = f"\n\n![Generated Image](/files/{db_path})\n"
                                            full_res += img_md
                                            publish_chunk("content", img_md)
                                        except: pass
                                    
                                    tt = part.thought if hasattr(part, 'thought') and part.thought else None
                                    if tt: 
                                        thought_accumulated += (tt if isinstance(tt, str) else "")
                                        publish_chunk("thought", tt if isinstance(tt, str) else "")
                                    
                                    if part.text: 
                                        full_res += part.text
                                        publish_chunk("content", part.text)
            else:
                client = xai_client_std if is_grok else openai_client
                msgs = []
                if options.get('system_prompt'): msgs.append({"role": "system", "content": options.get('system_prompt')})
                for m in history: msgs.append({"role": m['role'], "content": m['content']})
                
                content_list = [{"type": "text", "text": message_text}]
                for fi in loaded_files:
                    if fi['text']: content_list[0]['text'] += f"\n\n[File: {fi['name']}]\n{fi['text']}"
                    elif fi['mime'].startswith('image/'):
                        content_list.append({"type": "image_url", "image_url": {"url": f"data:{fi['mime']};base64,{base64.b64encode(fi['bytes']).decode('utf-8')}"}})
                msgs.append({"role": "user", "content": content_list})
                
                kwargs = {"model": model_key, "messages": msgs, "stream": True}
                if is_grok and options.get('enable_search'): kwargs["extra_body"] = {"search_parameters": {"mode": "on"}}
                if options.get('reasoning_effort') and not is_grok: kwargs['reasoning_effort'] = options.get('reasoning_effort')
                
                stream = client.chat.completions.create(**kwargs)
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        thought_accumulated += delta.reasoning_content
                        publish_chunk("thought", delta.reasoning_content)
                    if delta.content:
                        full_res += delta.content
                        publish_chunk("content", delta.content)

            final_content = full_res
            final_thought = json.dumps({'text': thought_accumulated}) if thought_accumulated else None
            is_enc = user_config.get('enable_e2ee', False)
            
            if is_enc:
                final_content = encrypt_val(final_content)
                if final_thought: final_thought = encrypt_val(final_thought)

            msg_entry = Message(
                thread_id=thread_id, 
                role='assistant', 
                content=final_content, 
                model=model_key, 
                image_url=json.dumps(generated_images) if generated_images else None, 
                thought_data=final_thought,
                tokens=count_tokens(full_res),
                is_encrypted=is_enc,
                thought_signature=final_signature
            )
            db.session.add(msg_entry)
            Thread.query.get(thread_id).updated_at = datetime.utcnow()
            db.session.commit()
            publish_chunk("done", "OK")

        except Exception as e:
            logger.error(f"Worker Error: {e}")
            publish_chunk("error", str(e))

# ... [Routes unchanged] ...
@app.route('/')
def index():
    if current_user.is_authenticated:
        if not current_user.is_setup_completed: return redirect(url_for('setup'))
        return render_template('chat.html')
    return render_template('landing.html')

@app.route('/files/<path:filename>')
@login_required
def serve_file(filename):
    parts = filename.split('/')
    if len(parts) > 1 and str(parts[0]) != str(current_user.id):
        abort(403)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/changelog')
def changelog():
    log_dir = app.config['CHANGELOG_FOLDER']
    logs = []
    if os.path.exists(log_dir):
        files = sorted(glob.glob(os.path.join(log_dir, '*.md')), reverse=True)
        for f in files:
            with open(f, 'r', encoding='utf-8') as file: logs.append({'content': file.read()})
    return render_template('changelog.html', logs=logs)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        if not verify_turnstile(request.form.get('cf-turnstile-response')): return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Auth Error")
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and user.check_password(request.form.get('password')):
            login_user(user, remember=True)
            return redirect(url_for('index'))
        return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Invalid credentials")
    return render_template('login.html', site_key=os.getenv('TURNSTILE_SITE_KEY'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        if not verify_turnstile(request.form.get('cf-turnstile-response')): return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Auth Error")
        if User.query.filter_by(username=request.form.get('username')).first(): return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'), error="Username taken")
        new_user = User(username=request.form.get('username'), is_setup_completed=False)
        new_user.set_password(request.form.get('password'))
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('setup'))
    return render_template('signup.html', site_key=os.getenv('TURNSTILE_SITE_KEY'))

@app.route('/setup', methods=['GET', 'POST'])
@login_required
def setup():
    if current_user.is_setup_completed: return redirect(url_for('index'))
    if request.method == 'POST':
        current_user.openai_api_key = encrypt_val(request.form.get('openai_key'))
        current_user.gemini_api_key = encrypt_val(request.form.get('gemini_key'))
        current_user.xai_api_key = encrypt_val(request.form.get('xai_key'))
        current_user.is_setup_completed = True
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('setup.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/api/account/delete', methods=['POST'])
@login_required
def delete_account():
    try:
        shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id)), ignore_errors=True)
        db.session.delete(current_user)
        db.session.commit()
        logout_user()
        return jsonify({'status': 'ok'})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def handle_settings():
    if request.method == 'GET':
        return jsonify({
            'system_prompt': current_user.system_prompt or "",
            'username': current_user.username, 
            'openai_key': decrypt_val(current_user.openai_api_key) or "", 
            'gemini_key': decrypt_val(current_user.gemini_api_key) or "", 
            'xai_key': decrypt_val(current_user.xai_api_key) or "",
            'enable_e2ee': current_user.enable_e2ee
        })
    d = request.json
    if 'system_prompt' in d: current_user.system_prompt = d['system_prompt']
    if 'openai_key' in d: current_user.openai_api_key = encrypt_val(d['openai_key'])
    if 'gemini_key' in d: current_user.gemini_api_key = encrypt_val(d['gemini_key'])
    if 'xai_key' in d: current_user.xai_api_key = encrypt_val(d['xai_key'])
    if d.get('new_password'): current_user.set_password(d['new_password'])
    if d.get('new_username') and d['new_username'] != current_user.username:
        if not User.query.filter_by(username=d['new_username']).first(): current_user.username = d['new_username']
    
    if 'enable_e2ee' in d and d['enable_e2ee'] != current_user.enable_e2ee:
        current_user.enable_e2ee = d['enable_e2ee']
        msgs = Message.query.join(Thread).filter(Thread.user_id == current_user.id).all()
        for m in msgs:
            if current_user.enable_e2ee and not m.is_encrypted:
                m.content = encrypt_val(m.content)
                if m.thought_data: m.thought_data = encrypt_val(m.thought_data)
                m.is_encrypted = True
            elif not current_user.enable_e2ee and m.is_encrypted:
                m.content = decrypt_val(m.content)
                if m.thought_data: m.thought_data = decrypt_val(m.thought_data)
                m.is_encrypted = False

    db.session.commit()
    flash("設定を保存しました")
    return jsonify({'status': 'ok'})

@app.route('/api/gems', methods=['GET', 'POST'])
@login_required
def handle_gems():
    if request.method == 'GET':
        gems = Gem.query.filter_by(user_id=current_user.id).order_by(Gem.created_at.desc()).all()
        return jsonify([{'id': g.id, 'name': g.name, 'description': g.description, 'instruction': g.instruction} for g in gems])
    d = request.json
    gem = Gem(user_id=current_user.id, name=d.get('name', 'My Gem'), description=d.get('description', ''), instruction=d.get('instruction', ''))
    db.session.add(gem)
    db.session.commit()
    return jsonify({'id': gem.id, 'name': gem.name})

@app.route('/api/gems/<int:gid>', methods=['DELETE'])
@login_required
def delete_gem(gid):
    gem = Gem.query.get_or_404(gid)
    if gem.user_id != current_user.id: return jsonify({'error': '403'}), 403
    db.session.delete(gem)
    db.session.commit()
    return jsonify({'status': 'deleted'})

@app.route('/api/threads', methods=['GET', 'POST'])
@login_required
def handle_threads():
    if request.method == 'GET':
        q = request.args.get('q', '').strip()
        query = Thread.query.filter_by(user_id=current_user.id)
        if q: 
            if current_user.enable_e2ee:
                 query = query.filter(Thread.title.contains(q))
            else:
                 query = query.join(Message).filter(or_(Thread.title.contains(q), Message.content.contains(q))).distinct()
        ts = query.order_by(Thread.updated_at.desc()).limit(50).all()
        return jsonify([{'id': t.id, 'title': t.title} for t in ts])
    t = Thread(user_id=current_user.id)
    db.session.add(t)
    db.session.commit()
    return jsonify({'id': t.id, 'title': t.title})

@app.route('/api/threads/<int:tid>', methods=['GET', 'DELETE'])
@login_required
def handle_thread_item(tid):
    t = Thread.query.get_or_404(tid)
    if t.user_id != current_user.id: return jsonify({'error': '403'}), 403
    if request.method == 'GET':
        ms = Message.query.filter_by(thread_id=tid).order_by(Message.timestamp).all()
        res = []
        for m in ms:
            cnt = decrypt_val(m.content) if m.is_encrypted else m.content
            tht = decrypt_val(m.thought_data) if (m.is_encrypted and m.thought_data) else m.thought_data
            res.append({'id': m.id, 'role': m.role, 'content': cnt, 'image_url': m.image_url, 'model': m.model, 'thought_data': tht})
        return jsonify(res)
    db.session.delete(t)
    db.session.commit()
    return jsonify({'status': 'deleted'})

@app.route('/api/threads/<int:tid>/title', methods=['PUT'])
@login_required
def update_title(tid):
    t = Thread.query.get_or_404(tid)
    if t.user_id != current_user.id: return jsonify({'error': '403'}), 403
    t.title = request.json.get('title', 'Untitled')
    db.session.commit()
    return jsonify({'status': 'ok'})

@app.route('/api/messages/<int:mid>', methods=['DELETE'])
@login_required
def delete_message(mid):
    msg = Message.query.get_or_404(mid)
    if msg.thread.user_id != current_user.id: return jsonify({'error': '403'}), 403
    Message.query.filter(Message.thread_id == msg.thread_id, Message.timestamp >= msg.timestamp).delete()
    db.session.commit()
    return jsonify({'status': 'ok'})

@app.route('/api/files', methods=['GET'])
@login_required
def get_files_lib():
    try:
        msgs = Message.query.join(Thread).filter(Thread.user_id == current_user.id, Message.image_url != None).order_by(Message.timestamp.desc()).all()
        files = []
        seen = set()
        for m in msgs:
            if not m.image_url: continue
            try:
                l = json.loads(m.image_url)
                if not isinstance(l, list): l = [m.image_url]
            except: l = [m.image_url]
            for p in l:
                if p and p not in seen:
                    fp = os.path.join(app.config['UPLOAD_FOLDER'], p)
                    if os.path.exists(fp):
                        seen.add(p)
                        ext = os.path.splitext(p)[1].lower().replace('.', '')
                        files.append({'filename': os.path.basename(p), 'filepath': p, 'url': url_for('serve_file', filename=p), 'type': 'image' if ext in ['png','jpg','webp'] else 'file', 'ext': ext})
        return jsonify(files)
    except: return jsonify([])

@app.route('/api/files/delete', methods=['POST'])
@login_required
def delete_files_batch():
    for f in request.json.get('filenames', []):
        if f.startswith(f"{current_user.id}/") and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], f)):
            try: os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
            except: pass
    return jsonify({'status': 'ok'})

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    files = request.files.getlist('file')
    if not files: return jsonify({'error': 'No file'}), 400
    ud = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    
    if not os.path.exists(ud):
        os.makedirs(ud, exist_ok=True)
        os.chmod(ud, 0o777)
    else:
        try: os.chmod(ud, 0o777)
        except: pass

    res = []
    for f in files:
        if f.filename:
            orig_name = secure_filename(f.filename)
            ext = os.path.splitext(orig_name)[1].lower()
            fname_base = f"{int(time.time())}_{os.urandom(4).hex()}"
            fname = f"{fname_base}{ext}"
            save_path = os.path.join(ud, fname)
            
            is_image = ext in ['.jpg', '.jpeg', '.png']
            if is_image and not orig_name.endswith('.webp'):
                try:
                    Image.open(f).convert('RGB').save(os.path.join(ud, f"{fname_base}.webp"), 'WEBP', quality=80)
                    fname = f"{fname_base}.webp"
                    res.append(f"{current_user.id}/{fname}")
                except:
                    f.seek(0)
                    f.save(save_path)
                    res.append(f"{current_user.id}/{fname}")
            else:
                f.save(save_path)
                res.append(f"{current_user.id}/{fname}")
                
    return jsonify({'filename': res[0] if res else '', 'filenames': res})

@app.route('/api/debug/log', methods=['GET'])
@login_required
def debug_log():
    if current_user.username != 'minashin1120': return abort(403)
    def generate():
        process = subprocess.Popen(['sudo', 'journalctl', '-u', 'ai-chat', '-n', '50', '--no-pager'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, _ = process.communicate()
        yield stdout
    return Response(generate(), mimetype='text/plain')

@app.route('/api/maintenance', methods=['POST'])
@login_required
def toggle_maintenance():
    if current_user.username != 'minashin1120': return abort(403)
    lock_file = os.path.join(os.path.dirname(__file__), 'maintenance.lock')
    if request.json.get('enabled'):
        with open(lock_file, 'w') as f: f.write('locked')
        app.config['MAINTENANCE_MODE'] = True
    else:
        if os.path.exists(lock_file): os.remove(lock_file)
        app.config['MAINTENANCE_MODE'] = False
    return jsonify({'status': 'ok', 'mode': app.config['MAINTENANCE_MODE']})

@app.route('/chat_stream', methods=['POST'])
@login_required
def chat_stream():
    import uuid
    data = request.json
    job_id = str(uuid.uuid4())
    sys_prompt = data.get('system_prompt') 
    if not sys_prompt and data.get('enable_system_prompt'): sys_prompt = current_user.system_prompt
    
    options = {
        'enable_search': data.get('enable_search', False), 
        'enable_thinking': data.get('enable_thinking', False), 
        'thinking_level': data.get('thinking_level', 'low'),
        'reasoning_effort': data.get('reasoning_effort', 'medium'), 
        'system_prompt': sys_prompt,
        'safety_setting': data.get('safety_setting', 'default')
    }
    
    api_keys = {
        'openai': get_key_for_user(current_user, 'OPENAI_API_KEY'),
        'gemini': get_key_for_user(current_user, 'GEMINI_API_KEY'),
        'xai': get_key_for_user(current_user, 'XAI_API_KEY')
    }
    
    user_config = {'enable_e2ee': current_user.enable_e2ee}

    u_msg = data.get('message')
    if current_user.enable_e2ee:
        u_msg_enc = encrypt_val(u_msg)
        msg_entry = Message(
            thread_id=data.get('thread_id'), role='user', content=u_msg_enc, 
            image_url=json.dumps(data.get('image_urls', [])), is_encrypted=True
        )
    else:
        msg_entry = Message(
            thread_id=data.get('thread_id'), role='user', content=u_msg, 
            image_url=json.dumps(data.get('image_urls', [])), is_encrypted=False
        )
    
    db.session.add(msg_entry)
    db.session.commit()

    task_queue.enqueue(background_chat_task, job_id, data.get('thread_id'), data.get('model'), u_msg, data.get('image_urls', []), options, api_keys, current_user.id, user_config, job_timeout=600)
    
    def generate():
        pubsub = redis_conn.pubsub()
        pubsub.subscribe(f"ai_chat:channel:{job_id}")
        st = time.time()
        try:
            for m in pubsub.listen():
                if time.time() - st > 600: yield json.dumps({"type": "error", "data": "Timeout"}) + "\n"; break
                if m['type'] == 'message':
                    yield m['data'].decode('utf-8') + "\n"
                    if json.loads(m['data'].decode('utf-8')).get('type') in ['done', 'error']: break
        finally: pubsub.close()
    return Response(stream_with_context(generate()), mimetype='application/json')

with app.app_context():
    db.create_all()
    try:
        with db.engine.connect() as conn:
            conn.execute(text("ALTER TABLE message ADD COLUMN thought_signature TEXT"))
    except: pass

if __name__ == '__main__':
    app.run(debug=True)
