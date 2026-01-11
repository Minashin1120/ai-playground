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
import random
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from rq import Queue
from datetime import datetime
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for, make_response, flash, send_file, abort
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
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True, 'pool_recycle': 280}
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'instance/uploads')
app.config['CHANGELOG_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static/changelogs')
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024
app.config['MAINTENANCE_MODE'] = os.path.exists(os.path.join(os.path.dirname(__file__), 'maintenance.lock'))

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/10')
redis_conn = redis.from_url(REDIS_URL)
task_queue = Queue('ai_chat_queue', connection=redis_conn)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

KEY_FILE = os.path.join(os.path.dirname(__file__), 'secret.key')
cipher = None
try:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, 'rb') as kf:
            cipher = Fernet(kf.read().strip())
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as kf:
            kf.write(key)
        cipher = Fernet(key)
except Exception as e:
    logger.error(f'Encryption setup failed: {e}')

def encrypt_val(val):
    if not val or not cipher:
        return val
    try:
        return cipher.encrypt(val.encode()).decode()
    except:
        return val

def decrypt_val(val):
    if not val or not cipher:
        return val
    try:
        return cipher.decrypt(val.encode()).decode()
    except:
        return val

def encrypt_bytes(data):
    if not cipher:
        return data
    return cipher.encrypt(data)

def decrypt_bytes(data):
    if not cipher:
        return data
    return cipher.decrypt(data)

def secure_delete(path):
    if os.path.exists(path):
        try:
            length = os.path.getsize(path)
            with open(path, "wb") as f:
                f.write(os.urandom(length))
            os.remove(path)
        except Exception as e:
            logger.error(f"Secure delete failed for {path}: {e}")

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
    user_key = getattr(user, name.lower(), None)
    if user_key and user_key.strip():
        dec = decrypt_val(user_key.strip())
        if dec:
            return dec
    if user.username == 'minashin1120':
        sys_key = os.getenv(name)
        if sys_key and "placeholder" not in sys_key:
            return sys_key
    return None

def verify_turnstile(token):
    secret = os.getenv('TURNSTILE_SECRET_KEY')
    if not secret:
        return True
    if not token:
        return False
    try:
        r = requests.post('https://challenges.cloudflare.com/turnstile/v0/siteverify', data={'secret': secret, 'response': token}, timeout=5)
        return r.json().get('success', False)
    except:
        return False

def count_tokens(text, model="gpt-4"):
    return len(text) // 4

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(exc.SQLAlchemyError))
def safe_db_commit():
    db.session.commit()

def migrate_e2ee_task(user_id, enable):
    with app.app_context():
        redis_conn.set(f"migration_status:{user_id}", "processing")
        try:
            user = User.query.get(user_id)
            if enable and user.system_prompt and not user.enable_e2ee:
                user.system_prompt = encrypt_val(user.system_prompt)
            elif not enable and user.system_prompt and user.enable_e2ee:
                user.system_prompt = decrypt_val(user.system_prompt)

            msgs = Message.query.join(Thread).filter(Thread.user_id == user_id).all()
            for m in msgs:
                if enable and not m.is_encrypted:
                    m.content = encrypt_val(m.content)
                    if m.thought_data:
                        m.thought_data = encrypt_val(m.thought_data)
                    m.is_encrypted = True
                elif not enable and m.is_encrypted:
                    m.content = decrypt_val(m.content)
                    if m.thought_data:
                        m.thought_data = decrypt_val(m.thought_data)
                    m.is_encrypted = False
            
            user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
            if os.path.exists(user_dir):
                for f in os.listdir(user_dir):
                    fp = os.path.join(user_dir, f)
                    if enable and not f.endswith('.enc'):
                        with open(fp, 'rb') as file:
                            data = file.read()
                        with open(fp + '.enc', 'wb') as file:
                            file.write(encrypt_bytes(data))
                        secure_delete(fp)
                    elif not enable and f.endswith('.enc'):
                        with open(fp, 'rb') as file:
                            data = file.read()
                        with open(fp[:-4], 'wb') as file:
                            file.write(decrypt_bytes(data))
                        secure_delete(fp)
            
            user.enable_e2ee = enable
            safe_db_commit()
            redis_conn.set(f"migration_status:{user_id}", "done")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            redis_conn.set(f"migration_status:{user_id}", "error")


def background_chat_task(job_id, thread_id, model_key, message_text, img_list, options, api_keys, user_id, user_config):
    with app.app_context():
        # DB接続リセット (念のため)
        db.engine.dispose()
        
        channel = f"ai_chat:channel:{job_id}"
        r = redis.from_url(REDIS_URL)
        def pub(dt, d):
            r.publish(channel, json.dumps({"type": dt, "data": d}))
        
        try:
            all_msgs = Message.query.filter_by(thread_id=thread_id).order_by(Message.timestamp).all()
            history = []
            for m in all_msgs[:-1]:
                cnt = decrypt_val(m.content) if m.is_encrypted else m.content
                sig = m.thought_signature
                history.append({'role': m.role, 'content': cnt, 'image_url': m.image_url, 'signature': sig})

            is_gem = 'gemini' in model_key or 'nano' in model_key
            is_grok = 'grok' in model_key
            
            # API Key Selection
            key = None
            if is_gem: key = api_keys.get('gemini')
            elif is_grok: key = api_keys.get('xai')
            else: key = api_keys.get('openai')
            
            if not key:
                pub("error", "API Key missing")
                return

            # Client Initialization
            g_client = None
            o_client = None
            x_client = None

            if is_gem:
                g_client = genai.Client(api_key=key, http_options={'api_version': 'v1alpha'})
            elif is_grok:
                # xAI Native SDK for Search
                from xai_sdk import Client as XAIClient
                x_client = XAIClient(api_key=key)
            else:
                o_client = OpenAI(api_key=key)

            # File Loading Logic
            loaded_files = []
            for fn in img_list:
                bp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
                ep = bp + '.enc'
                data = None
                mime = mimetypes.guess_type(bp)[0] or 'application/octet-stream'
                try:
                    if os.path.exists(bp):
                        with open(bp, 'rb') as f: data = f.read()
                    elif os.path.exists(ep):
                        with open(ep, 'rb') as f: data = decrypt_bytes(f.read())
                    if data:
                        if fn.lower().endswith('.pdf'):
                            reader = pypdf.PdfReader(BytesIO(data))
                            extracted = "".join([p.extract_text() + "\n" for p in reader.pages])
                            loaded_files.append({'name': fn, 'text': extracted[:50000], 'bytes': None, 'mime': 'application/pdf'})
                        elif fn.endswith(('.webp','.png','.jpg','.jpeg','.gif','.mp4')):
                            loaded_files.append({'name': fn, 'text': None, 'bytes': data, 'mime': mime})
                        else:
                            try: loaded_files.append({'name': fn, 'text': data.decode('utf-8'), 'bytes': None, 'mime': mime})
                            except: loaded_files.append({'name': fn, 'text': None, 'bytes': data, 'mime': mime})
                except: pass

            full_res, thought_accumulated, generated_images, final_signature = "", "", [], None

            # --- Gemini Processing ---
            if is_gem:
                rm = model_key
                if "gemini-3-flash" in model_key: rm = "gemini-3-flash-preview"
                elif "3.0" in model_key and "pro" in model_key: rm = "gemini-3.0-pro-preview"
                elif "2.5" in model_key: rm = "gemini-2.5-flash"
                elif "nano-banana-pro" in model_key: rm = "gemini-3.0-pro-image-preview"
                elif "nano-banana" in model_key: rm = "gemini-2.5-flash-image"

                conf = {'temperature': 0.7}
                if "nano" not in model_key:
                    if options.get('enable_thinking'):
                        b = 1024
                        if options.get('thinking_level') == 'medium': b = 4096
                        elif options.get('thinking_level') == 'high': b = 8192
                        conf['thinking_config'] = types.ThinkingConfig(include_thoughts=True, thinking_budget=b)
                    if options.get('enable_search'):
                        conf['tools'] = [types.Tool(google_search=types.GoogleSearch())]
                    if options.get('system_prompt'):
                        conf['system_instruction'] = options.get('system_prompt')
                    
                contents = []
                for m in history:
                    parts, s_bytes = [], None
                    if m.get('signature'):
                        try: s_bytes = base64.b64decode(m['signature'])
                        except: pass
                    if m['content']: parts.append(types.Part(text=m['content']))
                    # Images... (省略: 元のロジックと同じだが簡略化のため省略せず記述が必要)
                    # ここでは既存ロジックを維持
                    if m['image_url']:
                         try:
                            for h_img in json.loads(m['image_url']):
                                bp2 = os.path.join(app.config['UPLOAD_FOLDER'], h_img)
                                ep2 = bp2 + '.enc'
                                d2 = None
                                if os.path.exists(bp2):
                                    with open(bp2, 'rb') as f: d2 = f.read()
                                elif os.path.exists(ep2):
                                    with open(ep2, 'rb') as f: d2 = decrypt_bytes(f.read())
                                if d2:
                                    p = types.Part.from_bytes(data=d2, mime_type=mimetypes.guess_type(bp2)[0] or 'image/webp')
                                    if s_bytes and m['role'] == 'assistant': p.thought_signature = s_bytes
                                    parts.append(p)
                         except: pass

                    if not parts and s_bytes and m['role'] == 'assistant':
                         p = types.Part(text="")
                         p.thought_signature = s_bytes
                         parts.append(p)
                    if parts: contents.append(types.Content(role='model' if m['role'] == 'assistant' else 'user', parts=parts))

                curr = [types.Part(text=message_text)]
                for fi in loaded_files:
                    if fi['text']: curr.append(types.Part(text=f"\n\nFile: {fi['name']}\n{fi['text']}"))
                    elif fi['bytes']: curr.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime']))
                contents.append(types.Content(role='user', parts=curr))

                stream = g_client.models.generate_content_stream(model=rm, contents=contents, config=types.GenerateContentConfig(**conf))
                
                for chunk in stream:
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        for cand in chunk.candidates:
                            for part in cand.content.parts:
                                if hasattr(part, 'thought_signature') and part.thought_signature:
                                    final_signature = base64.b64encode(part.thought_signature).decode('utf-8')
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    # Image gen logic (simplified)
                                    pass 
                                is_th = False
                                if hasattr(part, 'thought') and part.thought is True: is_th = True
                                if is_th:
                                    if part.text:
                                        thought_accumulated += part.text
                                        pub("thought", part.text)
                                else:
                                    if part.text:
                                        full_res += part.text
                                        pub("content", part.text)
            
            # --- xAI (Grok) Processing ---
            elif is_grok:
                from xai_sdk.chat import user as x_user, assistant as x_assistant, system as x_system
                from xai_sdk.tools import web_search
                
                chat_session = x_client.chat.create(model=model_key)
                
                # System Prompt
                if options.get('system_prompt'):
                    chat_session.append(x_system(options.get('system_prompt')))
                
                # History
                for m in history:
                    if m['role'] == 'user': chat_session.append(x_user(m['content']))
                    else: chat_session.append(x_assistant(m['content']))
                
                # Current Message & Tools
                tools_list = []
                if options.get('enable_search'):
                    tools_list.append(web_search())
                
                # Set tools if any
                if tools_list:
                    # SDK仕様に合わせてtoolsを設定 (create時ではなくメソッドで制御が必要な場合もあるが、ここでは簡略化)
                    # SDK 1.3+ では create に tools 引数があるか、stream時に有効化
                    pass 

                # メッセージ追加
                chat_session.append(x_user(message_text))
                
                # 生成 (Stream)
                # Note: xAI SDKの仕様に合わせて修正
                # tools引数は create メソッドに渡すのが一般的
                chat_session = x_client.chat.create(model=model_key, tools=tools_list if tools_list else None)
                
                if options.get('system_prompt'): chat_session.append(x_system(options.get('system_prompt')))
                for m in history:
                     if m['role']=='user': chat_session.append(x_user(m['content']))
                     else: chat_session.append(x_assistant(m['content']))
                chat_session.append(x_user(message_text))

                stream = chat_session.stream()
                
                for chunk in stream:
                    # xAI SDK chunk handling
                    if chunk.content:
                        full_res += chunk.content
                        pub("content", chunk.content)
                    # Reasoning or Tool output handling if needed
            
            # --- OpenAI Processing ---
            else:
                msgs = []
                if options.get('system_prompt'):
                    msgs.append({"role": "system", "content": options.get('system_prompt')})
                for m in history:
                    msgs.append({"role": m['role'], "content": m['content']})
                
                content_list = [{"type": "text", "text": message_text}]
                for fi in loaded_files:
                    if fi['text']:
                        content_list[0]['text'] += f"\n\n[File: {fi['name']}]\n{fi['text']}"
                    elif fi['mime'].startswith('image/'):
                        content_list.append({"type": "image_url", "image_url": {"url": f"data:{fi['mime']};base64,{base64.b64encode(fi['bytes']).decode('utf-8')}"}})
                msgs.append({"role": "user", "content": content_list})
                
                kwargs = {"model": model_key, "messages": msgs, "stream": True}
                
                # OpenAIには検索パラメータを送らない (エラー回避)
                # if options.get('enable_search'): ... (削除)
                
                if options.get('reasoning_effort'):
                    kwargs['reasoning_effort'] = options.get('reasoning_effort')
                
                kwargs['max_completion_tokens'] = 4096

                stream = o_client.chat.completions.create(**kwargs)
                for chunk in stream:
                    if not chunk.choices: continue
                    delta = chunk.choices[0].delta
                    
                    r_content = getattr(delta, 'reasoning_content', None)
                    if r_content is None and hasattr(delta, 'model_extra') and delta.model_extra:
                        r_content = (delta.model_extra.get('reasoning_content') or delta.model_extra.get('thinking') or delta.model_extra.get('reasoning'))
                    
                    if r_content:
                        thought_accumulated += r_content
                        pub("thought", r_content)
                    
                    if delta.content:
                        full_res += delta.content
                        pub("content", delta.content)

            # Finalize (共通)
            final_content = full_res
            final_thought = json.dumps({'text': thought_accumulated}) if thought_accumulated else None
            is_enc = user_config.get('enable_e2ee', False)
            
            if is_enc:
                final_content = encrypt_val(final_content)
                if final_thought:
                    final_thought = encrypt_val(final_thought)
            
            msg_entry = Message(
                thread_id=thread_id, role='assistant', content=final_content, 
                model=model_key, image_url=json.dumps(generated_images) if generated_images else None, 
                thought_data=final_thought, tokens=count_tokens(full_res), 
                is_encrypted=is_enc, thought_signature=final_signature
            )
            db.session.add(msg_entry)
            Thread.query.get(thread_id).updated_at = datetime.utcnow()
            safe_db_commit()
            pub("done", "OK")

        except Exception as e:
            logger.error(f"Worker Error: {e}")
            pub("error", str(e))


@app@app.route('/')
def index():
    if current_user.is_authenticated:
        if not current_user.is_setup_completed: return redirect(url_for('setup'))
        status = redis_conn.get(f"migration_status:{current_user.id}")
        if status and status.decode() == 'processing': return render_template('maintenance.html')
        return render_template('chat.html')
    return render_template('landing.html')

# FIX: Missing chat_stream route added
@app.route('/chat_stream', methods=['POST'])
@login_required
def chat_stream():
    data = request.json
    thread_id = data.get('thread_id')
    message = data.get('message')
    model = data.get('model')
    image_urls = data.get('image_urls', [])
    
    api_keys = {
        'openai': decrypt_val(current_user.openai_api_key),
        'gemini': decrypt_val(current_user.gemini_api_key),
        'xai': decrypt_val(current_user.xai_api_key)
    }
    user_config = {'enable_e2ee': current_user.enable_e2ee}
    job_id = f"job_{int(time.time())}_{current_user.id}"
    
    task_queue.enqueue(
        background_chat_task, job_id, thread_id, model, message, image_urls, 
        data, api_keys, current_user.id, user_config,
        job_timeout=600
    )

    def generate():
        pubsub = redis_conn.pubsub()
        channel = f"ai_chat:channel:{job_id}"
        pubsub.subscribe(channel)
        start_time = time.time()
        try:
            for message in pubsub.listen():
                if time.time() - start_time > 600: break
                if message['type'] == 'message':
                    data = json.loads(message['data'])
                    yield json.dumps(data) + "\n"
                    if data['type'] in ['done', 'error']: break
        finally:
            pubsub.unsubscribe()

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

@app.route('/files/<path:filename>')
@login_required
def serve_file(filename):
    parts = filename.split('/')
    if len(parts) > 1 and str(parts[0]) != str(current_user.id): abort(403)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    enc_path = file_path + '.enc'
    mtype = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    if os.path.exists(file_path): return send_file(file_path, mimetype=mtype)
    elif os.path.exists(enc_path):
        with open(enc_path, 'rb') as f: data = decrypt_bytes(f.read())
        return send_file(BytesIO(data), download_name=os.path.basename(filename), as_attachment=False, mimetype=mtype)
    else: abort(404)

@app.route('/changelog')
def changelog():
    log_dir = app.config['CHANGELOG_FOLDER']
    logs = []
    if os.path.exists(log_dir):
        files = glob.glob(os.path.join(log_dir, '*.md'))
        files.sort(key=os.path.getmtime, reverse=True)
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
        safe_db_commit()
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
        current_user.enable_e2ee = (request.form.get('enable_e2ee') == 'on')
        current_user.is_setup_completed = True
        safe_db_commit()
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
        user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if os.path.exists(user_dir):
            for root, dirs, files in os.walk(user_dir, topdown=False):
                for name in files: secure_delete(os.path.join(root, name))
                for name in dirs: os.rmdir(os.path.join(root, name))
            os.rmdir(user_dir)
        db.session.delete(current_user)
        safe_db_commit()
        logout_user()
        return jsonify({'status': 'ok'})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def handle_settings():
    if request.method == 'GET':
        status = redis_conn.get(f"migration_status:{current_user.id}")
        mig_status = status.decode() if status else "idle"
        sp = current_user.system_prompt
        if current_user.enable_e2ee and sp: sp = decrypt_val(sp)
        return jsonify({
            'system_prompt': sp or "",
            'username': current_user.username, 
            'openai_key': decrypt_val(current_user.openai_api_key) or "", 
            'gemini_key': decrypt_val(current_user.gemini_api_key) or "", 
            'xai_key': decrypt_val(current_user.xai_api_key) or "",
            'enable_e2ee': current_user.enable_e2ee,
            'migration_status': mig_status
        })
    d = request.json
    if 'system_prompt' in d: 
        if current_user.enable_e2ee: current_user.system_prompt = encrypt_val(d['system_prompt'])
        else: current_user.system_prompt = d['system_prompt']
    if 'openai_key' in d: current_user.openai_api_key = encrypt_val(d['openai_key'])
    if 'gemini_key' in d: current_user.gemini_api_key = encrypt_val(d['gemini_key'])
    if 'xai_key' in d: current_user.xai_api_key = encrypt_val(d['xai_key'])
    if d.get('new_password'): current_user.set_password(d['new_password'])
    if d.get('new_username') and d['new_username'] != current_user.username:
        if not User.query.filter_by(username=d['new_username']).first(): current_user.username = d['new_username']
    if 'enable_e2ee' in d and d['enable_e2ee'] != current_user.enable_e2ee:
        target_enable = d['enable_e2ee']
        task_queue.enqueue(migrate_e2ee_task, current_user.id, target_enable)
        flash("暗号化設定の変更処理を開始しました。完了までしばらくお待ちください。")
    else:
        safe_db_commit()
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
    safe_db_commit()
    return jsonify({'id': gem.id, 'name': gem.name})

@app.route('/api/gems/<int:gid>', methods=['DELETE'])
@login_required
def delete_gem(gid):
    gem = Gem.query.get_or_404(gid)
    if gem.user_id != current_user.id: return jsonify({'error': '403'}), 403
    db.session.delete(gem)
    safe_db_commit()
    return jsonify({'status': 'deleted'})

@app.route('/api/threads', methods=['GET', 'POST'])
@login_required
def handle_threads():
    if request.method == 'GET':
        q = request.args.get('q', '').strip()
        query = Thread.query.filter_by(user_id=current_user.id)
        if q: 
            if current_user.enable_e2ee: query = query.filter(Thread.title.contains(q))
            else: query = query.join(Message).filter(or_(Thread.title.contains(q), Message.content.contains(q))).distinct()
        ts = query.order_by(Thread.updated_at.desc()).limit(50).all()
        return jsonify([{'id': t.id, 'title': t.title} for t in ts])
    t = Thread(user_id=current_user.id)
    db.session.add(t)
    safe_db_commit()
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
    safe_db_commit()
    return jsonify({'status': 'deleted'})

@app.route('/api/threads/<int:tid>/title', methods=['PUT'])
@login_required
def update_title(tid):
    t = Thread.query.get_or_404(tid)
    if t.user_id != current_user.id: return jsonify({'error': '403'}), 403
    t.title = request.json.get('title', 'Untitled')
    safe_db_commit()
    return jsonify({'status': 'ok'})

@app.route('/api/messages/<int:mid>', methods=['DELETE'])
@login_required
def delete_message(mid):
    msg = Message.query.get_or_404(mid)
    if msg.thread.user_id != current_user.id: return jsonify({'error': '403'}), 403
    Message.query.filter(Message.thread_id == msg.thread_id, Message.timestamp >= msg.timestamp).delete()
    safe_db_commit()
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
                    if os.path.exists(fp) or os.path.exists(fp + '.enc'):
                        seen.add(p)
                        ext = os.path.splitext(p)[1].lower().replace('.', '')
                        files.append({'filename': os.path.basename(p), 'filepath': p, 'url': url_for('serve_file', filename=p), 'type': 'image' if ext in ['png','jpg','webp'] else 'file', 'ext': ext})
        return jsonify(files)
    except: return jsonify([])

@app.route('/api/files/delete', methods=['POST'])
@login_required
def delete_files_batch():
    for f in request.json.get('filenames', []):
        if f.startswith(f"{current_user.id}/"):
            fp = os.path.join(app.config['UPLOAD_FOLDER'], f)
            secure_delete(fp)
            secure_delete(fp + '.enc')
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
            if current_user.enable_e2ee:
                is_image = ext in ['.jpg', '.jpeg', '.png']
                if is_image and not orig_name.endswith('.webp'):
                    try:
                        buf = BytesIO()
                        Image.open(f).convert('RGB').save(buf, 'WEBP', quality=80)
                        enc_data = encrypt_bytes(buf.getvalue())
                        fname = f"{fname_base}.webp"
                        with open(os.path.join(ud, fname + '.enc'), 'wb') as ef: ef.write(enc_data)
                    except:
                        f.seek(0)
                        with open(os.path.join(ud, fname + '.enc'), 'wb') as ef: ef.write(encrypt_bytes(f.read()))
                else:
                    with open(os.path.join(ud, fname + '.enc'), 'wb') as ef: ef.write(encrypt_bytes(f.read()))
            else:
                is_image = ext in ['.jpg', '.jpeg', '.png']
                if is_image and not orig_name.endswith('.webp'):
                    try:
                        Image.open(f).convert('RGB').save(os.path.join(ud, f"{fname_base}.webp"), 'WEBP', quality=80)
                        fname = f"{fname_base}.webp"
                    except:
                        f.seek(0)
                        f.save(save_path)
                else: f.save(save_path)
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

with app.app_context():
    db.create_all()
    try:
        with db.engine.connect() as conn: conn.execute(text("ALTER TABLE message ADD COLUMN thought_signature TEXT"))
    except: pass
    try:
        with db.engine.connect() as conn: conn.execute(text("ALTER TABLE user ADD COLUMN enable_e2ee BOOLEAN DEFAULT 0"))
    except: pass
    try:
        with db.engine.connect() as conn: conn.execute(text("ALTER TABLE message ADD COLUMN is_encrypted BOOLEAN DEFAULT 0"))
    except: pass

if __name__ == '__main__':
    app.run(debug=True)
