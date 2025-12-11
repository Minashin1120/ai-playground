import os
import json
import time
import logging
import base64
import mimetypes
import ast
import redis
import shutil
import glob
import requests
from rq import Queue
from datetime import datetime
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for, make_response, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import or_, text
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import pypdf

try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user as xai_user, file as xai_file, system as xai_system
except ImportError:
    XAIClient = None

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
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static/uploads')
app.config['CHANGELOG_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static/changelogs')
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
redis_conn = redis.from_url(REDIS_URL)
task_queue = Queue('ai_chat_queue', connection=redis_conn)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@app.after_request
def add_security_headers(response):
    csp = "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:;"
    response.headers['Content-Security-Policy'] = csp
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password_hash = db.Column(db.String(255))
    system_prompt = db.Column(db.Text, default="")
    openai_api_key = db.Column(db.Text, nullable=True)
    gemini_api_key = db.Column(db.Text, nullable=True)
    xai_api_key = db.Column(db.Text, nullable=True)
    is_setup_completed = db.Column(db.Boolean, default=False)
    threads = db.relationship('Thread', backref='user', lazy=True, cascade="all, delete-orphan")

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

@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))

def get_key_for_user(user, name):
    user_key_field = name.lower()
    user_key = getattr(user, user_key_field, None)
    if user_key and user_key.strip():
        return user_key.strip()
    if user.username == 'minashin1120':
        sys_key = os.getenv(name)
        if sys_key and "placeholder" not in sys_key:
            return sys_key
    return None

def verify_turnstile(token):
    secret = os.getenv('TURNSTILE_SECRET_KEY')
    if not secret:
        if os.getenv('TURNSTILE_SITE_KEY'):
            logger.error("Turnstile Site Key set but Secret Key missing.")
            return False
        return True
    
    if not token:
        return False
        
    try:
        url = 'https://challenges.cloudflare.com/turnstile/v0/siteverify'
        data = {'secret': secret, 'response': token}
        res = requests.post(url, data=data, timeout=5)
        result = res.json()
        return result.get('success', False)
    except Exception as e:
        logger.error(f"Turnstile Verification Error: {e}")
        return False

def background_chat_task(job_id, thread_id, model_key, message_text, img_list, options, api_keys, user_id):
    with app.app_context():
        channel = f"ai_chat:channel:{job_id}"
        r = redis.from_url(REDIS_URL)
        def publish_chunk(data_type, data):
            r.publish(channel, json.dumps({"type": data_type, "data": data}))
        
        try:
            all_msgs = Message.query.filter_by(thread_id=thread_id).order_by(Message.timestamp).all()
            history = all_msgs[:-1] if len(all_msgs) > 0 else []
            
            is_gemini = 'gemini' in model_key or 'nano' in model_key
            is_grok = 'grok' in model_key
            
            req_key = api_keys.get('gemini') if is_gemini else (api_keys.get('xai') if is_grok else api_keys.get('openai'))
            if not req_key:
                publish_chunk("error", "API Key missing.")
                return

            gemini_client = genai.Client(api_key=req_key, http_options={'api_version': 'v1alpha'}) if is_gemini else None
            openai_client = OpenAI(api_key=req_key) if not is_gemini and not is_grok else None
            xai_client_std = OpenAI(api_key=req_key, base_url="https://api.x.ai/v1") if is_grok else None

            loaded_files = []
            for fname in img_list:
                info = {'name': fname, 'text': None, 'bytes': None, 'mime': None, 'path': None}
                try:
                    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                    info['path'] = path
                    if os.path.exists(path):
                        mime, _ = mimetypes.guess_type(path)
                        info['mime'] = mime
                        if not mime:
                            info['mime'] = 'application/octet-stream'
                        
                        is_pdf = fname.lower().endswith('.pdf')
                        is_img = fname.endswith(('.webp','.png','.jpg','.jpeg','.gif','.mp4'))
                        
                        if is_pdf:
                            info['mime'] = 'application/pdf'
                        
                        if not is_img and not is_pdf:
                            try:
                                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                                    info['text'] = f.read()
                            except:
                                pass
                        
                        if not info['text']:
                            with open(path, 'rb') as f:
                                info['bytes'] = f.read()
                except:
                    pass
                loaded_files.append(info)

            full_res = ""
            thought_accumulated = ""
            collected_signatures = {}
            generated_images = []

            if is_gemini:
                real_model = model_key
                if "nano-banana-pro" in model_key: real_model = "gemini-3-pro-image-preview"
                elif "nano-banana" in model_key: real_model = "gemini-2.5-flash-image"
                elif "3.0" in model_key: real_model = "gemini-3-pro-preview"
                elif "2.5" in model_key: real_model = "gemini-2.5-flash"
                
                config_params = {'temperature': 0.7}
                if "nano" not in model_key:
                    if options.get('enable_thinking'):
                        if "3.0" in model_key: config_params['thinking_config'] = types.ThinkingConfig(include_thoughts=True, thinking_level="high")
                        elif "2.5" in model_key: config_params['thinking_config'] = types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)
                    if options.get('enable_search'): config_params['tools'] = [types.Tool(google_search=types.GoogleSearch())]
                    if options.get('system_prompt'): config_params['system_instruction'] = options.get('system_prompt')
                else:
                    config_params['tools'] = None

                config = types.GenerateContentConfig(**config_params)
                contents = []
                
                for m in history:
                    role = 'model' if m.role == 'assistant' else 'user'
                    parts = []
                    t_text = None
                    t_sig = None
                    if m.role == 'assistant' and m.thought_data:
                        try:
                            td = json.loads(m.thought_data)
                            if isinstance(td, dict):
                                t_text = td.get('text')
                                sigs = td.get('signatures')
                                if isinstance(sigs, dict):
                                    t_sig = sigs.get('signature')
                                    if t_sig:
                                        t_sig = base64.b64decode(t_sig)
                        except: pass
                    
                    if t_text and t_sig:
                        parts.append(types.Part(text=m.content, thought=t_text, thought_signature=t_sig))
                    else:
                        parts.append(types.Part(text=m.content))
                    
                    if m.image_url:
                        try:
                            h_imgs = json.loads(m.image_url)
                            if not isinstance(h_imgs, list): h_imgs = [m.image_url]
                            for h_img in h_imgs:
                                h_path = os.path.join(app.config['UPLOAD_FOLDER'], h_img)
                                if os.path.exists(h_path) and h_img.endswith(('.webp','.png','.jpg')):
                                    with open(h_path, 'rb') as f:
                                        parts.append(types.Part.from_bytes(data=f.read(), mime_type='image/webp'))
                        except: pass
                    contents.append(types.Content(role=role, parts=parts))

                curr_parts = [types.Part(text=message_text)]
                for fi in loaded_files:
                    if fi['text']:
                        curr_parts.append(types.Part(text=f"\n\nFile: {fi['name']}\nContent:\n{fi['text']}"))
                    elif fi['bytes']:
                        m_type = fi['mime']
                        if not m_type or m_type == 'application/octet-stream': m_type = 'image/webp'
                        curr_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=m_type))
                contents.append(types.Content(role='user', parts=curr_parts))

                stream = gemini_client.models.generate_content_stream(model=real_model, contents=contents, config=config)
                for chunk in stream:
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, 'thought_signature') and part.thought_signature:
                                try:
                                    collected_signatures['signature'] = base64.b64encode(part.thought_signature).decode('utf-8')
                                except: pass
                            
                            tt_part = None
                            if hasattr(part, 'thought') and part.thought:
                                tt_part = part.thought if isinstance(part.thought, str) else part.text
                            
                            if tt_part:
                                thought_accumulated += tt_part
                                publish_chunk("thought", tt_part)
                            elif part.text:
                                full_res += part.text
                                publish_chunk("content", part.text)
                            
                            if hasattr(part, 'inline_data') and part.inline_data:
                                try:
                                    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                                    if not os.path.exists(user_dir):
                                        os.makedirs(user_dir, exist_ok=True)
                                    fn = f"gen_{int(time.time())}_{len(generated_images)}.png"
                                    Image.open(BytesIO(part.inline_data.data)).save(os.path.join(user_dir, fn))
                                    db_path = f"{user_id}/{fn}"
                                    generated_images.append(db_path)
                                    publish_chunk("content", f"\n\n![Img](/static/uploads/{db_path})\n")
                                except: pass
            
            else:
                client = xai_client_std if is_grok else openai_client
                msgs = []
                if options.get('system_prompt'):
                    msgs.append({"role": "system", "content": options.get('system_prompt')})
                for m in history:
                    msgs.append({"role": m.role, "content": m.content})
                
                content_list = []
                u_text = message_text
                for fi in loaded_files:
                    if fi['text']:
                        u_text += f"\n\n[File]\n{fi['text']}"
                    elif fi['mime'] == 'application/pdf':
                         try:
                             reader = pypdf.PdfReader(fi['path'])
                             pt = ""
                             for p in reader.pages:
                                 pt += p.extract_text() + "\n"
                             u_text += f"\n\n[PDF]\n{pt[:30000]}"
                         except: pass
                
                content_list.append({"type": "text", "text": u_text})
                for fi in loaded_files:
                    if fi['bytes'] and fi['mime'].startswith('image/'):
                        b64 = base64.b64encode(fi['bytes']).decode('utf-8')
                        content_list.append({"type": "image_url", "image_url": {"url": f"data:{fi['mime']};base64,{b64}"}})
                
                msgs.append({"role": "user", "content": content_list})

                kwargs = {"model": model_key, "messages": msgs, "stream": True}
                if options.get('enable_search'):
                    publish_chunk("tool_status", "Searching...")
                    if is_grok:
                        kwargs["extra_body"] = {"search_parameters": {"mode": "on"}}
                
                stream = client.chat.completions.create(**kwargs)
                for chunk in stream:
                    c, t = None, None
                    d = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk.to_dict()
                    if 'choices' in d and len(d['choices']) > 0:
                        delta = d['choices'][0].get('delta', {})
                        c = delta.get('content')
                        t = delta.get('reasoning_content')
                    
                    if t:
                        thought_accumulated += t
                        publish_chunk("thought", t)
                    if c:
                        full_res += c
                        publish_chunk("content", c)

            if full_res or generated_images:
                t_data = json.dumps({'text': thought_accumulated, 'signatures': collected_signatures}) if (thought_accumulated or collected_signatures) else None
                msg_entry = Message(
                    thread_id=thread_id,
                    role='assistant',
                    content=full_res,
                    model=model_key,
                    image_url=json.dumps(generated_images) if generated_images else None,
                    thought_data=t_data
                )
                db.session.add(msg_entry)
                t = Thread.query.get(thread_id)
                if t:
                    t.updated_at = datetime.utcnow()
                db.session.commit()
                
            publish_chunk("done", "OK")

        except Exception as e:
            logger.error(f"Background Worker Error: {e}")
            publish_chunk("error", str(e))

@app.route('/')
def index():
    if current_user.is_authenticated:
        if not current_user.is_setup_completed:
            return redirect(url_for('setup'))
        return render_template('chat.html')
    return render_template('landing.html')

@app.route('/changelog')
def changelog():
    log_dir = app.config['CHANGELOG_FOLDER']
    logs = []
    if os.path.exists(log_dir):
        files = sorted(glob.glob(os.path.join(log_dir, '*.md')), reverse=True)
        for f in files:
            with open(f, 'r', encoding='utf-8') as file:
                logs.append({'content': file.read()})
    return render_template('changelog.html', logs=logs)

@app.route('/login', methods=['GET', 'POST'])
def login():
    site_key = os.getenv('TURNSTILE_SITE_KEY')
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        token = request.form.get('cf-turnstile-response')
        if not verify_turnstile(token):
            return render_template('login.html', site_key=site_key, error="認証チェックに失敗しました。再試行してください。")

        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            if not user.is_setup_completed:
                return redirect(url_for('setup'))
            return redirect(url_for('index'))
        else:
            return render_template('login.html', site_key=site_key, error="ユーザー名またはパスワードが違います")
    return render_template('login.html', site_key=site_key)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    site_key = os.getenv('TURNSTILE_SITE_KEY')
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        token = request.form.get('cf-turnstile-response')
        if not verify_turnstile(token):
             return render_template('signup.html', site_key=site_key, error="認証チェックに失敗しました。再試行してください。")

        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            return render_template('signup.html', site_key=site_key, error="そのユーザー名は既に使用されています")
        
        new_user = User(username=username, is_setup_completed=False)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for('setup'))
        
    return render_template('signup.html', site_key=site_key)

@app.route('/setup', methods=['GET', 'POST'])
@login_required
def setup():
    if current_user.is_setup_completed:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        current_user.openai_api_key = request.form.get('openai_key')
        current_user.gemini_api_key = request.form.get('gemini_key')
        current_user.xai_api_key = request.form.get('xai_key')
        current_user.is_setup_completed = True
        db.session.commit()
        return redirect(url_for('index'))
        
    return render_template('setup.html')

@app.route('/logout')
def logout():
    logout_user()
    resp = make_response(redirect(url_for('index')))
    resp.headers['Cache-Control'] = 'no-cache, no-store'
    return resp

@app.route('/api/account/delete', methods=['POST'])
@login_required
def delete_account():
    try:
        user_id = current_user.id
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
        if os.path.exists(user_upload_dir):
            shutil.rmtree(user_upload_dir)
        db.session.delete(current_user)
        db.session.commit()
        logout_user()
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"Delete Account Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def handle_settings():
    if request.method == 'GET':
        return jsonify({'system_prompt': current_user.system_prompt or "", 'username': current_user.username, 'openai_key': current_user.openai_api_key or "", 'gemini_key': current_user.gemini_api_key or "", 'xai_key': current_user.xai_api_key or ""})
    else:
        d = request.json
        if 'system_prompt' in d: current_user.system_prompt = d['system_prompt']
        if 'openai_key' in d: current_user.openai_api_key = d['openai_key']
        if 'gemini_key' in d: current_user.gemini_api_key = d['gemini_key']
        if 'xai_key' in d: current_user.xai_api_key = d['xai_key']
        
        if 'new_password' in d and d['new_password']:
            current_user.set_password(d['new_password'])
            
        if 'new_username' in d and d['new_username'] and d['new_username'] != current_user.username:
            if User.query.filter_by(username=d['new_username']).first():
                return jsonify({'error': 'Username taken'}), 400
            current_user.username = d['new_username']

        db.session.commit()
        return jsonify({'status': 'ok'})

@app.route('/api/threads', methods=['GET', 'POST'])
@login_required
def handle_threads():
    if request.method == 'GET':
        q = request.args.get('q', '').strip()
        query = Thread.query.filter_by(user_id=current_user.id)
        if q:
            query = query.join(Message).filter(or_(Thread.title.contains(q), Message.content.contains(q))).distinct()
        ts = query.order_by(Thread.updated_at.desc()).limit(50).all()
        return jsonify([{'id': t.id, 'title': t.title} for t in ts])
    else:
        t = Thread(user_id=current_user.id)
        db.session.add(t)
        db.session.commit()
        return jsonify({'id': t.id, 'title': t.title})

@app.route('/api/threads/<int:tid>', methods=['GET', 'DELETE'])
@login_required
def handle_thread_item(tid):
    t = Thread.query.get_or_404(tid)
    if t.user_id != current_user.id:
        return jsonify({'error': '403'}), 403
        
    if request.method == 'GET':
        ms = Message.query.filter_by(thread_id=tid).order_by(Message.timestamp).all()
        return jsonify([{
            'id': m.id, 'role': m.role, 'content': m.content, 
            'image_url': m.image_url, 'model': m.model, 'thought_data': m.thought_data
        } for m in ms])
    else:
        db.session.delete(t)
        db.session.commit()
        return jsonify({'status': 'deleted'})

@app.route('/api/threads/<int:tid>/title', methods=['PUT'])
@login_required
def update_title(tid):
    t = Thread.query.get_or_404(tid)
    if t.user_id != current_user.id:
        return jsonify({'error': '403'}), 403
    t.title = request.json.get('title', 'Untitled')
    db.session.commit()
    return jsonify({'status': 'ok'})

@app.route('/api/messages/<int:mid>', methods=['DELETE'])
@login_required
def delete_message(mid):
    msg = Message.query.get_or_404(mid)
    t = Thread.query.get(msg.thread_id)
    if t.user_id != current_user.id:
        return jsonify({'error': '403'}), 403
    Message.query.filter(Message.thread_id == msg.thread_id, Message.timestamp >= msg.timestamp).delete()
    db.session.commit()
    return jsonify({'status': 'ok'})


@app.route('/api/files', methods=['GET'])
@login_required
def get_files_lib():
    try:
        # DBクエリ
        msgs = Message.query.join(Thread).filter(
            Thread.user_id == current_user.id, 
            Message.image_url != None
        ).order_by(Message.timestamp.desc()).all()
        
        files = []
        seen = set()
        
        # アップロードフォルダの確認
        upload_base = app.config.get('UPLOAD_FOLDER', 'static/uploads')
        
        for m in msgs:
            if not m.image_url: continue
            try:
                img_list = json.loads(m.image_url)
                if not isinstance(img_list, list): img_list = [m.image_url]
            except: 
                # JSONパースエラー時は文字列として扱う
                img_list = [m.image_url]
            
            for path_str in img_list:
                if path_str and path_str not in seen:
                    try:
                        # パス連結の安全性確保
                        full_path = os.path.join(upload_base, path_str)
                        if os.path.exists(full_path):
                            seen.add(path_str)
                            ext = os.path.splitext(path_str)[1].lower().replace('.', '')
                            ftype = 'image' if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp'] else 'file'
                            files.append({
                                'id': m.id,
                                'filename': os.path.basename(path_str),
                                'filepath': path_str,
                                'url': url_for('static', filename='uploads/' + path_str),
                                'type': ftype, 
                                'ext': ext, 
                                'date': m.timestamp.strftime('%Y-%m-%d %H:%M')
                            })
                    except:
                        continue
        return jsonify(files)
    except Exception as e:
        # 万が一エラーが起きても、空リストを返してクライアントをフリーズさせない
        print(f"API Files Error: {e}")
        return jsonify([])

@app.route('/api/files/delete', methods=['POST'])
@login_required
def delete_files_batch():
    fnames = request.json.get('filenames', [])
    deleted_count = 0
    for rel_path in fnames:
        if not rel_path.startswith(f"{current_user.id}/"):
            continue
        fp = os.path.join(app.config['UPLOAD_FOLDER'], rel_path)
        if os.path.exists(fp):
            try: os.remove(fp)
            except: pass
        deleted_count += 1
    return jsonify({'status': 'ok', 'count': deleted_count})

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    files = request.files.getlist('file')
    if not files: return jsonify({'error': 'No file'}), 400
    
    user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    if not os.path.exists(user_upload_dir):
        os.makedirs(user_upload_dir, exist_ok=True)
        os.chmod(user_upload_dir, 0o777)

    results = []
    for f in files:
        if not f.filename: continue
        try:
            orig_name = secure_filename(f.filename)
            ext = os.path.splitext(orig_name)[1].lower()
            if not ext: ext = ""
            
            fname_base = f"{int(time.time())}_{os.urandom(4).hex()}"
            fname = f"{fname_base}{ext}"
            save_path = os.path.join(user_upload_dir, fname)
            
            is_image = ext in ['.jpg', '.jpeg', '.png']
            if is_image:
                try:
                    Image.open(f).convert('RGB').save(os.path.join(user_upload_dir, f"{fname_base}.webp"), 'WEBP', quality=80)
                    fname = f"{fname_base}.webp"
                except:
                    f.seek(0)
                    f.save(save_path)
            else:
                f.save(save_path)
            
            results.append(f"{current_user.id}/{fname}")
        except Exception as e: logger.error(f"Upload Error: {e}")
    
    resp = {'filenames': results}
    if results:
        resp['filename'] = results[0]
        resp['url'] = url_for('static', filename='uploads/' + results[0])
    return jsonify(resp)


@app.route('/chat_stream', methods=['POST'])
@login_required
def chat_stream():
    import uuid
    import time
    
    data = request.json
    thread_id = data.get('thread_id')
    message = data.get('message')
    model = data.get('model')
    image_urls = data.get('image_urls', [])
    
    options = {
        'enable_search': data.get('enable_search', False),
        'enable_thinking': data.get('enable_thinking', False),
        'reasoning_effort': data.get('reasoning_effort', 'medium'),
        'system_prompt': current_user.system_prompt if data.get('enable_system_prompt') else None
    }

    api_keys = {
        'openai': get_key_for_user(current_user, 'OPENAI_API_KEY'),
        'gemini': get_key_for_user(current_user, 'GEMINI_API_KEY'),
        'xai': get_key_for_user(current_user, 'XAI_API_KEY')
    }

    job_id = str(uuid.uuid4())
    
    # background_chat_taskをキューに追加
    task_queue.enqueue(
        background_chat_task,
        job_id, thread_id, model, message, image_urls, options, api_keys, current_user.id,
        job_timeout=600
    )

    # Redis PubSubを購読してクライアントにストリーミング
    def generate():
        pubsub = redis_conn.pubsub()
        channel = f"ai_chat:channel:{job_id}"
        pubsub.subscribe(channel)
        start_time = time.time()
        try:
            for message in pubsub.listen():
                if time.time() - start_time > 600:
                    yield json.dumps({"type": "error", "data": "Timeout"}) + "\n"
                    break
                if message['type'] == 'message':
                    data_str = message['data'].decode('utf-8')
                    yield data_str + "\n"
                    try:
                        # 終了シグナル判定
                        if json.loads(data_str).get('type') in ['done', 'error']:
                            break
                    except: pass
        finally:
            pubsub.unsubscribe()
            pubsub.close()

    return Response(stream_with_context(generate()), mimetype='application/json')


if __name__ == '__main__': app.run(debug=True)
