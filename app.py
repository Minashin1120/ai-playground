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
from rq import Queue
from datetime import datetime
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import or_
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import pypdf

# Logging
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

# Redis Setup
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
    if user_key and user_key.strip(): return user_key.strip()
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

# --- Worker Task (Inline to ensure consistency) ---
def background_chat_task(job_id, thread_id, model_key, message_text, img_list, options, api_keys, user_id):
    with app.app_context():
        channel = f"ai_chat:channel:{job_id}"
        r = redis.from_url(REDIS_URL)
        def publish_chunk(dt, d): r.publish(channel, json.dumps({"type": dt, "data": d}))
        
        try:
            all_msgs = Message.query.filter_by(thread_id=thread_id).order_by(Message.timestamp).all()
            history = all_msgs[:-1] if len(all_msgs) > 0 else []
            is_gemini, is_grok = 'gemini' in model_key or 'nano' in model_key, 'grok' in model_key
            req_key = api_keys.get('gemini') if is_gemini else (api_keys.get('xai') if is_grok else api_keys.get('openai'))
            if not req_key: publish_chunk("error", "API Key missing."); return

            gemini_client = genai.Client(api_key=req_key, http_options={'api_version': 'v1alpha'}) if is_gemini else None
            openai_client = OpenAI(api_key=req_key) if not is_gemini and not is_grok else None
            xai_client_std = OpenAI(api_key=req_key, base_url="https://api.x.ai/v1") if is_grok else None

            loaded_files = []
            for fname in img_list:
                info = {'name': fname, 'text': None, 'bytes': None, 'mime': None, 'path': os.path.join(app.config['UPLOAD_FOLDER'], fname)}
                try:
                    if os.path.exists(info['path']):
                        info['mime'] = mimetypes.guess_type(info['path'])[0] or 'application/octet-stream'
                        if fname.lower().endswith('.pdf'): info['mime'] = 'application/pdf'
                        is_img = fname.endswith(('.webp','.png','.jpg','.jpeg','.gif','.mp4'))
                        if not is_img and not fname.lower().endswith('.pdf'):
                            try:
                                with open(info['path'], 'r', encoding='utf-8', errors='ignore') as f: info['text'] = f.read()
                            except: pass
                        if not info['text']:
                            with open(info['path'], 'rb') as f: info['bytes'] = f.read()
                        loaded_files.append(info)
                except: pass

            full_res, thought_accumulated, generated_images = "", "", []
            
            if is_gemini:
                real_model = "gemini-3-pro-preview" if "3.0" in model_key else ("gemini-2.5-flash" if "2.5" in model_key else model_key)
                if "nano-banana-pro" in model_key: real_model = "gemini-3-pro-image-preview"
                elif "nano-banana" in model_key: real_model = "gemini-2.5-flash-image"
                
                config_params = {'temperature': 0.7}
                if "nano" not in model_key:
                    if options.get('enable_thinking'):
                        config_params['thinking_config'] = types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)
                    if options.get('enable_search'): config_params['tools'] = [types.Tool(google_search=types.GoogleSearch())]
                    if options.get('system_prompt'): config_params['system_instruction'] = options.get('system_prompt')
                else: config_params['tools'] = None

                contents = []
                for m in history:
                    parts = [types.Part(text=m.content)]
                    if m.image_url:
                        try:
                            for h_img in json.loads(m.image_url):
                                h_path = os.path.join(app.config['UPLOAD_FOLDER'], h_img)
                                if os.path.exists(h_path): 
                                    with open(h_path, 'rb') as f: parts.append(types.Part.from_bytes(data=f.read(), mime_type='image/webp'))
                        except: pass
                    contents.append(types.Content(role='model' if m.role == 'assistant' else 'user', parts=parts))
                
                curr_parts = [types.Part(text=message_text)]
                for fi in loaded_files:
                    if fi['text']: curr_parts.append(types.Part(text=f"\n\nFile: {fi['name']}\n{fi['text']}"))
                    elif fi['bytes']: curr_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime'] if fi['mime']!='application/octet-stream' else 'image/webp'))
                contents.append(types.Content(role='user', parts=curr_parts))
                
                stream = gemini_client.models.generate_content_stream(model=real_model, contents=contents, config=types.GenerateContentConfig(**config_params))
                for chunk in stream:
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        for part in chunk.candidates[0].content.parts:
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
                for m in history: msgs.append({"role": m.role, "content": m.content})
                
                content_list = [{"type": "text", "text": message_text}]
                for fi in loaded_files:
                    if fi['text']: content_list[0]['text'] += f"\n\n[File]\n{fi['text']}"
                    elif fi['mime'].startswith('image/'):
                        content_list.append({"type": "image_url", "image_url": {"url": f"data:{fi['mime']};base64,{base64.b64encode(fi['bytes']).decode('utf-8')}"}})
                msgs.append({"role": "user", "content": content_list})
                
                kwargs = {"model": model_key, "messages": msgs, "stream": True}
                if is_grok and options.get('enable_search'): kwargs["extra_body"] = {"search_parameters": {"mode": "on"}}
                
                stream = client.chat.completions.create(**kwargs)
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        thought_accumulated += delta.reasoning_content
                        publish_chunk("thought", delta.reasoning_content)
                    if delta.content:
                        full_res += delta.content
                        publish_chunk("content", delta.content)

            msg_entry = Message(thread_id=thread_id, role='assistant', content=full_res, model=model_key, image_url=json.dumps(generated_images) if generated_images else None, thought_data=json.dumps({'text': thought_accumulated}) if thought_accumulated else None)
            db.session.add(msg_entry)
            Thread.query.get(thread_id).updated_at = datetime.utcnow()
            db.session.commit()
            publish_chunk("done", "OK")
        except Exception as e:
            logger.error(f"Worker Error: {e}")
            publish_chunk("error", str(e))


# --- Routes ---
@app.route('/')
def index():
    if current_user.is_authenticated:
        if not current_user.is_setup_completed: return redirect(url_for('setup'))
        return render_template('chat.html')
    return render_template('landing.html')

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
        return jsonify({'system_prompt': current_user.system_prompt or "", 'username': current_user.username, 'openai_key': current_user.openai_api_key or "", 'gemini_key': current_user.gemini_api_key or "", 'xai_key': current_user.xai_api_key or ""})
    d = request.json
    if 'system_prompt' in d: current_user.system_prompt = d['system_prompt']
    if 'openai_key' in d: current_user.openai_api_key = d['openai_key']
    if 'gemini_key' in d: current_user.gemini_api_key = d['gemini_key']
    if 'xai_key' in d: current_user.xai_api_key = d['xai_key']
    if d.get('new_password'): current_user.set_password(d['new_password'])
    if d.get('new_username') and d['new_username'] != current_user.username:
        if not User.query.filter_by(username=d['new_username']).first(): current_user.username = d['new_username']
    db.session.commit()
    return jsonify({'status': 'ok'})

# Gems API
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
        if q: query = query.join(Message).filter(or_(Thread.title.contains(q), Message.content.contains(q))).distinct()
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
        return jsonify([{'id': m.id, 'role': m.role, 'content': m.content, 'image_url': m.image_url, 'model': m.model, 'thought_data': m.thought_data} for m in ms])
    db.session.delete(t)
    db.session.commit()
    return jsonify({'status': 'deleted'})

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
                        files.append({'filename': os.path.basename(p), 'filepath': p, 'url': url_for('static', filename='uploads/'+p), 'type': 'image' if ext in ['png','jpg','webp'] else 'file', 'ext': ext})
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
    os.makedirs(ud, exist_ok=True)
    res = []
    for f in files:
        if f.filename:
            fn = f"{int(time.time())}_{os.urandom(4).hex()}{os.path.splitext(secure_filename(f.filename))[1]}"
            f.save(os.path.join(ud, fn))
            res.append(f"{current_user.id}/{fn}")
    return jsonify({'filename': res[0] if res else '', 'filenames': res})

@app.route('/chat_stream', methods=['POST'])
@login_required
def chat_stream():
    import uuid
    data = request.json
    job_id = str(uuid.uuid4())
    sys_prompt = data.get('system_prompt') 
    if not sys_prompt and data.get('enable_system_prompt'): sys_prompt = current_user.system_prompt
    options = {'enable_search': data.get('enable_search', False), 'enable_thinking': data.get('enable_thinking', False), 'reasoning_effort': data.get('reasoning_effort', 'medium'), 'system_prompt': sys_prompt}
    api_keys = {'openai': get_key_for_user(current_user, 'OPENAI_API_KEY'), 'gemini': get_key_for_user(current_user, 'GEMINI_API_KEY'), 'xai': get_key_for_user(current_user, 'XAI_API_KEY')}
    task_queue.enqueue(background_chat_task, job_id, data.get('thread_id'), data.get('model'), data.get('message'), data.get('image_urls', []), options, api_keys, current_user.id, job_timeout=600)
    
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

if __name__ == '__main__':
    app.run(debug=True)
