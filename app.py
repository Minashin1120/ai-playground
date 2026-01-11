import os
import sys
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

try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user as x_user, assistant as x_assistant, system as x_system, image as x_image
    from xai_sdk.tools import web_search
    XAI_SDK_AVAILABLE = True
except ImportError:
    XAIClient = None
    XAI_SDK_AVAILABLE = False

# Logger Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_force(msg):
    """Force log to stdout/journalctl"""
    try:
        print(f"[AI-CHAT-DEBUG] {msg}", file=sys.stdout, flush=True)
        logger.info(msg)
    except:
        pass

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
        with open(KEY_FILE, 'rb') as kf: cipher = Fernet(kf.read().strip())
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as kf: kf.write(key)
        cipher = Fernet(key)
except Exception as e: logger.error(f'Encryption setup failed: {e}')

def encrypt_val(val):
    if not val or not cipher: return val
    try: return cipher.encrypt(val.encode()).decode()
    except: return val

def decrypt_val(val):
    if not val or not cipher: return val
    try: return cipher.decrypt(val.encode()).decode()
    except: return val

def encrypt_bytes(data):
    if not cipher: return data
    return cipher.encrypt(data)

def decrypt_bytes(data):
    if not cipher: return data
    return cipher.decrypt(data)

def secure_delete(path):
    if os.path.exists(path):
        try:
            with open(path, "wb") as f: f.write(os.urandom(os.path.getsize(path)))
            os.remove(path)
        except: pass

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
    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)

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
def load_user(uid): return User.query.get(int(uid))

@app.before_request
def check_maintenance():
    if app.config.get('MAINTENANCE_MODE'):
        if request.endpoint in ['static', 'login', 'logout', 'toggle_maintenance']: return
        if current_user.is_authenticated and current_user.username == 'minashin1120': return
        return render_template('maintenance.html'), 503

def verify_turnstile(token):
    secret = os.getenv('TURNSTILE_SECRET_KEY')
    if not secret: return True
    try: return requests.post('https://challenges.cloudflare.com/turnstile/v0/siteverify', data={'secret': secret, 'response': token}, timeout=5).json().get('success', False)
    except: return False

def count_tokens(text, model="gpt-4"):
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except: return len(text or "") // 4

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(exc.SQLAlchemyError))
def safe_db_commit(): db.session.commit()

# --- Background Tasks ---

def migrate_e2ee_task(user_id, target_enable):
    with app.app_context():
        db.engine.dispose()
        r = redis.from_url(REDIS_URL)
        r.set(f"migration_status:{user_id}", "processing")
        try:
            user = User.query.get(user_id)
            if not user: return
            user.enable_e2ee = target_enable
            if user.system_prompt:
                if target_enable: user.system_prompt = encrypt_val(user.system_prompt)
                else: user.system_prompt = decrypt_val(user.system_prompt)
            threads = Thread.query.filter_by(user_id=user_id).all()
            for t in threads:
                for m in t.messages:
                    if m.content:
                        if target_enable and not m.is_encrypted: m.content = encrypt_val(m.content)
                        elif not target_enable and m.is_encrypted: m.content = decrypt_val(m.content)
                    if m.thought_data:
                        if target_enable and not m.is_encrypted: m.thought_data = encrypt_val(m.thought_data)
                        elif not target_enable and m.is_encrypted: m.thought_data = decrypt_val(m.thought_data)
                    m.is_encrypted = target_enable
            user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
            if os.path.exists(user_dir):
                for root, dirs, files in os.walk(user_dir):
                    for file in files:
                        fp = os.path.join(root, file)
                        if target_enable:
                            if not file.endswith('.enc'):
                                with open(fp, 'rb') as f: data = f.read()
                                with open(fp + '.enc', 'wb') as f: f.write(encrypt_bytes(data))
                                secure_delete(fp)
                        else:
                            if file.endswith('.enc'):
                                with open(fp, 'rb') as f: data = decrypt_bytes(f.read())
                                new_fp = fp[:-4]
                                with open(new_fp, 'wb') as f: f.write(data)
                                secure_delete(fp)
            safe_db_commit()
            r.set(f"migration_status:{user_id}", "done")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            r.set(f"migration_status:{user_id}", "error")

def background_chat_task(job_id, thread_id, model_key, message_text, img_list, options, api_keys, user_id, user_config):
    with app.app_context():
        db.engine.dispose()
        channel = f"ai_chat:channel:{job_id}"
        r = redis.from_url(REDIS_URL)
        def pub(dt, d): r.publish(channel, json.dumps({"type": dt, "data": d}))
        
        def check_stop():
            if r.get(f"stop_job:{job_id}"):
                log_force(f"Job {job_id} stopped by user.")
                return True
            return False

        try:
            log_force(f"Task Start: model={model_key}, user={user_id}")
            user = User.query.get(user_id)
            final_sys_prompt = options.get('system_prompt')
            if not final_sys_prompt and options.get('enable_system_prompt'):
                if user.system_prompt:
                    sp = user.system_prompt
                    if user.enable_e2ee: sp = decrypt_val(sp)
                    final_sys_prompt = sp
            if final_sys_prompt: options['system_prompt'] = final_sys_prompt
            quote_text = options.get('quote_text')

            all_msgs = Message.query.filter_by(thread_id=thread_id).order_by(Message.timestamp).all()
            history = []
            total_history_tokens = 0
            MAX_CONTEXT_TOKENS = 60000
            
            # Pruning: Iterate backwards to keep most recent
            for m in reversed(all_msgs[:-1]):
                cnt = decrypt_val(m.content) if m.is_encrypted else m.content
                t_len = count_tokens(cnt)
                if total_history_tokens + t_len > MAX_CONTEXT_TOKENS:
                    break
                total_history_tokens += t_len
                sig = m.thought_signature
                # Insert at beginning to maintain order
                history.insert(0, {'role': m.role, 'content': cnt, 'image_url': m.image_url, 'signature': sig})

            model_key = model_key.strip()
            is_gem = 'gemini' in model_key or 'nano' in model_key
            is_grok = 'grok' in model_key
            
            key = None
            if is_gem: key = api_keys.get('gemini')
            elif is_grok: key = api_keys.get('xai')
            else: key = api_keys.get('openai') 

            if not key:
                pub("error", "API Key missing")
                return

            g_client = None; o_client = None; x_client = None
            if is_gem: g_client = genai.Client(api_key=key, http_options={'api_version': 'v1alpha'})
            elif is_grok and XAI_SDK_AVAILABLE: x_client = XAIClient(api_key=key)
            else: o_client = OpenAI(api_key=key, base_url="https://api.x.ai/v1" if is_grok else None)

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
                        else: loaded_files.append({'name': fn, 'text': None, 'bytes': data, 'mime': mime})
                except: pass

            full_res, thought_accumulated, generated_images, final_signature = "", "", [], None

            final_message_text = message_text
            if quote_text:
                final_message_text = f"Context (User Quote):\n\"\"\"\n{quote_text}\n\"\"\"\n\nUser Message:\n{message_text}"

            # --- 1. GEMINI & GEMINI IMAGE ---
            if is_gem:
                log_force("Routing: Gemini Branch")
                
                # Image Generation
                if "nano" in model_key or "image" in model_key:
                    try:
                        # [FIX] Apply System Prompt to Image Prompts if available
                        img_prompt = final_message_text
                        if options.get('system_prompt'):
                            img_prompt = f"{options.get('system_prompt')}\n\n{final_message_text}"

                        img_model = "gemini-2.5-flash-image" if "2.5" in model_key else "gemini-3.0-pro-image-preview"
                        
                        resp = g_client.models.generate_content(
                            model=img_model,
                            contents=[types.Part(text=img_prompt)],
                            config=types.GenerateContentConfig(
                                temperature=0.7,
                                safety_settings=[
                                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
                                ]
                            )
                        )
                        
                        if resp.candidates:
                            for part in resp.candidates[0].content.parts:
                                if hasattr(part, 'thought_signature') and part.thought_signature:
                                    final_signature = base64.b64encode(part.thought_signature).decode('utf-8')

                                if hasattr(part, 'inline_data') and part.inline_data:
                                    ud = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
                                    os.makedirs(ud, exist_ok=True)
                                    fn2 = f"gen_{int(time.time())}_{len(generated_images)}.png"
                                    fp2 = os.path.join(ud, fn2)
                                    if user_config.get('enable_e2ee'):
                                        with open(fp2 + '.enc', 'wb') as f: f.write(encrypt_bytes(part.inline_data.data))
                                    else:
                                        with open(fp2, 'wb') as f: f.write(part.inline_data.data)
                                    generated_images.append(f"{user_id}/{fn2}")
                                    pub("content", f"\n![Image](/files/{user_id}/{fn2})\n")
                                    full_res += f"Generated Image for: {img_prompt}\n"
                        else:
                             pub("error", "No image candidates returned.")
                    except Exception as e:
                        logger.exception("Gemini Image Gen Error")
                        pub("error", f"Gemini Image Gen Error: {str(e)}")

                else:
                    # Text/Chat generation mode
                    rm = model_key
                    if "gemini-3-flash" in model_key: rm = "gemini-3-flash-preview"
                    elif "gemini-3.0-pro" in model_key: rm = "gemini-3.0-pro-preview"
                    elif "gemini-2.5" in model_key: rm = "gemini-2.5-flash"

                    conf = {'temperature': 0.7}
                    if options.get('enable_thinking'):
                        raw_lvl = options.get('thinking_level', 'high').lower()
                        lvl = raw_lvl.upper()
                        conf['thinking_config'] = types.ThinkingConfig(include_thoughts=True, thinking_level=lvl)

                    if options.get('enable_search'):
                        conf['tools'] = [types.Tool(google_search=types.GoogleSearch())]
                    if options.get('system_prompt'):
                        conf['system_instruction'] = options.get('system_prompt')
                    
                    contents = []
                    for m in history:
                        parts = []
                        if m.get('signature'):
                            try: parts.append(types.Part(thought_signature=base64.b64decode(m['signature'])))
                            except: pass
                        if m['content']: parts.append(types.Part(text=m['content']))
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
                                    if d2: parts.append(types.Part.from_bytes(data=d2, mime_type=mimetypes.guess_type(bp2)[0] or 'image/webp'))
                            except: pass
                        if parts: contents.append(types.Content(role='model' if m['role'] == 'assistant' else 'user', parts=parts))

                    curr = [types.Part(text=final_message_text)]
                    for fi in loaded_files:
                        if fi['text']: curr.append(types.Part(text=f"\nFile: {fi['name']}\n{fi['text']}"))
                        elif fi['bytes']: curr.append(types.Part.from_bytes(data=fi['bytes'], mime_type=fi['mime']))
                    contents.append(types.Content(role='user', parts=curr))

                    stream = g_client.models.generate_content_stream(model=rm, contents=contents, config=types.GenerateContentConfig(**conf))
                    for chunk in stream:
                        if check_stop(): break
                        if hasattr(chunk, 'candidates') and chunk.candidates:
                            for cand in chunk.candidates:
                                if hasattr(cand, 'grounding_metadata') and cand.grounding_metadata:
                                    gm = cand.grounding_metadata
                                    if hasattr(gm, 'grounding_chunks'):
                                        sources_text = "\n\n**Sources:**\n"
                                        found = False
                                        for g_chunk in gm.grounding_chunks:
                                            if hasattr(g_chunk, 'web') and g_chunk.web:
                                                sources_text += f"- [{g_chunk.web.title}]({g_chunk.web.uri})\n"
                                                found = True
                                        if found: pub("content", sources_text)

                                for part in cand.content.parts:
                                    if hasattr(part, 'thought_signature') and part.thought_signature:
                                        final_signature = base64.b64encode(part.thought_signature).decode('utf-8')
                                    if hasattr(part, 'thought') and part.thought:
                                        thought_accumulated += part.text
                                        pub("thought", part.text)
                                    elif part.text:
                                        full_res += part.text
                                        pub("content", part.text)

            # --- 2. xAI Grok (Native SDK) ---
            elif is_grok and x_client:
                log_force("Routing: Grok Branch")
                tools = []
                if options.get('enable_search'): tools.append(web_search())
                if tools: chat_session = x_client.chat.create(model=model_key, tools=tools)
                else: chat_session = x_client.chat.create(model=model_key)
                if options.get('system_prompt'): chat_session.append(x_system(options.get('system_prompt')))
                
                for m in history:
                    if m['role'] == 'user':
                        chat_session.append(x_user(m['content']))
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
                                        mime = mimetypes.guess_type(bp2)[0] or 'image/webp'
                                        d_uri = f"data:{mime};base64,{base64.b64encode(d2).decode('utf-8')}"
                                        chat_session.append(x_image(d_uri))
                            except: pass
                    else: chat_session.append(x_assistant(m['content']))
                
                for fi in loaded_files:
                    if fi.get('text'): chat_session.append(x_user(f"\n\n[File: {fi['name']}]\n{fi['text']}"))
                    elif fi.get('bytes') and fi.get('mime', '').startswith('image/'):
                        d_uri = f"data:{fi['mime']};base64,{base64.b64encode(fi['bytes']).decode('utf-8')}"
                        chat_session.append(x_image(d_uri))
                
                chat_session.append(x_user(final_message_text))
                
                stream = chat_session.stream()
                for _, chunk in stream:
                    if check_stop(): break
                    if chunk.reasoning_content:
                        thought_accumulated += chunk.reasoning_content
                        pub("thought", chunk.reasoning_content)
                    if chunk.content:
                        full_res += chunk.content
                        pub("content", chunk.content)

            # --- 3. OpenAI Responses API (or Grok Fallback) ---
            else:
                log_force("Routing: OpenAI/Grok Fallback Branch")
                if 'gpt-image' in model_key:
                    pub("error", "Error: GPT-image models are currently disabled.")
                    return

                client = o_client
                input_data = []
                if options.get('system_prompt'): input_data.append({"role": "system", "content": options.get('system_prompt')})
                
                for m in history:
                    content_block = m['content']
                    input_data.append({"role": m['role'], "content": content_block})

                curr_content = []
                if quote_text: curr_content.append({"type": "text", "text": f"User Quote:\n{quote_text}\n---"})
                curr_content.append({"type": "text", "text": message_text})
                
                for fi in loaded_files:
                    if fi['text']: curr_content[0]['text'] += f"\n\n[File: {fi['name']}]\n{fi['text']}"
                    elif fi.get('bytes') and fi['mime'].startswith('image/'):
                         b64 = base64.b64encode(fi['bytes']).decode('utf-8')
                         curr_content.append({"type": "image_url", "image_url": {"url": f"data:{fi['mime']};base64,{b64}"}})
                
                input_data.append({"role": "user", "content": curr_content})
                
                if is_grok:
                    # Grok Fallback (Chat Completions)
                    kwargs = {"model": model_key, "messages": input_data, "stream": True}
                    stream = client.chat.completions.create(**kwargs)
                    for chunk in stream:
                        if check_stop(): break
                        delta = chunk.choices[0].delta
                        r_content = getattr(delta, 'reasoning_content', None)
                        if r_content:
                            thought_accumulated += r_content
                            pub("thought", r_content)
                        if delta.content:
                            full_res += delta.content
                            pub("content", delta.content)
                else:
                    # OpenAI Responses API
                    kwargs = {"model": model_key, "input": input_data, "stream": True}

                    if options.get('enable_search'):
                        kwargs['tools'] = [{"type": "web_search"}] 
                        log_force("Enabled Web Search Tool (Responses API)")

                    is_reasoning_model = any(x in model_key.lower() for x in ['o1', 'o3', 'gpt-5.2', 'reasoning'])
                    if options.get('reasoning_effort') and is_reasoning_model:
                        kwargs['reasoning_effort'] = options.get('reasoning_effort')

                    log_force(f"Responses API Params: {kwargs.keys()}")
                    stream = client.responses.create(**kwargs)
                    search_reported = False

                    for chunk in stream:
                        if check_stop(): break
                        
                        if hasattr(chunk, 'output_text_delta') and chunk.output_text_delta:
                            if search_reported:
                                pub("search_status", "done")
                                search_reported = False
                            full_res += chunk.output_text_delta
                            pub("content", chunk.output_text_delta)
                        
                        if hasattr(chunk, 'citations') and chunk.citations:
                            citations_text = "\n\n**Sources:**\n"
                            for c in chunk.citations:
                                title = getattr(c, 'title', 'Source')
                                url = getattr(c, 'url', '#')
                                citations_text += f"- [{title}]({url})\n"
                            full_res += citations_text
                            pub("content", citations_text)
                        
                        reasoning_delta = getattr(chunk, 'output_reasoning_text_delta', None)
                        if reasoning_delta:
                            thought_accumulated += reasoning_delta
                            pub("thought", reasoning_delta)

                        if not search_reported and hasattr(chunk, 'output_item') and chunk.output_item:
                             if chunk.output_item.type == "tool_call" or "search" in str(chunk.output_item).lower():
                                 pub("search_status", "searching")
                                 search_reported = True

            final_content = full_res
            final_thought = json.dumps({'text': thought_accumulated}) if thought_accumulated else None
            is_enc = user_config.get('enable_e2ee', False)
            if is_enc:
                final_content = encrypt_val(final_content)
                if final_thought: final_thought = encrypt_val(final_thought)
            
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
            logger.exception("Worker Error")
            log_force(f"Worker Exception: {e}")
            pub("error", str(e))
        finally:
            r.delete(f"stop_job:{job_id}")

@app.route('/')
def index():
    if current_user.is_authenticated:
        if not current_user.is_setup_completed: return redirect(url_for('setup'))
        status = redis_conn.get(f"migration_status:{current_user.id}")
        if status and status.decode() == 'processing': return render_template('maintenance.html')
        return render_template('chat.html')
    return render_template('landing.html')

@app.route('/c/<int:thread_id>')
@login_required
def chat_permalink(thread_id):
    thread = Thread.query.get(thread_id)
    if not thread or thread.user_id != current_user.id:
        return redirect(url_for('index'))
    return render_template('chat.html', initial_thread_id=thread_id)

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

# -----------------------------------------------------------
# Auth Routes
# -----------------------------------------------------------

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

# -----------------------------------------------------------
# API Routes
# -----------------------------------------------------------

@app.route('/chat_stream', methods=['POST'])
@login_required
def chat_stream():
    data = request.json
    def get_k(db_val, env_key):
        k = decrypt_val(db_val)
        if k and str(k).strip(): return k
        if current_user.username == 'minashin1120': return os.getenv(env_key)
        return None
    api_keys = {
        'openai': get_k(current_user.openai_api_key, 'OPENAI_API_KEY'),
        'gemini': get_k(current_user.gemini_api_key, 'GEMINI_API_KEY'),
        'xai': get_k(current_user.xai_api_key, 'XAI_API_KEY')
    }
    user_config = {'enable_e2ee': current_user.enable_e2ee}
    job_id = f"job_{int(time.time())}_{current_user.id}"
    
    try:
        msg_content = data.get('message')
        if user_config['enable_e2ee']: msg_content = encrypt_val(msg_content)
        user_msg = Message(
            thread_id=data.get('thread_id'),
            role='user',
            content=msg_content,
            image_url=json.dumps(data.get('image_urls', [])) if data.get('image_urls') else None,
            is_encrypted=user_config['enable_e2ee']
        )
        db.session.add(user_msg)
        safe_db_commit()
    except Exception as e: logger.error(f"Failed to save user msg: {e}")

    task_queue.enqueue(background_chat_task, job_id, data.get('thread_id'), data.get('model'), data.get('message'), data.get('image_urls', []), data, api_keys, current_user.id, user_config, job_timeout=600)
    
    def generate():
        pubsub = redis_conn.pubsub()
        channel = f"ai_chat:channel:{job_id}"
        pubsub.subscribe(channel)
        start_time = time.time()
        yield json.dumps({"type": "job_id", "data": job_id}) + "\n"
        try:
            for message in pubsub.listen():
                if time.time() - start_time > 600: break
                if message['type'] == 'message':
                    data = json.loads(message['data'])
                    yield json.dumps(data) + "\n"
                    if data['type'] in ['done', 'error']: break
        finally: pubsub.unsubscribe()
    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

@app.route('/api/stop_chat', methods=['POST'])
@login_required
def stop_chat():
    job_id = request.json.get('job_id')
    if job_id:
        redis_conn.set(f"stop_job:{job_id}", "1", ex=300)
        return jsonify({'status': 'stopped'})
    return jsonify({'error': 'no job_id'}), 400

@app.route('/api/generate_title', methods=['POST'])
@login_required
def generate_title_api():
    """Auto-generate chat title with multi-model fallback"""
    try:
        data = request.json
        thread_id = data.get('thread_id')
        thread = Thread.query.get(thread_id)
        if not thread or thread.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        first_msg = Message.query.filter_by(thread_id=thread_id, role='user').order_by(Message.timestamp).first()
        if not first_msg: return jsonify({'status': 'skipped'})
        
        content = decrypt_val(first_msg.content) if first_msg.is_encrypted else first_msg.content
        
        # [FIX] Multi-model fallback logic
        title = "New Chat"
        o_key = decrypt_val(current_user.openai_api_key) or os.getenv('OPENAI_API_KEY')
        g_key = decrypt_val(current_user.gemini_api_key) or os.getenv('GEMINI_API_KEY')
        x_key = decrypt_val(current_user.xai_api_key) or os.getenv('XAI_API_KEY')

        # Try OpenAI (gpt-4o-mini)
        if o_key:
            try:
                client = OpenAI(api_key=o_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Generate a short title (max 6 words) for this chat. Output JSON: {\"title\": \"...\"}"},
                        {"role": "user", "content": content[:500]}
                    ],
                    response_format={"type": "json_object"}
                )
                title = json.loads(resp.choices[0].message.content).get('title', 'New Chat')
            except: pass
        
        # Try Gemini (flash)
        elif g_key and title == "New Chat":
            try:
                g_client = genai.Client(api_key=g_key, http_options={'api_version': 'v1alpha'})
                resp = g_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[types.Part(text=f"Generate a short title (max 6 words) for this chat. JSON: {{'title': '...'}}\n\nChat: {content[:500]}")],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                title = json.loads(resp.text).get('title', 'New Chat')
            except: pass

        # Try xAI (grok-fast)
        elif x_key and XAI_SDK_AVAILABLE and title == "New Chat":
            try:
                x_client = XAIClient(api_key=x_key)
                chat = x_client.chat.create(model="grok-4-1-fast-non-reasoning")
                chat.append(x_system("Generate a short title (max 6 words). Output only the title text."))
                chat.append(x_user(content[:500]))
                # Note: xAI SDK structure might vary, basic call assumed
                resp = chat.stream() # Assuming stream for consistency, or non-stream if available
                # Fallback to simple first chunk or wait
                # For simplicity in this script, we skip deep xAI title impl to avoid complex async
                pass 
            except: pass
            
        thread.title = title
        safe_db_commit()
        return jsonify({'status': 'ok', 'title': title})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

@app.route('/api/threads', methods=['GET', 'POST'])
@login_required
def handle_threads():
    if request.method == 'GET':
        q = request.args.get('q', '').strip()
        page = request.args.get('page', 1, type=int)
        per_page = 20
        query = Thread.query.filter_by(user_id=current_user.id)
        if q: 
            if current_user.enable_e2ee: query = query.filter(Thread.title.contains(q))
            else: query = query.join(Message).filter(or_(Thread.title.contains(q), Message.content.contains(q))).distinct()
        
        pagination = query.order_by(Thread.updated_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
        threads = [{'id': t.id, 'title': t.title} for t in pagination.items]
        return jsonify({
            'threads': threads,
            'has_next': pagination.has_next,
            'next_page': pagination.next_num
        })
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
    
    for m in t.messages:
        if m.image_url:
            try:
                paths = json.loads(m.image_url)
                if not isinstance(paths, list): paths = [paths]
                for p in paths:
                    fp = os.path.join(app.config['UPLOAD_FOLDER'], p)
                    secure_delete(fp)
                    secure_delete(fp + '.enc')
            except: pass

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
    
    msgs_to_delete = Message.query.filter(Message.thread_id == msg.thread_id, Message.timestamp >= msg.timestamp).all()
    for m in msgs_to_delete:
        if m.image_url:
            try:
                paths = json.loads(m.image_url)
                if not isinstance(paths, list): paths = [paths]
                for p in paths:
                    fp = os.path.join(app.config['UPLOAD_FOLDER'], p)
                    secure_delete(fp)
                    secure_delete(fp + '.enc')
            except: pass

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

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp'}
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
            if ext not in ALLOWED_EXTENSIONS:
                return jsonify({'error': f'File type {ext} not allowed'}), 400
            
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
    try:
        with db.engine.connect() as conn: conn.execute(text("ALTER TABLE user ADD COLUMN xai_api_key TEXT"))
    except: pass

if __name__ == '__main__':
    app.run(debug=True)
