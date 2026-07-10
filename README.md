# AI Chat Playground (V4.8.616)

![Status](https://img.shields.io/badge/Status-Stable-green)
![Version](https://img.shields.io/badge/Version-4.8.616-blue)
![License](https://img.shields.io/badge/License-MIT-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)

> [!WARNING]
> **Security notice:** Versions `V4.8.615` and earlier contain known security
> weaknesses and are deprecated. Do not use them for new or production
> deployments. Upgrade to `V4.8.616` or later.

**AI Chat Playground** is a self-hosted, multi-model chat platform for power users who value privacy. It supports **OpenAI GPT-5.x**, **Google Gemini 3.0**, and **xAI Grok 4.1**, with **BYOK (Bring Your Own Key)**, **2FA (TOTP + Passkeys)**, and **E2EE** for chat data.

---

## 🚀 Key Features

- **Multi-Model Chat**: GPT-5.2 / GPT-5.1, Gemini 3.0 Pro/Flash, Gemini 2.5 Flash-Lite, Grok 4.1 Fast.
- **Image & Video Generation**: GPT Image 2 / GPT-Image, Gemini Image (Nano Banana), **Grok Imagine Image/Video** with option controls.
- **Full Multimodal**: Image/PDF analysis, voice STT/TTS, and realtime voice (OpenAI/xAI).
- **Security First**: BYOK with encrypted storage, optional E2EE for chat logs, TOTP + WebAuthn (Passkeys), session management, and bot protection (Turnstile).
- **File Library**: Drag & drop uploads, large-file chunking, upload history with previews, and per-user storage limits.
- **Customization**: Per-chat system prompts, theme color presets, enhanced code blocks (fold/copy), and smooth UI animations.
- **Admin Tools**: Web debug console, user/ban management, and audit-friendly logs.

---

## ⚙️ System Requirements

**Recommended OS**: Ubuntu 22.04/24.04 LTS or Debian 12

- Python 3.11
- MariaDB 10.x
- Redis
- ffmpeg (for audio/video features)

---

## ⚙️ Installation (Linux)

### 1. Install OS Packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-dev \
    mariadb-server libmysqlclient-dev \
    redis-server build-essential \
    libssl-dev libffi-dev pkg-config \
    ffmpeg
```

### 2. Database Setup (MariaDB)

```bash
sudo mariadb
```

```sql
CREATE DATABASE ai_chat_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'ai_chat_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON ai_chat_db.* TO 'ai_chat_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### 3. Application Setup

```bash
git clone https://github.com/Minashin1120/ai-playground.git
cd ai-playground

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configuration (.env)

Create `.env` in the project root. You can inspect `about_.env.txt` for structure (secrets are not included).

**Minimum required:**

```ini
FLASK_SECRET_KEY=generate_a_long_random_string_here
SITE_PASSWORD=your_site_password
PRIMARY_ADMIN_USERNAME=your_admin_username
DATABASE_URL=mysql+pymysql://ai_chat_user:your_password@localhost/ai_chat_db
REDIS_URL=redis://localhost:6379/10

OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
XAI_API_KEY=your_xai_key
GOOGLE_API_KEY=your_google_key
GOOGLE_CLOUD_PROJECT=your_gcp_project
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

### 5. Google OAuth Setup

To enable Google Login and Google One Tap, you must configure your OAuth 2.0 Client ID in the [Google Cloud Console](https://console.cloud.google.com/):

1.  **Authorized JavaScript origins**:
    *   Add your domain: `https://ai.minashin1120.com`
    *   *Required for Google One Tap.*
2.  **Authorized redirect URIs**:
    *   Add the callback URL: `https://ai.minashin1120.com/login/google/callback`
    *   *Required for standard Google Login button.*

### 6. Initialize & Run


```ini
RUN_SCHEMA_MIGRATIONS=0
UPLOAD_MAX_MB=512
USER_STORAGE_LIMIT_MB=100
ATTACHMENT_MAX_FILES=30
UPLOAD_CONCURRENCY=3
XAI_API_HOST=https://api.x.ai
```

### 5. Initialize & Run

```bash
# Create DB tables + generate keys on first run
python app.py
# Ctrl+C after "Running on http://127.0.0.1:5000"
```

In a separate terminal (same venv):

```bash
python worker.py
```

---

## 🪟 Windows Local Run

### Recommended: WSL2 + Ubuntu (best compatibility)

1. Install WSL2 and Ubuntu 22.04/24.04.
2. Open the Ubuntu terminal and follow the **Linux** steps above.
3. Access the app at `http://127.0.0.1:5000` from Windows.

This is the most reliable path because **Python sandboxing (Code Interpreter)** and some audio/video tooling rely on Linux features.

### Native Windows (limited)

Native Windows can run the Flask app, but some features may be unavailable (e.g., Linux-only sandboxing). You will need:

- Python 3.11
- MariaDB (or compatible MySQL)
- Redis
- ffmpeg in PATH

**Quick outline:**

1. Install Python 3.11, MariaDB, Redis, and ffmpeg (via winget or vendor installers).
2. Clone the repo and set up the venv:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Create `.env` as described above.
4. Run `python app.py` and `python worker.py` in separate terminals.

---

## 🚀 Production (Systemd + Gunicorn)

Use **Gunicorn** with `gthread` to avoid long-request hangs.

```bash
gunicorn --preload --timeout 300 \
  --worker-class gthread --threads 8 \
  -w 2 -b 127.0.0.1:3111 app:app
```

Worker services can use the provided templates in `app/ai-chat-worker@.service`.

---

## 🛠️ Troubleshooting

- **DB locks on startup**: set `RUN_SCHEMA_MIGRATIONS=0` to skip ALTER during startup.
- **Redis errors**: confirm Redis is running and `REDIS_URL` is correct.
- **Upload limits**: adjust `UPLOAD_MAX_MB`, `USER_STORAGE_LIMIT_MB`, and `ATTACHMENT_MAX_FILES`.
- **Too many parallel uploads in browser**: lower `UPLOAD_CONCURRENCY` (default: `3`).
- **ffmpeg not found**: ensure `ffmpeg` is installed and in PATH.

---

## 📜 License

This project is licensed under the MIT License.

Maintained by Minashin1120.
