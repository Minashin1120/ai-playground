# AI Chat Playground (V3.7)

![Status](https://img.shields.io/badge/Status-Stable-green)
![Version](https://img.shields.io/badge/Version-3.7.1-blue)
![License](https://img.shields.io/badge/License-MIT-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)

**AI Chat Playground** is a robust, self-hosted multi-model chat platform supporting **Google Gemini 3.0**, **OpenAI GPT-5**, and **xAI Grok**. It is designed for power users who value privacy, featuring a **BYOK (Bring Your Own Key)** architecture, **End-to-End Encryption (E2EE)**, and comprehensive file management.

---

## 🚀 Features

*   **Multi-Model Support:** Seamless switching between Gemini 3.0 Pro/Flash, GPT-5.1, and Grok 4.1.
*   **Thinking Process Visualization:** View the internal "Chain of Thought" for reasoning models.
*   **E2EE (End-to-End Encryption):** Optional database-level encryption for chat logs.
*   **File Isolation:** Secure upload handling with authenticated access routes.
*   **Multi-Modal:** Support for Images and PDF analysis.
*   **Web Debug Console:** Built-in log viewer for administrators.

---

## ⚙️ Installation Guide

This guide assumes you are using **Ubuntu 22.04/24.04 LTS**.

### 1. System Requirements & Dependencies

First, install the necessary system packages for Python compilation, Database connectivity, and Redis.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-dev \
    mariadb-server libmysqlclient-dev \
    redis-server build-essential \
    libssl-dev libffi-dev pkg-config \
    ffmpeg
```

### 2. Database Setup (MariaDB)

Log in to MariaDB and create the database and user.

```bash
sudo mariadb
```

Execute the following SQL commands (Replace `your_password` with a strong password):

```sql
CREATE DATABASE ai_chat_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'ai_chat_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON ai_chat_db.* TO 'ai_chat_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### 3. Application Setup

Clone the repository and set up the Python environment.

```bash
# 1. Clone repository
git clone https://github.com/Minashin1120/ai-playground.git
cd ai-playground

# 2. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python Dependencies
pip install -r requirements.txt
```

### 4. Configuration (.env)

Create a `.env` file in the project root directory.

```bash
nano .env
```

**Content:**

```ini
# Flask Security
FLASK_SECRET_KEY=generate_a_long_random_string_here

# Database Connection (Adjust user/password)
DATABASE_URL=mysql+pymysql://ai_chat_user:your_password@localhost/ai_chat_db

# Redis Connection (Recommend DB 10 to avoid conflicts)
REDIS_URL=redis://localhost:6379/10

# (Optional) Cloudflare Turnstile for Bot Protection
TURNSTILE_SITE_KEY=your_site_key
TURNSTILE_SECRET_KEY=your_secret_key
```

### 5. Initialization

Run the application once to initialize the database tables and generate encryption keys.

```bash
# Ensure venv is active
python app.py
# Press Ctrl+C after you see "Running on http://127.0.0.1:5000"
```

---

## 🚀 Deployment (Production)

For production environments, use **Gunicorn** and **Systemd**.

### 1. Systemd Service Files

Create service files to keep the app and worker running.

**App Service:** `/etc/systemd/system/ai-chat.service`
```ini
[Unit]
Description=Gunicorn instance to serve AI Chat
After=network.target

[Service]
User=your_linux_username
Group=www-data
WorkingDirectory=/path/to/ai-playground
Environment="PATH=/path/to/ai-playground/venv/bin"
ExecStart=/path/to/ai-playground/venv/bin/gunicorn --preload --timeout 300 -w 2 -b 127.0.0.1:3111 app:app

[Install]
WantedBy=multi-user.target
```

**Worker Service:** `/etc/systemd/system/ai-chat-worker@.service`
```ini
[Unit]
Description=AI Chat Worker %i
After=network.target

[Service]
User=your_linux_username
WorkingDirectory=/path/to/ai-playground
Environment="PATH=/path/to/ai-playground/venv/bin"
ExecStart=/path/to/ai-playground/venv/bin/python worker.py

[Install]
WantedBy=multi-user.target
```

### 2. Start Services

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ai-chat
sudo systemctl enable --now ai-chat-worker@1
```

### 3. Reverse Proxy (Apache Example)

To serve the app on port 80/443.

```apache
<VirtualHost *:80>
    ServerName chat.example.com

    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:3111/
    ProxyPassReverse / http://127.0.0.1:3111/
</VirtualHost>
```

---

## 🛠️ Troubleshooting

*   **Database Connection Error:** Ensure `libmysqlclient-dev` is installed and the SQL credentials in `.env` match what you created.
*   **Redis Error:** Check if Redis is running (`sudo systemctl status redis`). Ensure the `REDIS_URL` uses a database number (e.g., `/10`) that is not used by other apps.
*   **Image Upload Fails:** Ensure the `instance/uploads` directory exists and is writable by the user running the application.

---

## 📜 License

This project is licensed under the MIT License.

**Maintained by:** [Minashin1120](https://github.com/Minashin1120)
