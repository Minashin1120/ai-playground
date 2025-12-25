# AI Chat Playground (V3.7)

![Status](https://img.shields.io/badge/Status-Stable-green)
![Version](https://img.shields.io/badge/Version-3.7.1-blue)
![License](https://img.shields.io/badge/License-MIT-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Framework](https://img.shields.io/badge/Framework-Flask%203.x-lightgrey)

A robust, multi-model AI chat platform supporting **Google Gemini 3.0**, **OpenAI GPT-5**, and **xAI Grok**.
Designed for privacy, flexibility, and power users, utilizing a **BYOK (Bring Your Own Key)** architecture.

## 🚀 Key Features

### 🧠 Cutting-Edge AI Models
*   **Multi-Model Support:** Seamlessly switch between Gemini 3.0 Pro/Flash, GPT-5.1, and Grok 4.1.
*   **Thinking Process Visualization:** View the internal "Chain of Thought" for supported models (Gemini Thinking, etc.) in a dedicated UI box.
*   **Thinking Level Control:** Adjust the depth of reasoning (Low, Medium, High) directly from the UI.

### 🛡️ Privacy & Security (V3.x)
*   **BYOK Architecture:** Your API keys are encrypted (AES-256) and stored securely.
*   **E2EE (End-to-End Encryption):** Optional database-level encryption for chat history. Even the admin cannot read your messages when enabled.
*   **File Isolation:** Uploaded files are stored in a protected `instance` directory, accessible only via authenticated routes.

### 💻 Developer & Power User UX
*   **Web Debug Console:** Monitor server logs directly from the browser (Admin only).
*   **File Library:** Manage, view, and delete your uploaded images and PDFs.
*   **Markdown & Syntax Highlighting:** Full support for code blocks with copy functionality.
*   **Keyboard Shortcuts:** `Ctrl + Enter` to send, `Ctrl + V` to paste images.

## 🛠️ Tech Stack

*   **Backend:** Python 3.11, Flask, Gunicorn
*   **Database:** MariaDB (MySQL) with SQLAlchemy
*   **Queue System:** Redis, RQ (Redis Queue)
*   **Frontend:** HTML5, Tailwind CSS, JavaScript (Vanilla)
*   **Infrastructure:** Systemd managed services

## ⚙️ Setup & Installation

### Prerequisites
*   Python 3.11+
*   MariaDB / MySQL
*   Redis Server
*   `ffmpeg` (Optional, for media processing)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Minashin1120/ai-playground.git
    cd ai-playground
    ```

2.  **Set up Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables**
    Create a `.env` file in the root directory:
    ```ini
    FLASK_SECRET_KEY=your_secure_random_key
    DATABASE_URL=mysql+pymysql://user:password@localhost/dbname
    REDIS_URL=redis://localhost:6379/10
    
    # Cloudflare Turnstile (Optional)
    TURNSTILE_SITE_KEY=your_site_key
    TURNSTILE_SECRET_KEY=your_secret_key
    ```

4.  **Run the Application**
    ```bash
    # Run using Gunicorn (Production)
    gunicorn --preload --timeout 300 -w 1 -b 0.0.0.0:3111 app:app
    
    # Start the Worker
    python worker.py
    ```

## 📜 License

This project is licensed under the MIT License.

---
**Maintained by:** [Minashin1120](https://github.com/Minashin1120)
