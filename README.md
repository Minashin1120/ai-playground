# AI Chat Playground (Alpha)

![Alpha Version](https://img.shields.io/badge/Status-Alpha-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)

A multi-model AI chat application supporting Google Gemini, OpenAI GPT, and xAI Grok.
Designed for personal use with BYOK (Bring Your Own Key) architecture.

## Features

*   **Multi-Model Support**: Switch seamlessly between Gemini 3.0 Pro, GPT-5, and Grok.
*   **Multi-Modal**: Upload images and PDFs for analysis.
*   **User Isolation**: Individual file storage and settings for multiple users.
*   **Thinking Process**: Visualizes the AI's "Thought" process (for supported models).
*   **File Library**: Manage uploaded files with download/view capabilities.
*   **Markdown Support**: Full Markdown rendering with syntax highlighting and copy buttons.

## Setup

This application is built with Flask and uses MariaDB.

### Prerequisites

*   Python 3.11+
*   MariaDB / MySQL
*   `ffmpeg` (for media processing)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/ai-chat-app.git
    cd ai-chat-app
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Configure environment variables:
    Create a `.env` file:
    ```ini
    FLASK_SECRET_KEY=your_random_secret_key
    DATABASE_URL=mysql+pymysql://user:pass@localhost/dbname
    TURNSTILE_SITE_KEY=your_cloudflare_key (Optional)
    TURNSTILE_SECRET_KEY=your_cloudflare_secret (Optional)
    ```

4.  Run the application:
    ```bash
    gunicorn -w 1 -b 0.0.0.0:3111 app:app
    ```

## Usage

1.  **Register/Login**: Create an account.
2.  **Settings**: Go to settings (gear icon) and enter your API Keys (OpenAI, Gemini, xAI).
3.  **Chat**: Select a model and start chatting.

## License

This project is licensed under the MIT License.
