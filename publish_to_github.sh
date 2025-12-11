#!/bin/bash

# AI Chat Github Publisher (Smart Update)
# 実行方法: ./publish_to_github.sh

APP_DIR="/home/ai-chat-minashin1120/app"
cd "$APP_DIR" || exit

# 色設定
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}   AI Chat GitHub Updater (V267)        ${NC}"
echo -e "${YELLOW}========================================${NC}"

# 1. Gitの存在確認
if [ -d ".git" ]; then
    echo -e "${GREEN}既存のGitリポジトリを検出しました。${NC}"
else
    echo -e "${YELLOW}Gitリポジトリが見つかりません。初期化します...${NC}"
    git init
    git branch -M main
fi

# 2. セキュリティ設定 (.gitignore) の強制更新
# ※ 過去の設定があっても、セキュリティ事故(.env流出)を防ぐため最新の安全ルールで上書きします
echo "セキュリティ設定(.gitignore)を更新中..."
cat << IGNORE_EOF > .gitignore
# Secrets
.env
instance/
config.py
client_secrets.json

# DB
*.db
*.sqlite3
*.sql

# Python
__pycache__/
*.py[cod]
venv/
.pytest_cache/

# System
.DS_Store
Thumbs.db

# Uploads (プライバシー保護)
static/uploads/*
!static/uploads/.htaccess
!static/uploads/legal/
!static/uploads/changelogs/

# Logs
*.log
IGNORE_EOF

# 3. 変更のステージング
echo "変更ファイルをスキャン中..."
git add .

# ステータス確認
if git diff-index --quiet HEAD --; then
    echo -e "${GREEN}変更点はありません。全て最新です。${NC}"
    exit 0
fi

# 4. コミット
echo -e "今回の更新内容 (コミットメッセージ) を入力してください:"
echo -e "${YELLOW}[Enter]キーのみで 'Update to V2.6.7' となります${NC}"
read -r COMMIT_MSG
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Update to V2.6.7"
fi

git commit -m "$COMMIT_MSG"

# 5. リモートURLの確認とPush
CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null)

if [ -n "$CURRENT_REMOTE" ]; then
    echo -e "送信先リポジトリ: ${GREEN}$CURRENT_REMOTE${NC}"
    echo -e "GitHubへプッシュしますか？ (y/n)"
    read -r PUSH_CONFIRM
    if [ "$PUSH_CONFIRM" == "y" ]; then
        echo "Pushing..."
        git push origin main
        echo -e "${GREEN}公開完了しました！${NC}"
    else
        echo "プッシュをキャンセルしました（ローカルへのコミットは完了しています）。"
    fi
else
    # リモートが未設定の場合のみ聞く
    echo -e "${RED}送信先(GitHub URL)がまだ設定されていません。${NC}"
    echo "リポジトリURLを入力してください (例: https://github.com/user/repo.git):"
    read -r REPO_URL
    if [ -n "$REPO_URL" ]; then
        git remote add origin "$REPO_URL"
        git push -u origin main
        echo -e "${GREEN}初期公開が完了しました！${NC}"
    else
        echo "URLが入力されませんでした。"
    fi
fi

