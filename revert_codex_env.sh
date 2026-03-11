#!/bin/bash

# ホームディレクトリを $HOME で確実に扱う
ENV_FILE="$HOME/.env"
# シェルを検知して適切な rc ファイルを選ぶ
if [[ "$SHELL" == */zsh ]]; then
  RC_FILE="$HOME/.zshrc"
else
  RC_FILE="$HOME/.bashrc"
fi

# .env を削除（存在する場合）
if [ -f "$ENV_FILE" ]; then
  rm "$ENV_FILE"
  echo ".env ファイルを削除しました。"
else
  echo ".env ファイルは存在しません。"
fi

# rc ファイルから .env ロード部分を削除（sed で安全に除去）
if [ -f "$RC_FILE" ]; then
  sed -i '/# .env を自動ロード（export付き）/,+5d' "$RC_FILE"
  echo "rc ファイル ($RC_FILE) から設定を削除しました。"
else
  echo "rc ファイル ($RC_FILE) は存在しません。"
fi

# 即時反映（環境変数をクリア）
unset OPENAI_API_KEY
source "$RC_FILE" || echo "rc ファイルの反映に失敗しましたが、続行します。"

# Codex のログインをクリア
codex logout || echo "codex logout に失敗しましたが、続行します。"

# 確認
echo "元に戻しました！ OPENAI_API_KEY: $OPENAI_API_KEY (空のはず)"
echo "使用中の rc ファイル: $RC_FILE"
echo "新しいターミナルで確認してください。"
