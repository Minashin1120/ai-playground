# 回帰テスト

pytestによるバックエンド・テンプレート・JavaScript回帰テストを格納します。単体動作だけでなく、セキュリティ境界、ストリーム再接続、モーダル履歴、PWA、アップロード、モデル固有ルーティングを検証します。

```bash
cd /opt/ai-playground
venv/bin/python -m pytest -q
```

個別実行例:

```bash
venv/bin/python -m pytest -q tests/test_security_regressions.py
venv/bin/python -m pytest -q tests/test_progress_spinner_regressions.py
```

テストデータは一時領域とモックで生成されることを前提としています。実運用のデータベースや外部APIキーはテストに使用しません。
