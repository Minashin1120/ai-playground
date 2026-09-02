"""
mcp_service - AI Chat Playground 外部MCP連携パッケージ.

外部 MCP（Model Context Protocol）サーバーとの接続・認証・ツール連携を提供する。
このディレクトリ名は ``mcp`` ではなく ``mcp_service`` としている。理由:
サービスのワーカー / gunicorn / pytest はすべて ``app/`` ディレクトリを
sys.path 先頭に持つため、``app/mcp/`` という名前のパッケージは PyPI の公式
``mcp`` SDK をシャドウしてしまう（``from mcp import Client`` が壊れる）。
これを避けるため、実装プラン §5 のディレクトリ名 ``mcp/`` を ``mcp_service/``
に読み替えている（公開リポジトリのプレフィックスも ``mcp_service/`` で allowlist
へ登録）。

メモリ制約（RAM 1.9GiB）対策として、重い ``mcp`` SDK（starlette/uvicorn 等を
同梱）は ``mcp_service/client.py`` の関数内でのみ遅延 import する。
"""
