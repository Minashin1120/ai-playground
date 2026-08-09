# systemd設定

`ai-chat.service` はGunicornを127.0.0.1:3111で起動します。`ai-chat-worker@.service` はインスタンス番号に応じて優先キューを変えるRQワーカーです。1、2はfast優先、3、4はheavy優先です。

コピー前に次の値を確認してください。

- `User`／`Group`
- `WorkingDirectory`
- `EnvironmentFile`
- `ExecStart`と仮想環境のパス
- ログディレクトリへの書込権限

```bash
sudo cp ai-chat.service ai-chat-worker@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-chat.service 'ai-chat-worker@1.service' \
  'ai-chat-worker@2.service' 'ai-chat-worker@3.service' 'ai-chat-worker@4.service'
```

ワーカーは処理中ジョブを完了できるよう `TimeoutStopSec=660` としています。Web側はSSE切断後の再接続を前提に、短いgraceful timeoutを使用します。
