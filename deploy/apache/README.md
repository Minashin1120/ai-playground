# Apache設定

`ai-playground.conf` はHTTPからHTTPSへの転送と、HTTPSからGunicornへのリバースプロキシ例です。`chat.example.com`、証明書パス、アップロード上限を環境へ合わせて変更します。

```bash
sudo a2enmod proxy proxy_http headers rewrite ssl reqtimeout env alias deflate
sudo cp ai-playground.conf /etc/apache2/sites-available/chat.example.com.conf
sudo apache2ctl configtest
sudo a2ensite chat.example.com.conf
sudo systemctl reload apache2
```

SSEへ影響するため、プロキシやCDNでレスポンスをバッファしないでください。`LimitRequestBody` は `.env` の `UPLOAD_MAX_MB` 以上にします。TLS証明書はCertbot等で取得し、秘密鍵の権限を維持してください。

Android向けの設定と試験手順は [ANDROID_CLIENT.md](../ANDROID_CLIENT.md) を参照してください。通常のチャットはNDJSON、リアルタイム系はSSEです。`Authorization` ヘッダーを上流へ渡し、`/api/mobile/`・`/android/connect`・チャットストリームを共有キャッシュへ保存しない構成にしています。Android向けにCORSを開放する必要はありません。
