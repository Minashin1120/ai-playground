import os
import redis
# SimpleWorkerをインポート
from rq import SimpleWorker, Queue
from app import app

listen = ['ai_chat_queue']

if __name__ == '__main__':
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    conn = redis.from_url(REDIS_URL)

    try:
        queues = [Queue(name, connection=conn) for name in listen]
        # Worker -> SimpleWorker に変更 (フォークせず即実行)
        worker = SimpleWorker(queues, connection=conn)
        print("SimpleWorker starting (Latency optimized)...")
        worker.work()
    except Exception as e:
        print(f"Worker Error: {e}")
