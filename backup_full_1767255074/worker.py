import os
import redis
from rq import SimpleWorker, Queue
from app import app, db

listen = ['ai_chat_queue']

class SafeWorker(SimpleWorker):
    def perform_job(self, job, queue):
        # JOB実行前に必ずDB接続をリセットする
        with app.app_context():
            db.engine.dispose()
        return super().perform_job(job, queue)

if __name__ == '__main__':
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/10')
    conn = redis.from_url(REDIS_URL)

    try:
        # DB接続をクリーンにしてから開始
        with app.app_context():
            db.engine.dispose()
            
        queues = [Queue(name, connection=conn) for name in listen]
        worker = SafeWorker(queues, connection=conn)
        print("SafeWorker starting (DB Connection Reset enabled)...")
        worker.work()
    except Exception as e:
        print(f"Worker Error: {e}")
