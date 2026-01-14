import os
import redis
from rq import SimpleWorker, Queue
from app import app, db

listen = ['ai_chat_queue']

class SafeWorker(SimpleWorker):
    def perform_job(self, job, queue):
        with app.app_context():
            db.engine.dispose()
        return super().perform_job(job, queue)

if __name__ == '__main__':
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/10')
    conn = redis.from_url(REDIS_URL)
    try:
        with app.app_context():
            db.engine.dispose()
        queues = [Queue(name, connection=conn) for name in listen]
        worker = SafeWorker(queues, connection=conn)
        print("SafeWorker starting...")
        worker.work()
    except Exception as e:
        print(f"Worker Error: {e}")