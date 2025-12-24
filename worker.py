import os
import redis
from rq import SimpleWorker, Queue
from app import app

listen = ['ai_chat_queue']

if __name__ == '__main__':
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/10')
    conn = redis.from_url(REDIS_URL)

    try:
        queues = [Queue(name, connection=conn) for name in listen]
        worker = SimpleWorker(queues, connection=conn)
        print("SimpleWorker starting (Latency optimized)...")
        worker.work()
    except Exception as e:
        print(f"Worker Error: {e}")
