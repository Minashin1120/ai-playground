import os
import redis
from rq import SimpleWorker, Queue

# Preload app module at process start so first user job does not pay import cost.
from app import app as flask_app  # noqa: F401
from app import background_chat_task  # noqa: F401

listen = ['ai_chat_queue']

if __name__ == '__main__':
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/10')
    conn = redis.from_url(REDIS_URL)
    try:
        queues = [Queue(name, connection=conn) for name in listen]
        worker = SimpleWorker(queues, connection=conn)
        print("Worker starting...")
        worker.work()
    except Exception as e:
        print(f"Worker Error: {e}")
