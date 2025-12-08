import os
import redis
from rq import Worker, Queue, Connection
from app import app

listen = ['ai_chat_queue']

if __name__ == '__main__':
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    conn = redis.from_url(REDIS_URL)
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
