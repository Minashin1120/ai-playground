import os
import logging
import redis
from rq import SimpleWorker, Queue

# Preload app module at process start so first user job does not pay import cost.
from app import app as flask_app  # noqa: F401
from app import background_chat_task  # noqa: F401

listen = ['ai_chat_queue']
logger = logging.getLogger(__name__)


def _parse_queue_names():
    raw = os.getenv("RQ_QUEUES", ",".join(listen))
    names = [name.strip() for name in raw.split(",") if name.strip()]
    return names or list(listen)


def main():
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/10')
    conn = redis.from_url(redis_url)
    queue_names = _parse_queue_names()
    try:
        queues = [Queue(name, connection=conn) for name in queue_names]
        worker = SimpleWorker(queues, connection=conn)
        logger.info("Worker starting (queues=%s)", ",".join(queue_names))
        worker.work()
        return 0
    except Exception:
        logger.exception("Worker failed to start or exited with error")
        return 1

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    raise SystemExit(main())
