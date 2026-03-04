import os
import logging
import redis
from rq import SimpleWorker, Queue

# Preload app module at process start so first user job does not pay import cost.
from app import app as flask_app  # noqa: F401
from app import background_chat_task  # noqa: F401

FAST_QUEUE_NAME = os.getenv("AI_CHAT_FAST_QUEUE", "ai_chat_fast_queue")
HEAVY_QUEUE_NAME = os.getenv("AI_CHAT_HEAVY_QUEUE", "ai_chat_heavy_queue")
LEGACY_QUEUE_NAME = "ai_chat_queue"
logger = logging.getLogger(__name__)


def _default_queue_names():
    instance = (os.getenv("WORKER_INSTANCE") or "").strip()
    # 1,2 番は fast 優先。3,4 番は heavy 優先で詰まりを分散する。
    if instance in {"3", "4"}:
        return [HEAVY_QUEUE_NAME, FAST_QUEUE_NAME, LEGACY_QUEUE_NAME]
    return [FAST_QUEUE_NAME, HEAVY_QUEUE_NAME, LEGACY_QUEUE_NAME]


def _parse_queue_names():
    instance = (os.getenv("WORKER_INSTANCE") or "").strip()
    if instance:
        # インスタンス運用時は、global RQ_QUEUES より instance 優先順を優先する。
        raw_instance = os.getenv(f"RQ_QUEUES_INSTANCE_{instance}", "")
        names = [name.strip() for name in raw_instance.split(",") if name.strip()]
        return names or _default_queue_names()
    raw = os.getenv("RQ_QUEUES", "")
    names = [name.strip() for name in raw.split(",") if name.strip()]
    return names or _default_queue_names()


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
