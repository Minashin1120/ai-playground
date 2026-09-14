"""Exercise bounded history reads and concurrent cache accounting on synthetic data."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock
import threading
import time

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

import app as target


@pytest.fixture
def history_db():
    engine = create_engine('sqlite:///:memory:')
    target.Thread.__table__.create(engine)
    target.Message.__table__.create(engine)
    with Session(engine) as session:
        session.add_all([
            target.Thread(id=1, user_id=7, title='branch'),
            target.Thread(id=2, user_id=8, title='other user'),
            target.Thread(id=3, user_id=7, title='empty'),
        ])
        session.flush()
        session.add_all([
            target.Message(id=i, thread_id=1, role='user', content='body' * 100,
                           parent_id=i - 1 if i > 1 else None, is_encrypted=i % 2 == 0)
            for i in range(1, 151)
        ])
        session.add(target.Message(id=500, thread_id=1, parent_id=1,
                                   content='unrelated branch' * 10000))
        session.add(target.Message(id=600, thread_id=2, content='other account'))
        session.commit()
        message_proxy = SimpleNamespace(
            query=session.query(target.Message), id=target.Message.id,
            thread_id=target.Message.thread_id, parent_id=target.Message.parent_id,
            is_encrypted=target.Message.is_encrypted,
        )
        with mock.patch.object(target, 'db', SimpleNamespace(session=session)), \
             mock.patch.object(target, 'Message', message_proxy):
            yield session, engine
    engine.dispose()


def test_history_only_loads_selected_branch_in_bounded_batches(history_db):
    session, engine = history_db
    loaded = []
    queries = []
    event.listen(session, 'loaded_as_persistent', lambda _s, obj: loaded.append(obj.id))
    event.listen(engine, 'before_cursor_execute', lambda _c, _u, sql, *_args: queries.append(sql))
    result = [row.id for row in target._iter_chat_history_ancestors(1, 150, 7)]
    assert result == list(range(150, 0, -1))
    assert set(loaded) == set(result)
    assert len(queries) == 4  # Links once, then three batches, not 150 queries.
    assert 'content' not in queries[0]


def test_history_limit_and_account_scope(history_db):
    assert [m.id for m in target._iter_chat_history_ancestors(1, 150, 7, 2)] == [150, 149]
    assert list(target._iter_chat_history_ancestors(1, 150, 8)) == []
    assert list(target._iter_chat_history_ancestors(1, 600, 7)) == []
    assert list(target._iter_chat_history_ancestors(1, None, 7)) == []


def test_history_stops_at_cycles_and_preserves_token_updates(history_db):
    session, _engine = history_db
    model = session.query(target.Thread).first().messages[0].__class__
    session.query(model).filter(model.id == 1).update({'parent_id': 2})
    rows = list(target._iter_chat_history_ancestors(1, 2, 7))
    assert [m.id for m in rows] == [2, 1]
    rows[0].tokens_in = 123
    session.commit()
    assert session.get(model, 2).tokens_in == 123


def test_browser_history_preserves_order_decryption_and_signatures(history_db):
    session, _engine = history_db
    thread = session.get(target.Thread, 1)
    parent = SimpleNamespace(id=2)
    with mock.patch.object(target, 'decrypt_val', side_effect=lambda value: 'decoded:' + value):
        history = target._browser_fast_mode_history(thread, parent)
    assert [row['text'] for row in history] == ['body' * 100, 'decoded:' + 'body' * 100]
    assert all(row['role'] == 'user' for row in history)


def test_admin_counts_use_two_queries_without_loading_message_payloads(history_db):
    session, engine = history_db
    queries = []
    event.listen(engine, 'before_cursor_execute', lambda _c, _u, sql, *_args: queries.append(sql))
    thread_proxy = SimpleNamespace(query=session.query(target.Thread), updated_at=target.Thread.updated_at)
    with mock.patch.object(target, 'Thread', thread_proxy), \
         mock.patch.object(target, 'current_user', SimpleNamespace(id=7, is_admin=True, username='test')):
        with target.app.test_request_context('/api/admin/threads'):
            data = target.admin_threads_list.__wrapped__().get_json()
    counts = {row['thread_id']: (row['message_count'], row['encrypted_count']) for row in data['threads']}
    assert counts == {1: (151, 75), 3: (0, 0)}
    assert len(queries) == 2
    assert all('message.content' not in sql for sql in queries)


@pytest.mark.parametrize('kind', ['media', 'thumbnail'])
def test_concurrent_cache_writes_and_evictions_respect_byte_budget(kind):
    prefix = '_MEDIA_BYTES' if kind == 'media' else '_THUMBNAIL_BYTES'
    budget_name = '_MEDIA_BYTES_CACHE_MAX' if kind == 'media' else '_THUMBNAIL_CACHE_MAX'
    item_name = '_MEDIA_BYTES_CACHE_ITEM_MAX' if kind == 'media' else '_THUMBNAIL_CACHE_ITEM_MAX'
    cache = OrderedDict()
    original = target._ordered_lru_bytes_cache_put

    def slow_put(*args):
        # Force competing callers to overlap the counter read / assignment.
        time.sleep(0.001)
        return original(*args)

    changes = {prefix + '_CACHE': cache, prefix + '_CACHE_LOCK': threading.RLock(),
               prefix + '_CACHE_SIZE': 0, budget_name: 128, item_name: 32,
               '_ordered_lru_bytes_cache_put': slow_put}
    put = getattr(target, '_' + kind + '_bytes_cache_put')
    evict = getattr(target, '_' + kind + '_bytes_cache_evict_path')

    def work(i):
        put((str(i),), b'x' * 16)
        if i % 3 == 0:
            evict(str(i))

    with mock.patch.multiple(target, **changes):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(work, range(100)))
        actual = sum(map(len, cache.values()))
        assert getattr(target, prefix + '_CACHE_SIZE') == actual
        assert actual <= 128
