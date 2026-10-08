"""Android serverless mode: chat sync and API key export (isolated SQL and ephemeral Redis)."""
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid
from unittest import mock

import redis
import app as target


@unittest.skipUnless(shutil.which('redis-server'), 'redis-server needed for mobile API tests')
class MobileSyncApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.redis_dir = tempfile.TemporaryDirectory(prefix='mobile-sync-test-')
        socket = cls.redis_dir.name + '/redis.sock'
        cls.redis_process = subprocess.Popen(
            ['redis-server', '--port', '0', '--unixsocket', socket, '--save', '', '--appendonly', 'no'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        cls.redis = redis.Redis(unix_socket_path=socket)
        for _ in range(100):
            try:
                cls.redis.ping()
                break
            except redis.ConnectionError:
                time.sleep(.02)
        else:
            cls.redis_process.terminate()
            cls.redis_process.wait(timeout=5)
            cls.redis_dir.cleanup()
            raise RuntimeError('isolated Redis did not start')

    @classmethod
    def tearDownClass(cls):
        cls.redis.close()
        cls.redis_process.terminate()
        cls.redis_process.wait(timeout=5)
        cls.redis_dir.cleanup()

    def setUp(self):
        self.redis.flushdb()
        upload_dir = tempfile.TemporaryDirectory(prefix='mobile-sync-uploads-')
        self.addCleanup(upload_dir.cleanup)
        self.upload_dir = upload_dir.name
        for patcher in [
            mock.patch.object(target, 'redis_conn', self.redis),
            mock.patch.object(target, '_ensure_temp_chat_monitor_running'),
            mock.patch.object(target, '_bot_turnstile_active', return_value=False),
            mock.patch.object(target, '_bot_lock_info', return_value=(False, '', 0)),
            mock.patch.dict(target.app.config, TESTING=True, MOBILE_API_ENABLED=True,
                            MAINTENANCE_MODE=False, TRUSTED_HOSTS=['localhost'], UPLOAD_FOLDER=upload_dir.name),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            user = target.User(username='sync-owner', is_setup_completed=True)
            other = target.User(username='sync-other', is_setup_completed=True)
            target.db.session.add_all([user, other])
            target.db.session.commit()
            self.user_id, self.other_id = user.id, other.id
        self.native = target.app.test_client(use_cookies=False)
        self.browser = target.app.test_client()
        with self.browser.session_transaction() as sess:
            sess['_user_id'] = str(self.user_id)
            sess['_fresh'] = True
            sess['csrf_token'] = 'browser-csrf'

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()

    def token(self):
        grant = self.native.post('/api/mobile/v1/device', base_url='https://localhost',
                                 json={'client_id': 'official-android', 'device_name': 'Sync Android'}).json
        self.browser.post('/android/connect', base_url='https://localhost', data={
            'csrf_token': 'browser-csrf', 'user_code': grant['user_code'], 'decision': 'approve'})
        response = self.native.post('/api/mobile/v1/token', base_url='https://localhost',
                                    json={'client_id': 'official-android', 'device_code': grant['device_code']})
        self.assertEqual(response.status_code, 200)
        return response.json['access_token']

    def call(self, path, token, method='GET', **kwargs):
        return self.native.open(path, method=method, base_url='https://localhost',
                                headers={'Authorization': 'Bearer ' + token}, **kwargs)

    @staticmethod
    def uid():
        return str(uuid.uuid4())

    def push_chat(self, token, thread_uuid, messages, **meta):
        return self.call('/api/mobile/v1/sync/push', token, 'POST', json={'threads': [{
            'client_uuid': thread_uuid, 'title': meta.pop('title', 'Device chat'),
            'meta_changed_at_ms': int(time.time() * 1000), 'messages': messages, **meta,
        }]})

    def test_push_creates_chats_idempotently_with_parent_links(self):
        token = self.token()
        thread_uuid, first, second = self.uid(), self.uid(), self.uid()
        messages = [
            {'client_uuid': first, 'role': 'user', 'content': 'こんにちは', 'model': 'gemini-3.6-flash', 'created_at_ms': 1_700_000_000_000},
            {'client_uuid': second, 'role': 'assistant', 'content': '回答', 'thought': '考え', 'model': 'gemini-3.6-flash',
             'parent': {'client_uuid': first}, 'tokens_in': 5, 'tokens_out': 7, 'created_at_ms': 1_700_000_001_000},
        ]
        response = self.push_chat(token, thread_uuid, messages)
        self.assertEqual(response.status_code, 200, response.json)
        result = response.json['threads'][0]
        self.assertEqual(result['rejected'], [])
        ids = {m['client_uuid']: m['id'] for m in result['messages']}
        public_id = result['id']
        again = self.push_chat(token, thread_uuid, messages).json['threads'][0]
        self.assertEqual(again['id'], public_id)
        self.assertEqual({m['client_uuid']: m['id'] for m in again['messages']}, ids)
        thread = self.call(f'/api/threads/{public_id}', token).json
        self.assertEqual(len(thread['messages']), 2)
        self.assertEqual(thread['messages'][1]['parent_id'], ids[first])
        self.assertEqual(json.loads(thread['messages'][1]['thought_data'])['text'], '考え')
        self.assertEqual(thread['title'], 'Device chat')
        # A child of an unknown parent and another user's attachment are rejected.
        rejected = self.push_chat(token, thread_uuid, [
            {'client_uuid': self.uid(), 'role': 'user', 'content': 'x', 'parent': {'client_uuid': self.uid()}},
            {'client_uuid': self.uid(), 'role': 'user', 'content': 'y', 'files': [f'{self.other_id}/1_a.png']},
            {'client_uuid': self.uid(), 'role': 'system', 'content': 'z'},
        ]).json['threads'][0]['rejected']
        self.assertEqual(sorted(r['reason'] for r in rejected), ['invalid_files', 'invalid_role', 'parent_missing'])

    def test_push_encrypts_for_e2ee_users(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.enable_e2ee = True
            target.db.session.commit()
        token = self.token()
        self.push_chat(token, self.uid(), [{'client_uuid': self.uid(), 'role': 'user', 'content': '秘密の本文'}])
        with target.app.app_context():
            message = target.Message.query.one()
            self.assertTrue(message.is_encrypted)
            self.assertNotIn('秘密', message.content)
            self.assertEqual(target.decrypt_val(message.content), '秘密の本文')

    def test_changes_report_new_edited_and_deleted_chats(self):
        token = self.token()
        created = self.call('/api/threads', token, 'POST', json={}).json['id']
        temporary = self.call('/api/threads', token, 'POST', json={'is_temporary': True}).json['id']
        first = self.call('/api/mobile/v1/sync/changes', token)
        self.assertEqual(first.status_code, 200)
        ids = [row['id'] for row in first.json['threads']]
        self.assertIn(created, ids)
        self.assertNotIn(temporary, ids)
        since = first.json['server_time_ms'] - 1000
        self.assertEqual(self.call(f'/api/threads/{created}/title', token, 'PUT', json={'title': '改名'}).status_code, 200)
        changed = self.call(f'/api/mobile/v1/sync/changes?since={since}', token).json
        self.assertEqual([row['title'] for row in changed['threads'] if row['id'] == created], ['改名'])
        self.assertEqual(self.call(f'/api/threads/{created}', token, 'DELETE').status_code, 200)
        after = self.call(f'/api/mobile/v1/sync/changes?since={since}', token).json
        self.assertIn(created, [t['id'] for t in after['tombstones']])
        self.assertNotIn(created, [row['id'] for row in after['threads']])

    def test_message_deletion_touches_the_chat_and_push_can_delete(self):
        token = self.token()
        thread_uuid, first, second = self.uid(), self.uid(), self.uid()
        pushed = self.push_chat(token, thread_uuid, [
            {'client_uuid': first, 'role': 'user', 'content': 'a', 'created_at_ms': 1_700_000_000_000},
            {'client_uuid': second, 'role': 'assistant', 'content': 'b', 'parent': {'client_uuid': first}, 'created_at_ms': 1_700_000_001_000},
        ]).json['threads'][0]
        public_id = pushed['id']
        ids = {m['client_uuid']: m['id'] for m in pushed['messages']}
        with target.app.app_context():
            state = target.db.session.get(target.SyncThreadState, target.Thread.query.filter_by(public_id=public_id).one().id)
            state.changed_at = target.datetime(2000, 1, 1)
            target.db.session.commit()
        since = int(time.time() * 1000) - 1000
        self.assertEqual(self.call(f'/api/messages/{ids[second]}', token, 'DELETE').status_code, 200)
        changed = self.call(f'/api/mobile/v1/sync/changes?since={since}', token).json
        self.assertIn(public_id, [row['id'] for row in changed['threads']])
        with target.app.app_context():
            self.assertIsNone(target.SyncMessageRef.query.filter_by(client_uuid=second).first())
        deleted = self.call('/api/mobile/v1/sync/push', token, 'POST', json={'deleted_messages': [ids[first]], 'deleted_threads': [public_id]})
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.call(f'/api/threads/{public_id}', token).status_code, 403)

    def test_other_users_and_cookies_cannot_use_sync(self):
        token = self.token()
        pushed = self.push_chat(token, self.uid(), [{'client_uuid': self.uid(), 'role': 'user', 'content': 'mine'}]).json['threads'][0]
        with target.app.app_context():
            other_thread = target.Thread(user_id=self.other_id, public_id='other-thread', title='Other')
            target.db.session.add(other_thread)
            target.db.session.commit()
        foreign = self.call('/api/mobile/v1/sync/push', token, 'POST', json={'threads': [{'id': 'other-thread', 'messages': []}]})
        self.assertEqual(foreign.json['threads'][0]['error'], 'thread_not_found')
        self.assertNotIn('other-thread', [row['id'] for row in self.call('/api/mobile/v1/sync/changes', token).json['threads']])
        self.assertIn(pushed['id'], [row['id'] for row in self.call('/api/mobile/v1/sync/changes', token).json['threads']])
        browser = self.browser.get('/api/mobile/v1/sync/changes', base_url='https://localhost')
        self.assertIn(browser.status_code, (401, 403, 404))

    def test_diagnostics_are_accepted_only_from_administrators(self):
        log_dir = tempfile.TemporaryDirectory(prefix='mobile-diagnostics-')
        self.addCleanup(log_dir.cleanup)
        path = os.path.join(log_dir.name, 'diagnostics', 'android-diagnostics.log')
        patcher = mock.patch.dict(target.app.config, ANDROID_DIAGNOSTICS_LOG=path)
        patcher.start()
        self.addCleanup(patcher.stop)
        token = self.token()
        entries = {'entries': [{'t': 1, 'seq': 1, 'ev': 'gen.start', 'model': 'gemini-3.6-flash'}, 'not an object']}
        refused = self.call('/api/mobile/v1/diagnostics', token, 'POST', json=entries)
        self.assertEqual(refused.status_code, 403)
        self.assertFalse(os.path.exists(path))
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.is_admin = True
            target.db.session.commit()
        accepted = self.call('/api/mobile/v1/diagnostics', token, 'POST', json=entries)
        self.assertEqual(accepted.status_code, 200, accepted.json)
        self.assertEqual(accepted.json['accepted'], 1)
        with open(path, encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['user_id'], self.user_id)
        self.assertEqual(rows[0]['entry']['ev'], 'gen.start')
        invalid = self.call('/api/mobile/v1/diagnostics', token, 'POST', json={'entries': 'x'})
        self.assertEqual(invalid.status_code, 400)
        browser = self.browser.post('/api/mobile/v1/diagnostics', base_url='https://localhost', json=entries)
        self.assertIn(browser.status_code, (400, 401, 403, 404))

    def test_feedback_delete_is_owner_only_even_for_administrators(self):
        feedback_dir = tempfile.TemporaryDirectory(prefix='mobile-feedback-')
        self.addCleanup(feedback_dir.cleanup)
        patcher = mock.patch.dict(target.app.config, FEEDBACK_DIR=feedback_dir.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.is_admin = True
            mine = target.Feedback(user_id=self.user_id, message='mine')
            theirs = target.Feedback(user_id=self.other_id, message='theirs')
            target.db.session.add_all([mine, theirs])
            target.db.session.commit()
            mine_id, theirs_id = mine.public_id, theirs.public_id
        for fid in (mine_id, theirs_id):
            os.makedirs(os.path.join(feedback_dir.name, fid))
        token = self.token()
        self.assertEqual(self.call(f'/api/feedback/{theirs_id}', token, 'DELETE').status_code, 404)
        self.assertEqual(self.call(f'/api/feedback/{mine_id}', token, 'DELETE').status_code, 200)
        self.assertEqual(os.listdir(feedback_dir.name), [theirs_id])

    def test_secrets_export_needs_reauth_and_returns_only_user_keys(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.set_password('correct-horse-1')
            user.openai_api_key = target.encrypt_val('sk-user-key')
            target.db.session.commit()
        token = self.token()
        blocked = self.call('/api/mobile/v1/secrets/export', token, 'POST', json={})
        self.assertEqual(blocked.status_code, 403)
        self.assertEqual(blocked.json['error'], 'reauth_required')
        self.assertEqual(self.call('/api/mobile/v1/reauth', token, 'POST', json={'method': 'password', 'password': 'correct-horse-1'}).status_code, 200)
        with mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'env-operator-key', 'OPENAI_API_KEY': 'env-openai-key'}):
            exported = self.call('/api/mobile/v1/secrets/export', token, 'POST', json={})
        self.assertEqual(exported.status_code, 200)
        self.assertIn('no-store', exported.headers.get('Cache-Control', ''))
        self.assertEqual(exported.json['secrets']['openai_key'], 'sk-user-key')
        self.assertEqual(exported.json['secrets']['gemini_key'], '')
        self.assertNotIn('env-operator-key', json.dumps(exported.json))
        config = self.native.get('/api/mobile/v1/config', base_url='https://localhost').json
        self.assertEqual(config['sync_api_version'], 1)
        self.assertTrue(config['secrets_export'])

    def test_serverless_defaults_asset_matches_the_server(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(root, 'android', 'ci', 'sync-serverless-defaults.py')
        import importlib.util
        spec = importlib.util.spec_from_file_location('sync_serverless_defaults', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with open(module.ASSET, encoding='utf-8') as handle:
            self.assertEqual(handle.read(), module.render(module.build()),
                             'run android/ci/sync-serverless-defaults.py after changing models or notices')


if __name__ == '__main__':
    unittest.main()
