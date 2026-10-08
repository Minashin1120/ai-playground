"""Feedback "現在開いているチャットのコピーを送信する": everything the server can find from the chat's id
(server/feedback_chat_copy.py), plus what the client adds."""
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock


class _FakeRedis:
    def __init__(self, values):
        self.values = {k.encode(): v.encode() for k, v in values.items()}

    def get(self, key):
        return self.values.get(key.encode() if isinstance(key, str) else key)

    def scan_iter(self, count=None):
        return iter(list(self.values))

    def type(self, key):
        return b'string'

    def ttl(self, key):
        return -1


class FeedbackChatCopyTests(unittest.TestCase):
    def setUp(self):
        import app as target
        self.target = target
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=['localhost'])
        target._ensure_temp_chat_monitor_running = lambda: None
        dirs = [tempfile.TemporaryDirectory(prefix=p) for p in ('feedback-logs-', 'feedback-uploads-', 'feedback-root-')]
        for d in dirs:
            self.addCleanup(d.cleanup)
        self.log_dir, self.upload_dir, self.root = (d.name for d in dirs)
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            user = target.User(username='feedback-chat-test', is_setup_completed=True)
            user.set_password('test-password')
            target.db.session.add(user)
            target.db.session.commit()
            self.user_id = user.id
        self.redis = _FakeRedis({})
        self.journal = mock.Mock(return_value=iter([{'kind': 'log', 'source': 'journal', 'line': 'stub'}]))
        for patcher in [
            mock.patch.object(target, '_bot_turnstile_active', return_value=False),
            mock.patch.object(target, 'rate_limit', return_value=True),
            mock.patch.object(target, 'redis_conn', self.redis),
            mock.patch.object(target, '_feedback_chat_journal_rows', self.journal),
            mock.patch.object(target, '_FEEDBACK_CHAT_MIN_FREE_BYTES', 0),
            mock.patch.dict(target.app.config, FEEDBACK_DIR=self.log_dir, UPLOAD_FOLDER=self.upload_dir,
                            FEEDBACK_CHAT_LOG_ROOT=self.root,
                            ANDROID_DIAGNOSTICS_LOG=os.path.join(self.root, 'diagnostics', 'android-diagnostics.log')),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def client(self):
        client = self.target.app.test_client()
        with client.session_transaction() as sess:
            sess['_user_id'] = str(self.user_id)
            sess['_fresh'] = True
            sess['csrf_token'] = 'csrf-test-token'
        return client

    def post(self, payload):
        return self.client().post('/api/feedback', json=payload, headers={'X-CSRF-Token': 'csrf-test-token'},
                                  base_url='https://localhost')

    def make_thread(self):
        target = self.target
        user_dir = os.path.join(self.upload_dir, str(self.user_id))
        os.makedirs(user_dir)
        with open(os.path.join(user_dir, 'photo.png.enc'), 'wb') as handle:
            handle.write(target.encrypt_bytes(b'png-bytes'))
        with open(os.path.join(user_dir, 'report.txt'), 'wb') as handle:
            handle.write(b'plain report')
        with target.app.app_context():
            thread = target.Thread(user_id=self.user_id, public_id='feedback-chat-thread-public-id', title='T',
                                   custom_instruction='be brief', last_gem_uuid='gem-1')
            target.db.session.add(thread)
            target.db.session.flush()
            target.db.session.add(target.Message(
                thread_id=thread.id, role='user', is_encrypted=True, content=target.encrypt_val('secret question'),
                image_url=json.dumps([f'{self.user_id}/photo.png', '999/not-mine.png', 'gone.png']),
                timestamp=datetime.utcnow() - timedelta(minutes=5)))
            target.db.session.add(target.Message(
                thread_id=thread.id, role='assistant', is_encrypted=True, model='m',
                content=target.encrypt_val('see [report](/files/report.txt)'),
                thought_data=target.encrypt_val('thinking'), timestamp=datetime.utcnow()))
            target.db.session.add(target.Gem(uuid='gem-1', user_id=self.user_id, name='G', instruction='gem rules'))
            target.db.session.add(target.ChatLatencyTrace(user_id=self.user_id, thread_public_id=thread.public_id,
                                                          job_id='job-abcdef123', model='m'))
            target.db.session.add(target.FileCache(user_id=self.user_id, rel_path=f'{self.user_id}/photo.png',
                                                   provider='gemini', state='failed', last_error='upload refused'))
            target.db.session.commit()
            self.thread_db_id = thread.id
        self.redis.values = {k.encode(): v.encode() for k, v in {
            f'pending_job:{self.user_id}:{self.thread_db_id}': json.dumps({'job_id': 'job-pending-99'}),
            'stream_acc:job-pending-99:error': 'provider failed',
            'stream_acc:job-other-77:error': 'someone else',
        }.items()}
        with open(os.path.join(self.root, 'debug.log'), 'w') as handle:
            handle.write('unrelated\n'
                         f'ERROR job-abcdef123 failed\n'
                         f'Sandbox image reference resolution failed for thread {self.thread_db_id}: x\n'
                         f'other thread {self.thread_db_id}9\n')
        # Another user's feedback log never belongs to this chat and is not searched.
        os.makedirs(os.path.join(self.log_dir, '424242', 'logs'))
        with open(os.path.join(self.log_dir, '424242', 'logs', 'activity.jsonl'), 'w') as handle:
            handle.write('{"kind":"meta"}\n{"ev":"fetch","path":"/c/feedback-chat-thread-public-id"}\n')
        with open(os.path.join(self.root, 'access.log'), 'w') as handle:
            handle.write('GET /c/feedback-chat-thread-public-id 200\nGET /c/other 200\n')
        return 'feedback-chat-thread-public-id'

    def copy_paths(self):
        return [p for p in (os.path.join(self.log_dir, d, 'chat.jsonl') for d in os.listdir(self.log_dir))
                if os.path.isfile(p)]

    def read_copy(self):
        paths = self.copy_paths()
        self.assertEqual(len(paths), 1, os.listdir(self.log_dir))
        path = paths[0]
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        with open(path, encoding='utf-8') as handle:
            return path, [json.loads(line) for line in handle]

    def test_copy_has_decrypted_chat_attachments_records_and_logs(self):
        thread_id = self.make_thread()
        self.assertEqual(self.post({'message': 'm', 'chat_copy': 'x'}).status_code, 400)
        response = self.post({'message': 'with chat', 'chat_copy': {
            'client': 'web', 'version': 'V1', 'thread_id': thread_id,
            'client_state': {'local_storage': {f'fixed_branch_{thread_id}': '12'}}}})
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertIs(body['chat_copy_saved'], True)
        self.assertRegex(body['public_id'], r'^[0-9a-f]{12}$')
        path, rows = self.read_copy()
        self.assertEqual(path, os.path.join(self.log_dir, body['public_id'], 'chat.jsonl'))
        by_kind = {}
        for row in rows:
            by_kind.setdefault(row['kind'], []).append(row)
        meta = rows[0]
        self.assertEqual((meta['kind'], meta['source'], meta['thread_id'], meta['rows']['message']),
                         ('meta', 'server', thread_id, 2))
        self.assertEqual(by_kind['thread'][0]['custom_instruction'], 'be brief')
        self.assertEqual([m['content'] for m in by_kind['message']], ['secret question', 'see [report](/files/report.txt)'])
        self.assertEqual(by_kind['message'][1]['thought_data'], 'thinking')
        self.assertEqual(by_kind['gem'][0]['instruction'], 'gem rules')
        self.assertEqual(by_kind['latency_trace'][0]['job_id'], 'job-abcdef123')
        self.assertEqual(by_kind['file_cache'][0]['last_error'], 'upload refused')
        self.assertEqual(by_kind['client_state'][0]['value']['local_storage'], {f'fixed_branch_{thread_id}': '12'})
        self.assertEqual(sorted(r['key'] for r in by_kind['redis']),
                         [f'pending_job:{self.user_id}:{self.thread_db_id}', 'stream_acc:job-pending-99:error'])

        files = {row['ref']: row for row in by_kind['file']}
        self.assertEqual(set(files), {f'{self.user_id}/photo.png', f'{self.user_id}/gone.png', f'{self.user_id}/report.txt'})
        self.assertEqual(files[f'{self.user_id}/gone.png']['status'], 'missing')
        files_dir = os.path.join(os.path.dirname(path), 'chat.files')
        for ref, content in ((f'{self.user_id}/photo.png', b'png-bytes'), (f'{self.user_id}/report.txt', b'plain report')):
            self.assertEqual(files[ref]['status'], 'copied')
            with open(os.path.join(files_dir, files[ref]['file']), 'rb') as handle:
                self.assertEqual(handle.read(), content)
        self.assertIs(files[f'{self.user_id}/photo.png']['encrypted_at_rest'], True)

        logs = [(r['source'], r.get('line')) for r in by_kind['log']]
        self.assertIn(('debug.log', 'ERROR job-abcdef123 failed'), logs)
        self.assertIn(('debug.log', f'Sandbox image reference resolution failed for thread {self.thread_db_id}: x'), logs)
        self.assertIn(('access.log', f'GET /c/{thread_id} 200'), logs)
        self.assertIn(('journal', 'stub'), logs)
        self.assertEqual(len(logs), 4, logs)
        pattern = self.journal.call_args[0][0]
        self.assertIn('job\\-pending\\-99', pattern)
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(path), 'logs')))
        with open(os.path.join(os.path.dirname(path), 'feedback.json'), encoding='utf-8') as handle:
            info = json.load(handle)
        self.assertEqual((info['message'], info['client'], info['chat_copy_saved']), ('with chat', 'web', True))

    def test_attachment_limit_and_another_users_chat(self):
        thread_id = self.make_thread()
        with mock.patch.object(self.target, '_FEEDBACK_CHAT_FILES_MAX_BYTES', 12):
            self.post({'message': 'm', 'chat_copy': {'client': 'web', 'thread_id': thread_id}})
        _, rows = self.read_copy()
        statuses = sorted(r['status'] for r in rows if r['kind'] == 'file')
        self.assertEqual(statuses, ['copied', 'missing', 'skipped_copy_limit'])
        target = self.target
        with target.app.app_context():
            other = target.User(username='feedback-other', is_setup_completed=True)
            other.set_password('test-password')
            target.db.session.add(other)
            target.db.session.flush()
            target.db.session.add(target.Thread(user_id=other.id, public_id='not-mine', title='X'))
            target.db.session.commit()
        response = self.post({'message': 'm', 'chat_copy': {'client': 'web', 'thread_id': 'not-mine'}})
        self.assertIs(response.get_json()['chat_copy_saved'], False)

    def test_device_chat_and_device_files(self):
        device = {'title': 'On device', 'messages': [{'id': 1, 'role': 'user', 'content': 'hello',
                                                      'image_url': '["local/abc.png"]'}, 'bad']}
        response = self.post({'message': 'm', 'chat_copy': {
            'client': 'android', 'thread_id': 'l_1', 'thread': device, 'device_files': 1,
            'diagnostics': [{'ev': 'chat.load', 'thread': 'l_1'}]}})
        body = response.get_json()
        self.assertIs(body['chat_copy_saved'], True)
        paths = self.copy_paths()
        self.assertEqual(len(paths), 1)
        upload = self.client().post(f"/api/feedback/{body['public_id']}/chat_files",
                                    data={'ref': 'local/abc.png', 'file': (io.BytesIO(b'device-bytes'), 'abc.png')},
                                    headers={'X-CSRF-Token': 'csrf-test-token'}, base_url='https://localhost')
        self.assertEqual(upload.status_code, 200, upload.get_json())
        self.assertIs(upload.get_json()['saved'], True)
        path = paths[0]
        with open(path, encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual((rows[0]['source'], rows[0]['thread_id'], rows[0]['device_files_expected']), ('device', 'l_1', 1))
        self.assertEqual([r['kind'] for r in rows[1:]],
                         ['client_thread', 'client_message', 'client_diagnostics', 'file'])
        self.assertEqual((rows[-1]['source'], rows[-1]['ref'], rows[-1]['status']), ('device', 'local/abc.png', 'copied'))
        with open(os.path.join(os.path.dirname(path), 'chat.files', rows[-1]['file']), 'rb') as handle:
            self.assertEqual(handle.read(), b'device-bytes')
        missing = self.client().post('/api/feedback/999999/chat_files', data={'file': (io.BytesIO(b'x'), 'x')},
                                     headers={'X-CSRF-Token': 'csrf-test-token'}, base_url='https://localhost')
        self.assertEqual(missing.status_code, 404)

    def test_account_deletion_removes_feedback_files(self):
        thread_id = self.make_thread()
        self.post({'message': 'm', 'chat_copy': {'client': 'web', 'thread_id': thread_id},
                   'client_logs': {'client': 'web', 'entries': [{'t': 1, 'ev': 'click'}]}})
        own = [d for d in os.listdir(self.log_dir) if d != '424242']
        self.assertEqual(len(own), 1)
        self.assertEqual(sorted(os.listdir(os.path.join(self.log_dir, own[0]))),
                         ['chat.files', 'chat.jsonl', 'feedback.json', 'logs'])
        target = self.target
        with target.app.app_context():
            target._delete_user_account_immediately(target.db.session.get(target.User, self.user_id))
        self.assertEqual(os.listdir(self.log_dir), ['424242'])

    def test_prune_keeps_the_newest_copies_within_the_size_limit(self):
        target = self.target
        paths = []
        for index in range(3):
            name = f'{index + 1:012x}'
            os.makedirs(os.path.join(self.log_dir, name))
            path = os.path.join(self.log_dir, name, 'chat.jsonl')
            with open(path, 'w') as handle:
                handle.write('x' * 100)
            os.utime(path, (1000 + index, 1000 + index))
            paths.append(path)
        with target.app.app_context(), mock.patch.object(target, '_FEEDBACK_CHAT_TOTAL_MAX_BYTES', 250):
            target._prune_feedback_chat_files(keep=paths[0])
        self.assertEqual([os.path.isfile(p) for p in paths], [True, False, True])


if __name__ == '__main__':
    unittest.main()
