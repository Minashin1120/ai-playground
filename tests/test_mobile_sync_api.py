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
        with target.app.test_request_context(base_url='https://localhost'):
            user = target.db.session.get(target.User, self.user_id)
            return target._mobile_issue_token(user, 'Sync Android')

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

    def test_history_uuid_is_null_for_server_messages_and_other_accounts_are_denied(self):
        token = self.token()
        with target.app.app_context():
            mine = target.Thread(user_id=self.user_id, public_id=target.generate_thread_public_id(), title='Mine')
            other = target.Thread(user_id=self.other_id, public_id=target.generate_thread_public_id(), title='Other')
            target.db.session.add_all([mine, other])
            target.db.session.flush()
            target.db.session.add(target.Message(thread_id=mine.id, role='user', content='server message', is_encrypted=False))
            target.db.session.commit()
            mine_id, other_id = mine.public_id, other.public_id
        rows = self.call(f'/api/threads/{mine_id}', token).json['messages']
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]['client_uuid'])
        self.assertEqual(self.call(f'/api/threads/{other_id}', token).status_code, 403)

    def test_search_completion_survives_sync_without_reasoning_text(self):
        token = self.token()
        response = self.push_chat(token, self.uid(), [
            {'client_uuid': self.uid(), 'role': 'assistant', 'content': 'answer', 'search_used': True},
        ])
        self.assertEqual(response.status_code, 200)
        public_id = response.json['threads'][0]['id']
        message = self.call(f'/api/threads/{public_id}', token).json['messages'][0]
        self.assertEqual(json.loads(message['thought_data']), {'search_used': True})

    def test_push_creates_chats_idempotently_with_parent_links(self):
        token = self.token()
        thread_uuid, first, second = self.uid(), self.uid(), self.uid()
        messages = [
            {'client_uuid': first, 'role': 'user', 'content': 'こんにちは', 'model': 'gemini-3.6-flash', 'created_at_ms': 1_700_000_000_000},
            {'client_uuid': second, 'role': 'assistant', 'content': '回答', 'thought': '考え', 'search_used': True, 'model': 'gemini-3.6-flash',
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
        self.assertEqual({m['client_uuid']: m['id'] for m in thread['messages']}, ids)
        newest = self.call(f'/api/threads/{public_id}?limit=1', token).json['messages']
        self.assertEqual([(m['id'], m['client_uuid']) for m in newest], [(ids[second], second)])
        older = self.call(f'/api/threads/{public_id}?limit=1&before_id={ids[second]}&include_meta=0', token).json['messages']
        self.assertEqual([(m['id'], m['client_uuid']) for m in older], [(ids[first], first)])
        self.assertEqual(thread['messages'][1]['parent_id'], ids[first])
        self.assertEqual(json.loads(thread['messages'][1]['thought_data']), {'text': '考え', 'search_used': True})
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

    def test_delete_branch_with_children_obeys_immediate_parent_constraint(self):
        token = self.token()
        first, second, third = self.uid(), self.uid(), self.uid()
        pushed = self.push_chat(token, self.uid(), [
            {'client_uuid': first, 'role': 'user', 'content': 'keep', 'created_at_ms': 1_700_000_000_000},
            {'client_uuid': second, 'role': 'assistant', 'content': 'delete', 'parent': {'client_uuid': first}, 'created_at_ms': 1_700_000_001_000},
            {'client_uuid': third, 'role': 'user', 'content': 'child', 'parent': {'client_uuid': second}, 'created_at_ms': 1_700_000_002_000},
        ]).json['threads'][0]
        ids = {m['client_uuid']: m['id'] for m in pushed['messages']}
        # Reproduce MySQL's per-row self-FK check, which SQLite normally defers
        # until the whole DELETE statement has finished.
        with target.app.app_context():
            target.db.session.execute(target.db.text("""
                CREATE TRIGGER immediate_message_parent BEFORE DELETE ON message
                WHEN EXISTS (SELECT 1 FROM message WHERE parent_id = OLD.id)
                BEGIN SELECT RAISE(ABORT, 'message still has children'); END
            """))
            target.db.session.commit()
        try:
            response = self.call(f'/api/messages/{ids[second]}', token, 'DELETE')
            self.assertEqual(response.status_code, 200)
            with target.app.app_context():
                self.assertIsNotNone(target.db.session.get(target.Message, ids[first]))
                for key in (second, third):
                    self.assertIsNone(target.db.session.get(target.Message, ids[key]))
                    self.assertIsNone(target.SyncMessageRef.query.filter_by(client_uuid=key).first())
        finally:
            with target.app.app_context():
                target.db.session.rollback()
                target.db.session.execute(target.db.text('DROP TRIGGER IF EXISTS immediate_message_parent'))
                target.db.session.commit()

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

    def test_feedback_management_is_owner_only_except_for_administrators(self):
        # Administrators manage all feedback (status, reply, delete) in the app as on Web.
        feedback_dir = tempfile.TemporaryDirectory(prefix='mobile-feedback-')
        self.addCleanup(feedback_dir.cleanup)
        patcher = mock.patch.dict(target.app.config, FEEDBACK_DIR=feedback_dir.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        with target.app.app_context():
            mine = target.Feedback(user_id=self.user_id, message='mine')
            theirs = target.Feedback(user_id=self.other_id, message='theirs')
            other = target.Feedback(user_id=self.other_id, message='other')
            target.db.session.add_all([mine, theirs, other])
            target.db.session.commit()
            mine_id, theirs_id, other_id = mine.public_id, theirs.public_id, other.public_id
        for fid in (mine_id, theirs_id, other_id):
            os.makedirs(os.path.join(feedback_dir.name, fid))
        token = self.token()
        # A user sees and deletes only their own, and cannot answer feedback.
        listed = self.call('/api/feedback?all=1', token).json
        self.assertFalse(listed['is_admin'])
        self.assertEqual([item['public_id'] for item in listed['items']], [mine_id])
        self.assertEqual(self.call(f'/api/feedback/{theirs_id}', token, 'DELETE').status_code, 404)
        self.assertEqual(self.call(f'/api/feedback/{theirs_id}/update', token, 'POST',
                                   json={'status': 'resolved'}).status_code, 403)
        with target.app.app_context():
            target.db.session.get(target.User, self.user_id).is_admin = True
            target.db.session.commit()
        listed = self.call('/api/feedback?all=1', token).json
        self.assertTrue(listed['is_admin'])
        self.assertEqual({item['public_id'] for item in listed['items']}, {mine_id, theirs_id, other_id})
        updated = self.call(f'/api/feedback/{theirs_id}/update', token, 'POST',
                            json={'status': 'replied', 'admin_reply': '確認しました。'})
        self.assertEqual(updated.status_code, 200)
        with target.app.app_context():
            row = target.Feedback.query.filter_by(public_id=theirs_id).one()
            self.assertEqual((row.status, row.admin_reply, row.handled_by), ('replied', '確認しました。', 'sync-owner'))
        self.assertEqual(self.call(f'/api/feedback/{other_id}', token, 'DELETE').status_code, 200)
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


    def branch_fixture(self, temporary=False):
        from datetime import datetime, timedelta
        from pathlib import Path
        with target.app.app_context():
            thread = target.Thread(user_id=self.user_id, public_id=target.generate_thread_public_id(),
                                   title='Source', is_temporary=temporary, custom_instruction='Keep this')
            target.db.session.add(thread)
            target.db.session.flush()
            start = datetime.utcnow() - timedelta(minutes=1)
            attachment = f'{self.user_id}/attached.txt'
            generated = f'{self.user_id}/generated.txt'
            missing = f'{self.user_id}/missing.png'
            rows = []
            for index, (role, parent, content) in enumerate([
                ('user', None, 'root'), ('assistant', 0, f'[document](/files/{generated})'),
                ('assistant', 0, 'alternate'), ('user', 1, 'selected'),
                ('user', 2, 'later alternate'), ('assistant', 3, 'later selected'),
            ]):
                row = target.Message(thread_id=thread.id, role=role, content=content,
                                     parent_id=rows[parent].id if parent is not None else None,
                                     timestamp=start + timedelta(seconds=index),
                                     image_url=json.dumps([attachment]) if index == 0 else json.dumps([missing]) if index == 2 else None,
                                     prompt_options=json.dumps({'enable_search': True}) if index == 3 else None)
                target.db.session.add(row)
                target.db.session.flush()
                rows.append(row)
            # Same time as the selected message, but created after it: excluded.
            target.db.session.add(target.Message(thread_id=thread.id, role='assistant', content='later tie',
                parent_id=rows[0].id, timestamp=rows[3].timestamp))
            target.db.session.commit()
            public_id, selected = thread.public_id, rows[3].id
        for ref in [attachment, generated]:
            file = Path(self.upload_dir, ref)
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('synthetic branch attachment')
        return public_id, selected, attachment, generated, missing

    def test_branch_copies_all_routes_up_to_timestamp_and_selected_id(self):
        public_id, selected, attachment, generated, missing = self.branch_fixture()
        token = self.token()
        body = {'message_id': selected, 'request_id': self.uid().replace('-', '')}
        response = self.call(f'/api/threads/{public_id}/branch', token, 'POST', json=body)
        self.assertEqual(response.status_code, 201, response.get_json())
        branch = response.get_json()
        replay = self.call(f'/api/threads/{public_id}/branch', token, 'POST', json=body)
        self.assertEqual(replay.get_json()['id'], branch['id'])
        payload = self.call('/api/threads/' + branch['id'], token).get_json()
        self.assertEqual(len(payload['messages']), 4)
        by_content = {row['content']: row for row in payload['messages']}
        root = by_content['root']
        self.assertEqual(by_content['alternate']['parent_id'], root['id'])
        self.assertEqual(by_content['selected']['id'], branch['leaf_id'])
        self.assertEqual(by_content['selected']['prompt_options'], {'enable_search': True})
        self.assertEqual(by_content['alternate']['missing_files'], [missing])
        self.assertEqual(payload['custom_instruction'], 'Keep this')
        self.assertTrue(payload['branch_origin']['available'])
        self.assertEqual(payload['branch_origin']['message_id'], selected)
        # Deleting either owner preserves references from all routes, including
        # generated links in content and routes not initially on screen.
        from pathlib import Path
        self.assertEqual(self.call('/api/threads/' + public_id, token, 'DELETE').status_code, 200)
        for ref in [attachment, generated]:
            self.assertTrue(Path(self.upload_dir, ref).is_file())
        payload = self.call('/api/threads/' + branch['id'], token).get_json()
        replay = self.call(f'/api/threads/{public_id}/branch', token, 'POST', json=body)
        self.assertEqual(replay.get_json()['id'], branch['id'])
        self.assertFalse(payload['branch_origin']['available'])
        self.assertEqual(self.call('/api/threads/' + branch['id'], token, 'DELETE').status_code, 200)
        for ref in [attachment, generated]:
            self.assertFalse(Path(self.upload_dir, ref).exists())

    def test_branch_temporary_chat_requires_confirmation_and_survives_expiry(self):
        public_id, selected, attachment, generated, missing = self.branch_fixture(temporary=True)
        token = self.token()
        body = {'message_id': selected, 'request_id': self.uid().replace('-', '')}
        response = self.call(f'/api/threads/{public_id}/branch', token, 'POST', json=body)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()['code'], 'confirmation_required')
        body['confirm_persist_temporary'] = True
        created = self.call(f'/api/threads/{public_id}/branch', token, 'POST', json=body)
        self.assertEqual(created.status_code, 201, created.get_json())
        branch = created.get_json()['id']
        with target.app.app_context():
            original = target.resolve_thread_for_user(public_id, self.user_id)
            target._track_temp_chat_uploaded_refs(original, self.user_id, [attachment, generated])
            self.assertTrue(target._cleanup_temp_chat_member(target._temp_chat_member(original), force=True))
        from pathlib import Path
        self.assertTrue(Path(self.upload_dir, attachment).exists())
        self.assertTrue(Path(self.upload_dir, generated).exists())
        payload = self.call('/api/threads/' + branch, token).get_json()
        self.assertFalse(payload['is_temporary'])
        self.assertEqual(len(payload['messages']), 4)

    def test_branch_checks_owner_message_and_csrf(self):
        public_id, selected, *_ = self.branch_fixture()
        token = self.token()
        with target.app.app_context():
            other = target.Thread(user_id=self.other_id, public_id=target.generate_thread_public_id())
            target.db.session.add(other)
            target.db.session.commit()
            other_id = other.public_id
        body = {'message_id': selected, 'request_id': self.uid().replace('-', '')}
        self.assertEqual(self.call(f'/api/threads/{other_id}/branch', token, 'POST', json=body).status_code, 403)
        body['message_id'] = -1
        self.assertEqual(self.call(f'/api/threads/{public_id}/branch', token, 'POST', json=body).status_code, 400)
        self.assertEqual(self.browser.post(f'/api/threads/{public_id}/branch', json=body).status_code, 403)

    def test_branch_missing_file_references_survive_sync(self):
        token = self.token()
        missing = f'{self.user_id}/gone.pdf'
        result = self.push_chat(token, self.uid(), [
            {'client_uuid': self.uid(), 'role': 'user', 'content': 'missing', 'files': [missing], 'missing_files': [missing]},
            {'client_uuid': self.uid(), 'role': 'assistant', 'content': '[file](local/abcdef1234)',
             'missing_local_files': [{'reference': 'local/abcdef1234', 'attached': True}]},
        ]).get_json()['threads'][0]
        self.assertFalse(result.get('rejected'))
        rows = self.call('/api/threads/' + result['id'], token).get_json()['messages']
        self.assertEqual(rows[0]['missing_files'], [missing])
        self.assertEqual(rows[1]['missing_files'], [f'{self.user_id}/missing_abcdef1234'])
        self.assertIn('/files/', rows[1]['content'])

    def test_branch_preserves_encrypted_payloads_and_exported_origin(self):
        public_id, selected, attachment, generated, missing = self.branch_fixture()
        with target.app.app_context():
            thread = target.resolve_thread_for_user(public_id, self.user_id)
            row = target.Message.query.filter_by(thread_id=thread.id, role='assistant').order_by(target.Message.id).first()
            row.content = target.encrypt_val(row.content)
            row.thought_data = target.encrypt_val('{"text":"reasoning"}')
            row.is_encrypted = True
            raw_content = row.content
            target.db.session.commit()
        token = self.token()
        created = self.call(f'/api/threads/{public_id}/branch', token, 'POST', json={
            'message_id': selected, 'request_id': self.uid().replace('-', '')}).get_json()
        with target.app.app_context():
            branch = target.resolve_thread_for_user(created['id'], self.user_id)
            clone = target.Message.query.filter_by(thread_id=branch.id, role='assistant').order_by(target.Message.id).first()
            self.assertTrue(clone.is_encrypted)
            self.assertEqual(clone.content, raw_content)
            self.assertEqual(target.decrypt_val(clone.thought_data), '{"text":"reasoning"}')
            exported = next(t for t in target._account_export_threads(self.user_id) if t['public_id'] == branch.public_id)
            self.assertEqual(exported['branch_origin']['thread_id'], public_id)
            self.assertEqual(exported['branch_origin']['message_id'], selected)

    def test_branch_creation_failure_rolls_back_every_clone(self):
        public_id, selected, *_ = self.branch_fixture()
        token = self.token()
        with mock.patch.object(target, 'safe_db_commit', side_effect=RuntimeError('synthetic commit failure')):
            with self.assertRaises(RuntimeError):
                self.call(f'/api/threads/{public_id}/branch', token, 'POST', json={
                    'message_id': selected, 'request_id': self.uid().replace('-', '')})
        with target.app.app_context():
            self.assertEqual(target.Thread.query.filter_by(user_id=self.user_id).count(), 1)
            self.assertEqual(target.ThreadBranch.query.count(), 0)
            self.assertEqual(target.Message.query.count(), 7)

    def test_branch_files_survive_message_deletion_and_cleanup_rollback(self):
        public_id, selected, attachment, generated, _ = self.branch_fixture()
        token = self.token()
        created = self.call(f'/api/threads/{public_id}/branch', token, 'POST', json={
            'message_id': selected, 'request_id': self.uid().replace('-', '')}).get_json()
        source_rows = self.call('/api/threads/' + public_id, token).get_json()['messages']
        self.assertEqual(self.call('/api/messages/' + str(source_rows[0]['id']), token, 'DELETE').status_code, 200)
        from pathlib import Path
        for ref in [attachment, generated]:
            self.assertTrue(Path(self.upload_dir, ref).exists())
        with target.app.app_context():
            branch = target.resolve_thread_for_user(created['id'], self.user_id)
            target._delete_unreferenced_upload_refs(self.user_id, [attachment, generated], exclude_threads=[branch.id])
            target.db.session.rollback()
        for ref in [attachment, generated]:
            self.assertTrue(Path(self.upload_dir, ref).exists())


    def test_branch_file_links_remap_on_account_import(self):
        with target.app.app_context():
            mapping = {'900/a.pdf': f'{self.user_id}/imported.pdf'}
            content = target._rewrite_imported_file_links('[file](/files/900/a.pdf) ![](/files/900/gone.png)', mapping, self.user_id)
            self.assertIn(f'/files/{self.user_id}/imported.pdf', content)
            self.assertIn(f'/files/{self.user_id}/missing_import_', content)
            self.assertNotIn('/files/900/', content)


    def test_branch_shared_file_collector_accepts_legacy_relative_urls(self):
        with target.app.app_context():
            self.assertEqual(target._message_file_refs({'image_url': '["a.pdf"]', 'content': '<audio src="/files/b.wav"></audio>'}, self.user_id),
                             {f'{self.user_id}/a.pdf', f'{self.user_id}/b.wav'})


    def test_branch_shared_file_collector_skips_content_no_key_opens(self):
        from cryptography.fernet import Fernet
        lost = Fernet(Fernet.generate_key()).encrypt(b'![](/files/x.png)').decode()
        with target.app.app_context():
            self.assertEqual(target._message_file_refs({'image_url': '["a.pdf"]', 'content': lost, 'is_encrypted': True}, self.user_id),
                             {f'{self.user_id}/a.pdf'})


    def test_server_diagnostics_are_account_opt_in_and_shared_between_clients(self):
        token = self.token()
        before = self.call('/api/mobile/v1/preferences', token)
        self.assertFalse(before.json['enable_server_diagnostic_log'])
        with target.app.app_context():
            target._record_server_diagnostic(self.user_id, 'server.worker.start')
            self.assertEqual(target._recent_server_diagnostics(self.user_id), [])
        enabled = self.call('/api/mobile/v1/preferences', token, 'PUT', json={'enable_server_diagnostic_log': True})
        self.assertEqual(enabled.status_code, 200)
        self.assertTrue(enabled.json['enable_server_diagnostic_log'])
        web = self.browser.get('/api/settings', base_url='https://localhost')
        self.assertTrue(web.json['enable_server_diagnostic_log'])
        with target.app.app_context():
            target._record_server_diagnostic(self.user_id, 'server.worker.start', job='a' * 32,
                title='PRIVATE-TITLE', message='PRIVATE-PROMPT', ms=42)
            target._record_server_diagnostic(self.other_id, 'server.worker.start', ms=99)
            rows = target._recent_server_diagnostics(self.user_id)
            self.assertTrue(any(e.get('ms') == 42 for e in rows))
            self.assertNotIn('PRIVATE', json.dumps(rows))
            self.assertEqual(target._recent_server_diagnostics(self.other_id), [])
        self.call('/api/mobile/v1/preferences', token, 'PUT', json={'enable_server_diagnostic_log': True})
        with target.app.app_context():
            self.assertTrue(any(e.get('ms') == 42 for e in target._recent_server_diagnostics(self.user_id)))
        invalid = self.call('/api/mobile/v1/preferences', token, 'PUT', json={'enable_server_diagnostic_log': 'false'})
        self.assertEqual(invalid.status_code, 400)
        disabled = self.call('/api/mobile/v1/preferences', token, 'PUT', json={'enable_server_diagnostic_log': False})
        self.assertFalse(disabled.json['enable_server_diagnostic_log'])
        self.assertFalse(self.browser.get('/api/settings', base_url='https://localhost').json['enable_server_diagnostic_log'])
        with target.app.app_context():
            self.assertEqual(target._recent_server_diagnostics(self.user_id), [])
        self.assertFalse(list(self.redis.scan_iter('diagnostic:v2:*:*:*')))

    def test_server_diagnostic_retention_is_shared_validated_and_applied(self):
        token = self.token()
        for value in [0, -1, 8761, 1.5, True, "2", None]:
            response = self.call('/api/mobile/v1/preferences', token, 'PUT',
                json={'server_diagnostic_log_retention_hours': value})
            self.assertEqual(response.status_code, 400)
            response = self.browser.post('/api/settings', base_url='https://localhost', headers={'X-CSRF-Token': 'browser-csrf'},
                json={'server_diagnostic_log_retention_hours': value})
            self.assertEqual(response.status_code, 400)
        response = self.call('/api/mobile/v1/preferences', token, 'PUT',
            json={'enable_server_diagnostic_log': True, 'server_diagnostic_log_retention_hours': 4})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.browser.get('/api/settings', base_url='https://localhost').json['server_diagnostic_log_retention_hours'], 4)
        with target.app.app_context():
            target._record_server_diagnostic(self.user_id, 'server.worker.start', ms=123)
            keys = list(self.redis.scan_iter('diagnostic:v2:*:*:*'))
            self.assertTrue(keys)
            self.assertTrue(all(14390 <= self.redis.ttl(key) <= 14400 for key in keys))
        self.browser.post('/api/settings', base_url='https://localhost', headers={'X-CSRF-Token': 'browser-csrf'}, json={'server_diagnostic_log_retention_hours': 1})
        self.assertEqual(self.call('/api/mobile/v1/preferences', token).json['server_diagnostic_log_retention_hours'], 1)
        self.assertTrue(all(0 < self.redis.ttl(key) <= 3600 for key in keys))
        self.call('/api/mobile/v1/preferences', token, 'PUT', json={'server_diagnostic_log_retention_hours': 8})
        self.assertTrue(all(0 < self.redis.ttl(key) <= 3600 for key in keys))

    def test_server_diagnostic_event_expiry_does_not_slide_and_snapshot_survives(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.enable_server_diagnostic_log = True
            target.db.session.commit()
            target._record_server_diagnostic(self.user_id, 'server.worker.start', op='a' * 32)
            event_keys = list(self.redis.scan_iter('diagnostic:v2:*:*:*'))
            self.assertEqual(len(event_keys), 1)
            first = event_keys[0]
            self.redis.expire(first, 20)
            target._recent_server_diagnostics(self.user_id)
            target._record_server_diagnostic(self.user_id, 'server.worker.end', op='a' * 32)
            self.assertLessEqual(self.redis.ttl(first), 20)
            self.assertGreater(self.redis.ttl(first), 0)
            snapshot = target._recent_server_diagnostics(self.user_id)
            self.assertEqual(len(snapshot), 2)
            for key in self.redis.scan_iter('diagnostic:v2:*:*:*'):
                self.redis.delete(key)
            self.assertEqual(target._recent_server_diagnostics(self.user_id), [])
            self.assertEqual(len(snapshot), 2)

    def test_server_logs_attach_without_device_logs_only_when_requested(self):
        feedback_dir = tempfile.TemporaryDirectory(prefix='server-feedback-test-')
        self.addCleanup(feedback_dir.cleanup)
        patcher = mock.patch.dict(target.app.config, FEEDBACK_DIR=feedback_dir.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        token = self.token()
        self.call('/api/mobile/v1/preferences', token, 'PUT', json={'enable_server_diagnostic_log': True})
        with target.app.app_context():
            target._record_server_diagnostic(self.user_id, 'server.worker.start', job='job_1792500000_1_abcdef0123456789',
                title='PRIVATE-TITLE', message='PRIVATE-CONTENT')
        sent = self.call('/api/feedback', token, 'POST', json={'message': 'synthetic report', 'client_logs': {
            'client': 'android', 'version': '1.70.0', 'entries': []}})
        self.assertEqual(sent.status_code, 200, sent.json)
        self.assertTrue(sent.json['logs_saved'])
        self.assertGreater(sent.json['logs_count'], 0)
        path = os.path.join(feedback_dir.name, sent.json['public_id'], 'logs', 'activity.jsonl')
        with open(path, encoding='utf-8') as handle:
            saved = handle.read()
        self.assertNotIn('PRIVATE', saved)
        self.assertEqual(len(saved.splitlines()) - 1, sent.json['logs_count'])
        reply = self.call('/api/feedback/' + sent.json['public_id'] + '/replies', token, 'POST', json={'message': 'without logs'})
        self.assertNotIn('logs_saved', reply.json)
        with target.app.app_context():
            target._clear_server_diagnostics(self.user_id)
        with open(path, encoding='utf-8') as handle:
            self.assertEqual(handle.read(), saved)

if __name__ == '__main__':
    unittest.main()
