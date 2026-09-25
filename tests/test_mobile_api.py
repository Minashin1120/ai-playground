"""Exercise native authentication with isolated SQL and real, ephemeral Redis."""
import json
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from unittest import mock

import redis
import app as target


@unittest.skipUnless(shutil.which('redis-server'), 'redis-server needed for atomic grant tests')
class MobileApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.redis_dir = tempfile.TemporaryDirectory(prefix='mobile-api-test-')
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
        self.redis.flushdb()  # Only this test's Unix-socket Redis, never configured REDIS_URL.
        upload_dir = tempfile.TemporaryDirectory(prefix='mobile-api-uploads-')
        self.addCleanup(upload_dir.cleanup)
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
            user = target.User(username='android-owner', is_setup_completed=True)
            other = target.User(username='another-owner', is_setup_completed=True)
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

    def device(self):
        response = self.native.post('/api/mobile/v1/device', base_url='https://localhost',
                                    json={'client_id': 'official-android', 'device_name': 'Test Android'})
        self.assertEqual(response.status_code, 200)
        return response.json

    def approve(self, grant, decision='approve'):
        return self.browser.post('/android/connect', base_url='https://localhost', data={
            'csrf_token': 'browser-csrf', 'user_code': grant['user_code'], 'decision': decision,
        })

    def poll(self, grant):
        return self.native.post('/api/mobile/v1/token', base_url='https://localhost',
                                json={'client_id': 'official-android', 'device_code': grant['device_code']})

    def token(self):
        grant = self.device()
        self.assertEqual(self.approve(grant).status_code, 200)
        response = self.poll(grant)
        self.assertEqual(response.status_code, 200)
        return response.json['access_token']

    def call(self, path, token, method='GET', **kwargs):
        return self.native.open(path, method=method, base_url='https://localhost',
                                headers={'Authorization': 'Bearer ' + token}, **kwargs)

    def pass_integrity_gate(self):
        # The Integrity/Turnstile gate itself is covered by the integrity tests.
        patcher = mock.patch.object(target, '_mobile_integrity_gate', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pairing_token_hash_scope_and_revocation(self):
        token = self.token()
        response = self.call('/api/mobile/v1/me', token)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['username'], 'android-owner')
        self.assertEqual(response.json['model_catalog_version'], 1)
        catalog = {item['id']: item for item in response.json['models']}
        self.assertEqual(catalog['gpt-5.6-sol']['provider_label'], 'OpenAI')
        self.assertTrue(catalog['gpt-5.6-sol']['selectable'])
        self.assertIn('thinking', catalog['gpt-5.6-sol']['capabilities'])
        self.assertEqual(catalog['gpt-image-2']['mode'], 'image')
        self.assertTrue(catalog['gpt-image-2']['selectable'])
        self.assertEqual(catalog['gpt-realtime-2']['mode'], 'realtime_audio')
        self.assertTrue(catalog['gpt-realtime-2']['selectable'])
        self.assertTrue(catalog['gpt-realtime-translate']['selectable'])
        self.assertFalse(catalog['gpt-realtime-whisper']['selectable'])
        self.assertIn('batch', catalog['gpt-5.6-sol']['capabilities'])
        self.assertTrue(catalog['gemini-3-pro-preview']['deprecated'])
        self.assertIn('no-store', response.headers['Cache-Control'])
        with target.app.app_context():
            row = target.UserSession.query.filter(target.UserSession.session_id.startswith('android:')).one()
            self.assertEqual(row.session_id, 'android:' + target._mobile_digest(token))
            self.assertNotIn(token, row.session_id)
        for path in ['/api/settings', '/api/sessions', '/logout', '/api/browser_fast_mode/bootstrap']:
            self.assertEqual(self.call(path, token).status_code, 403)
        self.assertEqual(self.call('/api/mobile/v1/revoke', token, 'POST', json={}).status_code, 200)
        self.assertEqual(self.call('/api/mobile/v1/me', token).status_code, 401)

    def test_cookie_only_me_and_invalid_token_are_rejected(self):
        self.assertEqual(self.browser.get('/api/mobile/v1/me', base_url='https://localhost').status_code, 401)
        self.assertEqual(self.call('/api/threads', target.MOBILE_TOKEN_PREFIX + 'A' * 43).status_code, 401)

    def test_native_thread_crud_and_cross_account_denial(self):
        token = self.token()
        created = self.call('/api/threads', token, 'POST', json={})
        self.assertEqual(created.status_code, 200)
        thread_id = created.json['id']
        self.assertEqual(self.call('/api/threads/' + thread_id + '?limit=50', token).status_code, 200)
        self.assertEqual(len(self.call('/api/threads', token).json['threads']), 1)
        with target.app.app_context():
            foreign = target.Thread(user_id=self.other_id, public_id=target.generate_thread_public_id())
            target.db.session.add(foreign)
            target.db.session.commit()
            foreign_id = foreign.public_id
        for method in ['GET', 'DELETE']:
            self.assertEqual(self.call('/api/threads/' + foreign_id, token, method).status_code, 403)
        self.assertEqual(self.call('/api/threads/' + thread_id, token, 'DELETE').status_code, 200)

    def test_native_message_delete_is_owner_scoped_and_removes_later_history(self):
        token = self.token()
        with target.app.app_context():
            thread = target.Thread(user_id=self.user_id, public_id=target.generate_thread_public_id())
            foreign = target.Thread(user_id=self.other_id, public_id=target.generate_thread_public_id())
            target.db.session.add_all([thread, foreign])
            target.db.session.commit()
            base = datetime.utcnow()
            first = target.Message(thread_id=thread.id, role='user', content='keep', timestamp=base)
            second = target.Message(thread_id=thread.id, role='user', content='drop', timestamp=base + timedelta(seconds=1))
            third = target.Message(thread_id=thread.id, role='assistant', content='drop too', timestamp=base + timedelta(seconds=2))
            other = target.Message(thread_id=foreign.id, role='user', content='not mine', timestamp=base)
            target.db.session.add_all([first, second, third, other])
            target.db.session.commit()
            thread_id, second_id, other_id = thread.id, second.id, other.id
        self.assertEqual(self.call(f'/api/messages/{other_id}', token, 'DELETE').status_code, 403)
        self.assertEqual(self.call(f'/api/messages/{second_id}', token, 'DELETE').status_code, 200)
        with target.app.app_context():
            remaining = [m.content for m in target.Message.query.filter_by(thread_id=thread_id).order_by(target.Message.timestamp)]
            self.assertEqual(remaining, ['keep'])
            self.assertIsNotNone(target.db.session.get(target.Message, other_id))
        # Without the bearer (and without a cookie) the native client cannot reach the route.
        unauthenticated = self.native.open(f'/api/messages/{other_id}', method='DELETE', base_url='https://localhost')
        self.assertIn(unauthenticated.status_code, (401, 403))

    def test_native_thread_settings_bookmark_title_and_temporary_heartbeat(self):
        token = self.token()
        created = self.call('/api/threads', token, 'POST', json={'is_temporary': True})
        self.assertEqual(created.status_code, 200)
        thread_id = created.json['id']
        self.assertTrue(created.json['is_temporary'])
        response = self.call('/api/threads/' + thread_id + '/settings', token, 'PUT', json={
            'custom_instruction': '簡潔に答える',
            'include_global_instruction': False,
            'is_temporary': True,
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['is_temporary'])
        settings = self.call('/api/threads/' + thread_id + '/settings', token)
        self.assertEqual(settings.status_code, 200)
        self.assertEqual(settings.json['custom_instruction'], '簡潔に答える')
        self.assertFalse(settings.json['include_global_instruction'])
        renamed = self.call('/api/threads/' + thread_id + '/title', token, 'PUT', json={'title': 'Android Phase 2'})
        self.assertEqual(renamed.json['title'], 'Android Phase 2')
        bookmarked = self.call('/api/threads/' + thread_id + '/bookmark', token, 'POST', json={})
        self.assertTrue(bookmarked.json['is_bookmarked'])
        heartbeat = self.call('/api/temporary_chat/heartbeat', token, 'POST', json={'thread_id': thread_id, 'active': True})
        self.assertEqual(heartbeat.status_code, 200)
        self.assertTrue(heartbeat.json['is_temporary'])
        listed = self.call('/api/threads', token).json['threads'][0]
        self.assertTrue(listed['is_bookmarked'])
        self.assertTrue(listed['is_temporary'])

    def test_native_thread_settings_remain_owner_scoped(self):
        token = self.token()
        with target.app.app_context():
            foreign = target.Thread(user_id=self.other_id, public_id=target.generate_thread_public_id())
            target.db.session.add(foreign)
            target.db.session.commit()
            foreign_id = foreign.public_id
        calls = [
            ('/api/threads/' + foreign_id + '/settings', 'GET', None),
            ('/api/threads/' + foreign_id + '/settings', 'PUT', {'is_temporary': True}),
            ('/api/threads/' + foreign_id + '/title', 'PUT', {'title': 'No'}),
            ('/api/threads/' + foreign_id + '/bookmark', 'POST', {}),
        ]
        for path, method, payload in calls:
            response = self.call(path, token, method, **({'json': payload} if payload is not None else {}))
            self.assertEqual(response.status_code, 403, (path, method))

    def test_browser_csrf_is_preserved(self):
        grant = self.device()
        bad = self.browser.post('/android/connect', base_url='https://localhost', data={'user_code': grant['user_code'], 'decision': 'approve'})
        self.assertEqual(bad.status_code, 403)
        self.assertEqual(self.browser.post('/api/threads', base_url='https://localhost', json={}).status_code, 403)
        self.assertEqual(self.poll(grant).json['error'], 'authorization_pending')

    def test_chat_validation_and_turnstile_still_apply(self):
        token = self.token()
        response = self.call('/chat_stream', token, 'POST', json={'message': ''})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json['error'], 'Invalid or oversized message')
        with mock.patch.object(target, '_bot_turnstile_active', return_value=True), \
             mock.patch.object(target, '_bot_turnstile_verified', return_value=False):
            response = self.call('/chat_stream', token, 'POST', json={'message': 'hello'})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json['error'], 'turnstile_required')

    def test_native_upload_download_and_foreign_file_denial(self):
        token = self.token()
        response = self.call('/upload', token, 'POST', data={'file': (io.BytesIO(b'android attachment'), 'note.txt')},
                             content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        filename = response.json['filename']
        response = self.call('/files/' + filename, token)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'android attachment')
        self.assertIn('no-store', response.headers['Cache-Control'])
        response.close()
        self.assertEqual(self.call('/files/' + str(self.other_id) + '/private.txt', token).status_code, 403)

    def test_native_chunked_upload_finalizes_and_enforces_scope(self):
        token = self.token()
        # Chunk endpoints are POST-only for the native client.
        self.assertEqual(self.call('/upload/init', token).status_code, 403)
        payload = b'chunked android attachment'
        init = self.call('/upload/init', token, 'POST', json={'filename': 'note.txt', 'size': len(payload)})
        self.assertEqual(init.status_code, 200)
        upload_id = init.json['upload_id']
        chunk_size = init.json['chunk_size']
        total = (len(payload) + chunk_size - 1) // chunk_size
        response = self.call('/upload/chunk', token, 'POST', data={
            'upload_id': upload_id, 'index': '0', 'total': str(total),
            'chunk': (io.BytesIO(payload), 'chunk'),
        }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['received'], len(payload))
        complete = self.call('/upload/complete', token, 'POST', json={'upload_id': upload_id})
        self.assertEqual(complete.status_code, 200)
        filename = complete.json['filename']
        self.assertTrue(filename.endswith('.txt'))
        got = self.call('/files/' + filename, token)
        self.assertEqual(got.status_code, 200)
        self.assertEqual(got.data, payload)
        got.close()

    def test_native_chunked_upload_rejects_out_of_order_chunks(self):
        token = self.token()
        init = self.call('/upload/init', token, 'POST', json={'filename': 'note.txt', 'size': 12})
        self.assertEqual(init.status_code, 200)
        upload_id = init.json['upload_id']
        # index 1 is invalid while index 0 has not been received yet.
        response = self.call('/upload/chunk', token, 'POST', data={
            'upload_id': upload_id, 'index': '1', 'total': '1',
            'chunk': (io.BytesIO(b'0123456789ab'), 'chunk'),
        }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 409)

    def test_native_library_and_gems_are_owner_scoped(self):
        token = self.token()
        response = self.call('/upload', token, 'POST',
            data={'file': (io.BytesIO(b'library bytes'), 'library.txt')},
            content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        filename = response.json['filename']
        library = self.call('/api/files', token)
        self.assertEqual(library.status_code, 200)
        entries = {item['filepath']: item for item in library.json['files']}
        self.assertIn(filename, entries)
        self.assertEqual(entries[filename]['ext'], 'txt')
        self.assertEqual(entries[filename]['filepath'], filename)
        # Favorite, rename and delete only touch the owner's library.
        favorite = self.call('/api/files/favorite', token, 'POST', json={'filepath': filename})
        self.assertEqual(favorite.status_code, 200)
        self.assertTrue(favorite.json['is_favorite'])
        rename = self.call('/api/files/rename', token, 'POST',
                           json={'filepath': filename, 'filename': 'renamed.txt'})
        self.assertEqual(rename.status_code, 200)
        self.assertEqual(rename.json['filename'], 'renamed.txt')
        with target.app.app_context():
            foreign = target.User(username='library-other', is_setup_completed=True)
            target.db.session.add(foreign)
            target.db.session.commit()
            foreign_id = foreign.id
        self.assertEqual(self.call('/api/files/favorite', token, 'POST',
                                   json={'filepath': f'{foreign_id}/secret.txt'}).status_code, 403)
        deleted = self.call('/api/files/delete', token, 'POST', json={'filenames': [filename]})
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse(self.call('/files/' + filename, token).status_code == 200)
        # Gems CRUD is owner scoped as well.
        created = self.call('/api/gems', token, 'POST', json={'name': 'Android Gem', 'instruction': 'Be brief'})
        self.assertEqual(created.status_code, 200)
        gem_uuid = created.json['uuid']
        listed = self.call('/api/gems', token)
        self.assertEqual(listed.status_code, 200)
        self.assertTrue(any(gem['uuid'] == gem_uuid for gem in listed.json))
        updated = self.call('/api/gems/' + gem_uuid, token, 'PUT', json={'name': 'Renamed Gem'})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json['name'], 'Renamed Gem')
        with target.app.app_context():
            foreign_gem = target.Gem(uuid='11111111-1111-1111-1111-111111111111',
                                     user_id=self.other_id, name='Foreign', instruction='No')
            target.db.session.add(foreign_gem)
            target.db.session.commit()
        for method in ['GET', 'PUT', 'DELETE']:
            payload = {'name': 'No'} if method == 'PUT' else None
            response = self.call('/api/gems/11111111-1111-1111-1111-111111111111', token, method,
                                 **({'json': payload} if payload is not None else {}))
            self.assertEqual(response.status_code, 403, method)
        self.assertEqual(self.call('/api/gems/' + gem_uuid, token, 'DELETE').status_code, 200)

    def test_native_preferences_are_limited_and_validated(self):
        token = self.token()
        prefs = self.call('/api/mobile/v1/preferences', token)
        self.assertEqual(prefs.status_code, 200)
        self.assertEqual(prefs.json['username'], 'android-owner')
        self.assertEqual(prefs.json['device_name'], 'Test Android')
        self.assertIn('session_expires_at', prefs.json)
        updated = self.call('/api/mobile/v1/preferences', token, 'PUT', json={
            'default_enable_thinking': True,
            'enter_to_send': True,
            'light_mode_enabled': True,
            'theme_color': '#123456',
            'temp_chat_timeout_seconds': 900,
            'prompt_bar_mode': 'minimal',
            'use_last_chat_settings': True,
            'system_prompt': 'Be concise.',
            'system_prompt_enabled': True,
            'stt_model': 'gpt-transcribe',
            'mic_transcribe_mode': 'llm',
            'liquid_glass_enabled': True,
            'default_2fa_method': 'webauthn',
        })
        self.assertEqual(updated.status_code, 200)
        self.assertTrue(updated.json['default_enable_thinking'])
        self.assertTrue(updated.json['enter_to_send'])
        self.assertTrue(updated.json['light_mode_enabled'])
        self.assertEqual(updated.json['theme_color'], '#123456')
        self.assertEqual(updated.json['temp_chat_timeout_seconds'], 900)
        self.assertEqual(updated.json['prompt_bar_mode'], 'minimal')
        self.assertTrue(updated.json['minimal_prompt_mode'])
        self.assertFalse(updated.json['compact_prompt_mode'])
        self.assertTrue(updated.json['use_last_chat_settings'])
        self.assertEqual(updated.json['system_prompt'], 'Be concise.')
        self.assertEqual(updated.json['stt_model'], 'gpt-transcribe')
        self.assertEqual(updated.json['mic_transcribe_mode'], 'llm')
        self.assertTrue(updated.json['liquid_glass_enabled'])
        self.assertEqual(updated.json['default_2fa_method'], 'webauthn')
        self.assertIn('global_system_prompt', updated.json)
        self.assertEqual(self.call('/api/mobile/v1/preferences', token, 'PUT',
                                   json={'default_model': 'definitely-not-a-model'}).status_code, 400)
        self.assertEqual(self.call('/api/mobile/v1/preferences', token, 'PUT',
                                   json={'stt_model': 'not-a-stt'}).status_code, 400)
        self.assertEqual(self.call('/api/mobile/v1/preferences', token, 'PUT',
                                   json={'prompt_bar_mode': 'huge'}).status_code, 400)
        # Provider secrets cannot be injected through the native preference endpoint.
        ignored = self.call('/api/mobile/v1/preferences', token, 'PUT', json={'openai_key': 'sk-live-secret'})
        self.assertEqual(ignored.status_code, 200)
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertNotEqual(user.openai_api_key, 'sk-live-secret')
            self.assertTrue(user.default_enable_thinking)
            self.assertEqual(user.temp_chat_timeout_seconds, 900)
            self.assertEqual(user.system_prompt, 'Be concise.')

    def test_native_feedback_and_mcp_list_are_owner_scoped(self):
        token = self.token()
        created = self.call('/api/feedback', token, 'POST', json={'title': 'Android', 'message': 'native note'})
        self.assertEqual(created.status_code, 200)
        listed = self.call('/api/feedback', token)
        self.assertEqual(listed.status_code, 200)
        self.assertTrue(any(item['message'] == 'native note' for item in listed.json['items']))
        servers = self.call('/api/mcp/servers', token)
        self.assertEqual(servers.status_code, 200)
        self.assertIn('servers', servers.json)
        self.assertEqual(self.call('/api/mcp/servers', token, 'POST', json={'name': 'x', 'url': 'https://example.com'}).status_code, 403)

    def test_native_thread_pdf_export_is_owner_scoped(self):
        token = self.token()
        created = self.call('/api/threads', token, 'POST', json={})
        self.assertEqual(created.status_code, 200)
        thread_id = created.json['id']
        export = self.call('/c/' + thread_id + '/pdf', token)
        self.assertEqual(export.status_code, 200)
        self.assertIn('messages', export.json)
        self.assertEqual(export.json['thread']['public_id'], thread_id)
        with target.app.app_context():
            foreign = target.Thread(user_id=self.other_id, public_id=target.generate_thread_public_id())
            target.db.session.add(foreign)
            target.db.session.commit()
            foreign_id = foreign.public_id
        self.assertEqual(self.call('/c/' + foreign_id + '/pdf', token).status_code, 403)
        # Message deletion is in the native scope (Web parity) and stays owner-checked; see the delete test.
        self.assertEqual(self.call('/api/messages/999999', token, 'DELETE').status_code, 404)

    def test_native_batch_history_is_owner_scoped(self):
        token = self.token()
        with target.app.app_context():
            own_thread = target.Thread(user_id=self.user_id, title='Android Batch')
            foreign_thread = target.Thread(user_id=self.other_id, title='Foreign Batch')
            target.db.session.add_all([own_thread, foreign_thread])
            target.db.session.flush()
            own = target.GeminiBatchJob(job_id='android-batch', user_id=self.user_id,
                thread_id=own_thread.id, user_message_id=1, assistant_message_id=2,
                provider='openai', model='gpt-5.6-sol', state='JOB_STATE_SUCCEEDED')
            foreign = target.GeminiBatchJob(job_id='foreign-batch', user_id=self.other_id,
                thread_id=foreign_thread.id, user_message_id=3, assistant_message_id=4,
                provider='openai', model='gpt-5.6-sol', state='JOB_STATE_SUCCEEDED')
            target.db.session.add_all([own, foreign])
            target.db.session.commit()
        listed = self.call('/api/batch/jobs', token)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual([row['job_id'] for row in listed.json['jobs']], ['android-batch'])
        self.assertEqual(self.call('/api/batch/jobs/foreign-batch', token, 'DELETE').status_code, 404)
        self.assertEqual(self.call('/api/batch/jobs/android-batch', token, 'DELETE').status_code, 200)

    def test_encrypted_account_pairs_and_reads_encrypted_history_and_attachment(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.enable_e2ee = True
            thread = target.Thread(user_id=self.user_id, title='Encrypted history')
            target.db.session.add(thread)
            target.db.session.flush()
            target.db.session.add(target.Message(thread_id=thread.id, role='assistant',
                content=target.encrypt_val('暗号化された回答'),
                thought_data=target.encrypt_val('暗号化された思考'), is_encrypted=True))
            target.db.session.commit()
            thread_id = thread.public_id or str(thread.id)
        token = self.token()
        config = self.native.get('/api/mobile/v1/config', base_url='https://localhost').json
        self.assertTrue(config['e2ee_supported'])
        self.assertEqual(config['encryption_mode'], 'server_managed_at_rest')
        me = self.call('/api/mobile/v1/me', token).json
        self.assertTrue(me['e2ee_enabled'])
        self.assertEqual(me['encryption_mode'], 'server_managed_at_rest')
        history = self.call('/api/threads/' + thread_id, token)
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json['messages'][0]['content'], '暗号化された回答')
        self.assertEqual(history.json['messages'][0]['thought_data'], '暗号化された思考')
        response = self.call('/upload', token, 'POST',
            data={'file': (io.BytesIO(b'encrypted android attachment'), 'secret.txt')},
            content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        filename = response.json['filename']
        # Only the per-test temporary upload directory, never application account-transfer data.
        stored = Path(target.app.config['UPLOAD_FOLDER']) / (filename + '.enc')
        self.assertNotIn(b'encrypted android attachment', stored.read_bytes())
        response = self.call('/files/' + filename, token)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'encrypted android attachment')
        response.close()
        self.assertEqual(self.call('/files/' + str(self.other_id) + '/private.txt', token).status_code, 403)
        with target.app.app_context():
            self.assertTrue(target.db.session.get(target.User, self.user_id).enable_e2ee)
            self.assertNotEqual(target.Message.query.one().content, '暗号化された回答')

    def test_public_auth_rejects_browser_origin_cookie_and_non_json(self):
        for header in [{'Origin': 'https://evil.example'}, {'Cookie': 'session=test'}]:
            response = self.native.post('/api/mobile/v1/device', base_url='https://localhost', headers=header,
                                        json={'client_id': 'official-android'})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.native.post('/api/mobile/v1/device', base_url='https://localhost', json=[]).status_code, 400)
        self.assertEqual(self.native.post('/api/mobile/v1/device', base_url='https://localhost', data='{}').status_code, 415)

    def test_native_signup_setup_login_and_totp(self):
        self.pass_integrity_gate()
        signup = self.native.post('/api/mobile/v1/auth/signup', base_url='https://localhost', json={
            'username': 'native-new-user', 'password': 'correct-horse-battery', 'device_name': 'Pixel test',
        })
        self.assertEqual(signup.status_code, 201)
        token = signup.json['access_token']
        self.assertTrue(signup.json['setup_required'])
        blocked = self.call('/api/threads', token)
        self.assertEqual(blocked.status_code, 403)
        self.assertEqual(blocked.json['error'], 'setup_required')
        setup = self.call('/api/mobile/v1/setup', token)
        self.assertEqual(setup.status_code, 200)
        self.assertEqual(setup.json['status'], 'setup_required')
        model = next(item['id'] for item in setup.json['models'] if item['selectable'])
        completed = self.call('/api/mobile/v1/setup', token, 'PUT', json={
            'default_model': model, 'anthropic_api_key': 'sk-ant-test', 'enable_e2ee': True,
        })
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.json['status'], 'ok')
        self.assertEqual(self.call('/api/mobile/v1/me', token).status_code, 200)
        # First-run values cannot be replayed onto the configured account.
        repeated = self.call('/api/mobile/v1/setup', token, 'PUT', json={'default_model': model, 'enable_e2ee': False})
        self.assertEqual(repeated.status_code, 409)
        self.assertEqual(repeated.json['error'], 'setup_already_completed')
        with target.app.app_context():
            self.assertTrue(target.User.query.filter_by(username='native-new-user').one().enable_e2ee)

        with target.app.app_context():
            password_user = target.User(username='native-password', is_setup_completed=True)
            password_user.set_password('correct-password')
            totp_user = target.User(username='native-totp', is_setup_completed=True,
                                    is_2fa_enabled=True, default_2fa_method='totp')
            totp_user.set_password('correct-password')
            totp_secret = target.pyotp.random_base32()
            totp_user.totp_secret = target.encrypt_val(totp_secret)
            target.db.session.add_all([password_user, totp_user])
            target.db.session.commit()
        login = self.native.post('/api/mobile/v1/auth/login', base_url='https://localhost', json={
            'username': 'native-password', 'password': 'correct-password', 'device_name': 'Pixel test',
        })
        self.assertEqual(login.status_code, 200)
        self.assertTrue(login.json['access_token'].startswith(target.MOBILE_TOKEN_PREFIX))
        two_factor = self.native.post('/api/mobile/v1/auth/login', base_url='https://localhost', json={
            'username': 'native-totp', 'password': 'correct-password', 'device_name': 'Pixel test',
        })
        self.assertEqual(two_factor.status_code, 200)
        self.assertEqual(two_factor.json['status'], '2fa_required')
        code = target.pyotp.TOTP(totp_secret).now()
        verified = self.native.post('/api/mobile/v1/auth/totp', base_url='https://localhost', json={
            'transaction_id': two_factor.json['transaction_id'], 'code': code, 'device_name': 'Pixel test',
        })
        self.assertEqual(verified.status_code, 200)
        self.assertTrue(verified.json['access_token'].startswith(target.MOBILE_TOKEN_PREFIX))

    def test_integrity_verdict_and_turnstile_ticket_are_bound_and_one_time(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.set_password('integrity-password')
            target.db.session.commit()
        path = '/api/mobile/v1/auth/login'
        body = {'username': 'android-owner', 'password': 'integrity-password', 'device_name': 'Pixel'}
        # Omitting every Integrity field must not skip the gate.
        omitted = self.native.post(path, base_url='https://localhost', json=body)
        self.assertEqual(omitted.status_code, 428)
        self.assertEqual(omitted.json['code'], 'turnstile_required')
        official_cert = bytes(range(32))
        env = mock.patch.dict(target.os.environ, {'ANDROID_APP_LINK_SHA256': ':'.join(f'{b:02X}' for b in official_cert)})
        env.start()
        self.addCleanup(env.stop)
        request_id = 'integrity-test-request-0001'
        request_hash = target._mobile_integrity_hash(request_id, path, body)
        verdict = {
            'requestDetails': {'requestPackageName': 'com.minashin1120.aiplayground',
                               'timestampMillis': int(time.time() * 1000), 'requestHash': request_hash},
            'appIntegrity': {'packageName': 'com.minashin1120.aiplayground',
                             'certificateSha256Digest': [target.base64.urlsafe_b64encode(official_cert).decode().rstrip('=')]},
            'deviceIntegrity': {'deviceRecognitionVerdict': ['MEETS_DEVICE_INTEGRITY']},
        }
        # A repackaged build keeps the package name but carries another certificate.
        repackaged_id = 'integrity-test-request-repackaged'
        repackaged = {
            **verdict,
            'requestDetails': {**verdict['requestDetails'],
                               'requestHash': target._mobile_integrity_hash(repackaged_id, path, body)},
            'appIntegrity': {**verdict['appIntegrity'],
                             'certificateSha256Digest': [target.base64.urlsafe_b64encode(bytes(32)).decode()]},
        }
        with mock.patch.object(target, '_mobile_integrity_decode', return_value=repackaged):
            rejected = self.native.post(path, base_url='https://localhost', json={
                **body, 'integrity_request_id': repackaged_id, 'integrity_token': 'mock-repackaged-token',
            })
        self.assertEqual(rejected.status_code, 428)
        with mock.patch.object(target, '_mobile_integrity_decode', return_value=verdict):
            accepted = self.native.post(path, base_url='https://localhost', json={
                **body, 'integrity_request_id': request_id, 'integrity_token': 'mock-integrity-token-value',
            })
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json['status'], 'ok')

        risky = {**body, 'integrity_request_id': 'integrity-test-request-0002', 'integrity_token': 'mock-risk-token-value'}
        with mock.patch.object(target, '_mobile_integrity_decode', return_value={
            **verdict,
            'requestDetails': {**verdict['requestDetails'], 'requestHash': 'wrong'},
        }):
            challenged = self.native.post(path, base_url='https://localhost', json=risky)
        self.assertEqual(challenged.status_code, 428)
        self.assertEqual(challenged.json['code'], 'turnstile_required')
        for request_suffix, invalid_verdict in [
            ('expired', {**verdict, 'requestDetails': {**verdict['requestDetails'],
                                                       'timestampMillis': int(time.time() * 1000) - 180_000}}),
            ('indeterminate', {**verdict, 'deviceIntegrity': {}}),
        ]:
            candidate = f'integrity-test-request-{request_suffix}-0001'
            candidate_body = {**body, 'integrity_enabled': True}
            invalid_verdict = {**invalid_verdict, 'requestDetails': {
                **invalid_verdict['requestDetails'],
                'requestHash': target._mobile_integrity_hash(candidate, path, candidate_body),
            }}
            with mock.patch.object(target, '_mobile_integrity_decode', return_value=invalid_verdict):
                response = self.native.post(path, base_url='https://localhost', json={
                    **candidate_body, 'integrity_request_id': candidate, 'integrity_token': 'mock-invalid-token',
                })
            self.assertEqual(response.status_code, 428)
        turnstile_page = self.browser.get(challenged.json['turnstile_url'], base_url='https://localhost')
        self.assertEqual(turnstile_page.status_code, 200)
        csrf_token = turnstile_page.data.decode().split('name="csrf_token" value="', 1)[1].split('"', 1)[0]
        with mock.patch.object(target, 'verify_turnstile', return_value=True):
            verified = self.browser.post('/android/integrity/turnstile/verify', base_url='https://localhost', data={
                'challenge': challenged.json['turnstile_url'].split('=')[-1],
                'csrf_token': csrf_token,
                'cf-turnstile-response': 'mock-turnstile-response',
            })
        self.assertEqual(verified.status_code, 303)
        ticket = verified.location.split('integrity_ticket=', 1)[1]
        cross_endpoint = self.native.post('/api/mobile/v1/auth/signup', base_url='https://localhost', json={
            'username': 'turnstile-cross-endpoint', 'password': 'integrity-password',
            'device_name': 'Pixel', 'integrity_enabled': True, 'integrity_turnstile_ticket': ticket,
        })
        self.assertEqual(cross_endpoint.status_code, 428)
        resumed = self.native.post(path, base_url='https://localhost', json={
            **body, 'integrity_request_id': risky['integrity_request_id'],
            'integrity_token': risky['integrity_token'], 'integrity_turnstile_ticket': ticket,
        })
        self.assertEqual(resumed.status_code, 200)
        self.assertEqual(resumed.json['status'], 'ok')
        replay = self.native.post(path, base_url='https://localhost', json={
            **body, 'integrity_enabled': True, 'integrity_turnstile_ticket': ticket,
        })
        self.assertEqual(replay.status_code, 428)
        expired_challenge = challenged.json['turnstile_url'].split('=', 1)[1]
        self.redis.delete('mobile:integrity:challenge:' + target._mobile_digest(expired_challenge))
        expired_page = self.browser.get(challenged.json['turnstile_url'], base_url='https://localhost')
        self.assertEqual(expired_page.status_code, 410)

    def test_mobile_config_exposes_integrity_project_number_without_credentials(self):
        with mock.patch.dict('os.environ', {
            'PLAY_INTEGRITY_CLOUD_PROJECT_NUMBER': '808778798504',
            'PLAY_INTEGRITY_SERVICE_ACCOUNT_FILE': '/home/private/service-account.json',
        }):
            config = self.native.get('/api/mobile/v1/config', base_url='https://localhost')
        self.assertEqual(config.status_code, 200)
        self.assertEqual(config.json['play_integrity_cloud_project_number'], '808778798504')
        self.assertNotIn('service_account', json.dumps(config.json).lower())
        self.assertNotIn('/home/private', json.dumps(config.json))

    def test_native_google_login_uses_verified_id_token(self):
        self.pass_integrity_gate()
        with mock.patch.dict(target.os.environ, {'GOOGLE_CLIENT_ID': 'android-server-client'}), \
                mock.patch.object(target.id_token, 'verify_oauth2_token', return_value={
                    'sub': 'google-sub-native', 'email': 'NativeUser@Example.com',
                    'email_verified': True, 'nonce': 'native-google-nonce',
                }) as verify:
            response = self.native.post('/api/mobile/v1/auth/google', base_url='https://localhost', json={
                'id_token': 'header.payload.signature', 'nonce': 'native-google-nonce',
                'device_name': 'Pixel test',
            })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['access_token'].startswith(target.MOBILE_TOKEN_PREFIX))
        verify.assert_called_once()
        self.assertEqual(verify.call_args.args[2], 'android-server-client')
        with target.app.app_context():
            user = target.User.query.filter_by(google_id='google-sub-native').one()
            self.assertEqual(user.google_email, 'nativeuser@example.com')

    def test_native_auth_code_is_one_time_and_assetlinks_is_configurable(self):
        self.pass_integrity_gate()
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            code = target._mobile_native_auth_code(user, 'App Link test')
        exchanged = self.native.post('/api/mobile/v1/auth/exchange', base_url='https://localhost', json={'code': code})
        self.assertEqual(exchanged.status_code, 200)
        self.assertTrue(exchanged.json['access_token'].startswith(target.MOBILE_TOKEN_PREFIX))
        self.assertEqual(self.native.post('/api/mobile/v1/auth/exchange', base_url='https://localhost', json={'code': code}).status_code, 401)
        with mock.patch.dict(target.os.environ, {
            'ANDROID_APP_ID': 'com.example.test',
            'ANDROID_APP_LINK_SHA256': 'AA:BB,CC:DD',
        }):
            links = self.native.get('/.well-known/assetlinks.json', base_url='https://localhost')
        self.assertEqual(links.status_code, 200)
        self.assertEqual(links.json[0]['target']['package_name'], 'com.example.test')
        self.assertEqual(links.json[0]['target']['sha256_cert_fingerprints'], ['AA:BB', 'CC:DD'])

    def test_cookie_and_bearer_cannot_mix(self):
        token = self.token()
        response = self.browser.get('/api/threads', base_url='https://localhost', headers={'Authorization': 'Bearer ' + token})
        self.assertEqual(response.status_code, 400)

    def test_https_and_disable_switch(self):
        self.assertEqual(self.native.get('/api/mobile/v1/config').json['error'], 'https_required')
        token = self.token()
        with mock.patch.dict(target.app.config, MOBILE_API_ENABLED=False):
            self.assertEqual(self.call('/api/threads', token).status_code, 503)
            self.assertEqual(self.native.get('/api/mobile/v1/config', base_url='https://localhost').status_code, 503)

    def test_expired_session_ban_e2ee_and_maintenance(self):
        token = self.token()
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.is_bot_banned = True
            target.db.session.commit()
        self.assertEqual(self.call('/api/threads', token).status_code, 403)
        self.assertEqual(self.call('/chat_stream', token, 'POST', json={}).json['error'], 'request_blocked')
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.is_bot_banned = False
            user.enable_e2ee = True
            target.db.session.commit()
        self.assertEqual(self.call('/api/threads', token).status_code, 200)
        self.assertEqual(self.call('/api/mobile/v1/me', token).status_code, 200)
        with mock.patch.dict(target.app.config, MAINTENANCE_MODE=True):
            self.assertEqual(self.call('/api/mobile/v1/me', token).json['error'], 'maintenance')
        with target.app.app_context():
            row = target.UserSession.query.filter(target.UserSession.session_id.startswith('android:')).one()
            row.created_at = datetime.utcnow() - timedelta(days=31)
            target.db.session.commit()
        self.assertEqual(self.call('/api/mobile/v1/me', token).status_code, 401)

    def test_web_revoke_all_revokes_native_token(self):
        token = self.token()
        with target.app.app_context():
            target.revoke_user_sessions(self.user_id)
        self.assertEqual(self.call('/api/threads', token).status_code, 401)

    def test_revocation_remains_possible_during_bot_gate_lock_ban_and_maintenance(self):
        for state in ['turnstile', 'lock', 'ban', 'maintenance', 'setup']:
            token = self.token()
            with target.app.app_context():
                user = target.db.session.get(target.User, self.user_id)
                user.is_bot_banned = state == 'ban'
                user.is_setup_completed = state != 'setup'
                target.db.session.commit()
            with mock.patch.object(target, '_bot_turnstile_active', return_value=state == 'turnstile'), \
                 mock.patch.object(target, '_bot_turnstile_verified', return_value=False), \
                 mock.patch.object(target, '_bot_lock_info', return_value=(state == 'lock', '', 300)), \
                 mock.patch.dict(target.app.config, MAINTENANCE_MODE=state == 'maintenance'):
                response = self.call('/api/mobile/v1/revoke', token, 'POST', json={})
                self.assertEqual(response.status_code, 200, state)
            with target.app.app_context():
                user = target.db.session.get(target.User, self.user_id)
                user.is_bot_banned = False
                user.is_setup_completed = True
                target.db.session.commit()

    def test_pending_poll_rate_limit_and_expiry(self):
        grant = self.device()
        self.assertEqual(self.poll(grant).json['error'], 'authorization_pending')
        self.assertEqual(self.poll(grant).json['error'], 'slow_down')
        digest = target._mobile_digest(grant['device_code'])
        self.redis.delete('mobile:poll:' + digest, 'mobile:grant:' + digest)
        self.assertEqual(self.poll(grant).json['error'], 'expired_token')

    def test_denial_and_one_time_redemption(self):
        grant = self.device()
        self.assertEqual(self.approve(grant, 'deny').status_code, 200)
        self.assertEqual(self.poll(grant).json['error'], 'access_denied')
        granted = self.device()
        self.assertEqual(self.approve(granted).status_code, 200)
        self.assertEqual(self.approve(granted).status_code, 400)
        self.assertEqual(self.poll(granted).status_code, 200)
        self.redis.delete('mobile:poll:' + target._mobile_digest(granted['device_code']))
        self.assertEqual(self.poll(granted).json['error'], 'expired_token')

    def test_atomic_scripts_allow_only_one_approval_and_redemption(self):
        from concurrent.futures import ThreadPoolExecutor
        key = 'mobile:test:race'
        self.redis.set(key, json.dumps({'status': 'pending'}), ex=60)
        def approve(_):
            return self.redis.eval(target._MOBILE_APPROVE_SCRIPT, 1, key, json.dumps({'status': 'approved'}))
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.assertEqual(sum(executor.map(approve, range(8))), 1)
            results = list(executor.map(lambda _: self.redis.eval(target._MOBILE_REDEEM_SCRIPT, 1, key), range(8)))
        self.assertEqual(sum(bool(value) for value in results), 1)

    def test_redis_failure_fails_closed(self):
        with mock.patch.object(target, 'redis_conn') as failed:
            failed.incr.side_effect = redis.ConnectionError('test offline')
            response = self.native.post('/api/mobile/v1/device', base_url='https://localhost', json={'client_id': 'official-android'})
            self.assertEqual(response.status_code, 503)

    def test_device_rate_limit_and_safe_name_rendering(self):
        grant = self.device()
        key = 'mobile:grant:' + target._mobile_digest(grant['device_code'])
        self.redis.set(key, json.dumps({'status': 'pending', 'device_name': '<script>unsafe</script>'}), ex=60)
        response = self.approve(grant, 'review')
        self.assertIn('&lt;script&gt;', response.get_data(as_text=True))
        self.assertNotIn('<script>unsafe', response.get_data(as_text=True))
        for _ in range(9):
            self.device()
        response = self.native.post('/api/mobile/v1/device', base_url='https://localhost', json={'client_id': 'official-android'})
        self.assertEqual(response.status_code, 429)

    def test_login_returns_to_connect_and_get_never_approves(self):
        anonymous = target.app.test_client()
        response = anonymous.get('/android/connect', base_url='https://localhost')
        self.assertEqual(response.status_code, 302)
        with anonymous.session_transaction() as sess:
            self.assertTrue(sess['mobile_connect_pending'])
            sess['_user_id'] = str(self.user_id)
        response = anonymous.get('/', base_url='https://localhost')
        self.assertEqual(response.headers['Location'], '/android/connect')
        self.assertEqual(self.browser.get('/android/connect', base_url='https://localhost').status_code, 200)
        with target.app.app_context():
            self.assertEqual(target.UserSession.query.filter(target.UserSession.session_id.startswith('android:')).count(), 0)

    def test_login_preserves_url_code_and_renders_review_without_manual_input(self):
        grant = self.device()
        anonymous = target.app.test_client()
        response = anonymous.get('/android/connect?code=' + grant['user_code'], base_url='https://localhost')
        self.assertEqual(response.status_code, 302)
        with anonymous.session_transaction() as sess:
            self.assertEqual(sess['mobile_connect_code'], grant['user_code'])
            sess['_user_id'] = str(self.user_id)
            sess['_fresh'] = True
        response = anonymous.get('/', base_url='https://localhost')
        self.assertEqual(response.headers['Location'], '/android/connect?code=' + grant['user_code'])
        response = anonymous.get(response.headers['Location'], base_url='https://localhost')
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('Test Android', body)
        self.assertIn(grant['user_code'], body)
        self.assertNotIn('id="user-code"', body)
        self.assertEqual(self.poll(grant).json['error'], 'authorization_pending')

    def test_native_security_totp_passkey_and_import_scope(self):
        self.pass_integrity_gate()
        token = self.token()
        status = self.call('/api/mobile/v1/security', token)
        self.assertEqual(status.status_code, 200)
        self.assertFalse(status.json['has_totp'])
        self.assertFalse(status.json['has_webauthn'])

        # TOTP setup -> enable -> disable round trip through the bearer API.
        setup = self.call('/api/mobile/v1/security/totp/setup', token, 'POST', json={})
        self.assertEqual(setup.status_code, 200)
        secret = setup.json['secret']
        self.assertTrue(setup.json['otpauth_uri'].startswith('otpauth://totp/'))
        enabled = self.call('/api/mobile/v1/security/totp/enable', token, 'POST',
                            json={'code': target.pyotp.TOTP(secret).now()})
        self.assertEqual(enabled.status_code, 200)
        self.assertTrue(enabled.json['has_totp'])
        self.assertTrue(enabled.json['is_2fa_enabled'])
        disabled = self.call('/api/mobile/v1/security/totp/disable', token, 'POST',
                             json={'code': target.pyotp.TOTP(secret).now()})
        self.assertEqual(disabled.status_code, 200)
        self.assertFalse(disabled.json['has_totp'])
        self.assertFalse(disabled.json['is_2fa_enabled'])
        self.assertEqual(self.call('/api/mobile/v1/security/totp/enable', token, 'POST',
                                   json={'code': '000000'}).status_code, 400)

        # Preference validation and the passkey-only guard.
        self.assertEqual(self.call('/api/mobile/v1/security/preferences', token, 'POST',
                                   json={'default_2fa_method': 'sms'}).status_code, 400)
        self.assertEqual(self.call('/api/mobile/v1/security/preferences', token, 'POST',
                                   json={'passkey_only_login': True}).status_code, 400)
        saved = self.call('/api/mobile/v1/security/preferences', token, 'POST',
                          json={'default_2fa_method': 'webauthn', 'skip_2fa_on_google_login': True})
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json['default_2fa_method'], 'webauthn')
        self.assertTrue(saved.json['skip_2fa_on_google_login'])

        # Unknown accounts must not reveal whether a passkey exists.
        unknown = self.native.post('/api/mobile/v1/auth/passkey/options', base_url='https://localhost',
                                   json={'username': 'no-such-user', 'device_name': 'Pixel test'})
        self.assertEqual(unknown.status_code, 401)
        self.assertEqual(unknown.json['error'], 'passkey_unavailable')

        # Android credential origins are derived from the configured fingerprints.
        with mock.patch.dict(target.os.environ, {'ANDROID_APP_LINK_SHA256': '00' * 32}):
            origins = target._mobile_android_origins()
        self.assertEqual(len(origins), 1)
        self.assertTrue(origins[0].startswith('android:apk-key-hash:'))
        self.assertNotIn('=', origins[0])

        # The chunked account import routes are registered for the native bearer
        # and remain unavailable to unauthenticated callers. The routes themselves
        # are not exercised here because they write into the app's account-import
        # directory, which automated checks must not touch.
        self.assertIn('start_account_import_upload', target.MOBILE_ENDPOINT_METHODS)
        self.assertIn('account_import_upload_chunk', target.MOBILE_ENDPOINT_METHODS)
        self.assertIn('complete_account_import_upload', target.MOBILE_ENDPOINT_METHODS)
        self.assertIn('cancel_account_import_upload', target.MOBILE_ENDPOINT_METHODS)
        self.assertIn('import_account_data', target.MOBILE_ENDPOINT_METHODS)
        for name in ['start_account_import_upload', 'account_import_upload_chunk',
                     'complete_account_import_upload', 'cancel_account_import_upload', 'import_account_data']:
            self.assertIn(name, target.MOBILE_SETUP_ENDPOINTS)
        self.assertNotEqual(self.native.post('/api/account/import/upload/start', base_url='https://localhost',
                                             json={'size': 10}).status_code, 200)

    def test_setup_security_and_import_are_allowed_before_setup_completion(self):
        self.pass_integrity_gate()
        signup = self.native.post('/api/mobile/v1/auth/signup', base_url='https://localhost', json={
            'username': 'native-import-user', 'password': 'correct-horse-battery', 'device_name': 'Pixel test',
        })
        self.assertEqual(signup.status_code, 201)
        token = signup.json['access_token']
        self.assertEqual(self.call('/api/threads', token).status_code, 403)
        # Security management is reachable while setup is still pending, but the
        # regular chat API stays blocked behind setup_required.
        security = self.call('/api/mobile/v1/security', token)
        self.assertEqual(security.status_code, 200)
        self.assertIn('has_totp', security.json)

    def test_browser_login_code_is_bound_to_pkce_verifier(self):
        self.pass_integrity_gate()
        # RFC 7636 Appendix B.
        verifier = 'dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk'
        challenge = 'E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM'

        def code(code_challenge):
            with target.app.app_context():
                user = target.db.session.get(target.User, self.user_id)
                return target._mobile_native_auth_code(user, 'App Link test', 'minashin', code_challenge)

        def exchange(body):
            return self.native.post('/api/mobile/v1/auth/exchange', base_url='https://localhost', json=body)

        # An intercepted or injected code is useless without the starting app's verifier.
        self.assertEqual(exchange({'code': code(challenge)}).status_code, 401)
        self.assertEqual(exchange({'code': code(challenge), 'code_verifier': 'x' * 43}).status_code, 401)
        # A code minted without a challenge cannot be injected into a PKCE client.
        self.assertEqual(exchange({'code': code(None), 'code_verifier': verifier}).status_code, 401)
        accepted = exchange({'code': code(challenge), 'code_verifier': verifier})
        self.assertEqual(accepted.status_code, 200)
        self.assertTrue(accepted.json['access_token'].startswith(target.MOBILE_TOKEN_PREFIX))

        browser = target.app.test_client()
        started = browser.get('/android/auth/minashin/start?code_challenge_method=S256&code_challenge=' + challenge,
                              base_url='https://localhost')
        self.assertEqual(started.status_code, 302)
        with browser.session_transaction() as sess:
            self.assertEqual(sess['mobile_native_code_challenge'], challenge)
            self.assertTrue(sess['mobile_native_auth'])
        # A later Web login in the same browser is not captured by the abandoned native start.
        self.assertEqual(browser.get('/login/minashin', base_url='https://localhost').status_code, 302)
        with browser.session_transaction() as sess:
            self.assertNotIn('mobile_native_auth', sess)
            self.assertNotIn('mobile_native_code_challenge', sess)
        with target.app.test_request_context('/?code_challenge_method=plain&code_challenge=' + challenge):
            self.assertIsNone(target._mobile_native_code_challenge(target.request.args))

    def test_passkey_sign_in_requires_passkey_only_login(self):
        self.pass_integrity_gate()
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            target._save_user_webauthn_credentials(user, [{'id': 'AQID', 'public_key': 'BAUG', 'sign_count': 0}])
            user.is_2fa_enabled = True
            target.db.session.commit()

        def options():
            return self.native.post('/api/mobile/v1/auth/passkey/options', base_url='https://localhost',
                                    json={'username': 'android-owner', 'device_name': 'Pixel test'})

        # A passkey registered as a second factor is not a password replacement.
        second_factor_only = options()
        self.assertEqual(second_factor_only.status_code, 401)
        self.assertEqual(second_factor_only.json['error'], 'passkey_unavailable')
        with target.app.app_context():
            target.db.session.get(target.User, self.user_id).passkey_only_login = True
            target.db.session.commit()
        allowed = options()
        self.assertEqual(allowed.status_code, 200)
        self.assertTrue(allowed.json['transaction_id'])
        with target.app.app_context():
            target.db.session.get(target.User, self.user_id).passkey_only_login = False
            target.db.session.commit()
        # Turning the setting off also invalidates an options transaction already issued.
        with mock.patch.object(target, 'verify_authentication_response',
                               return_value=mock.Mock(new_sign_count=1)):
            revoked = self.native.post('/api/mobile/v1/auth/passkey/verify', base_url='https://localhost', json={
                'transaction_id': allowed.json['transaction_id'], 'credential': {'id': 'AQID'},
            })
        self.assertEqual(revoked.status_code, 401)

    def test_account_import_is_limited_to_first_run_setup(self):
        token = self.token()
        # Rejected by the bearer guard before the view runs; should that regress,
        # the view still writes into a throwaway directory, never the real one.
        import_root = tempfile.TemporaryDirectory(prefix='mobile-import-guard-')
        self.addCleanup(import_root.cleanup)
        patcher = mock.patch.object(target, '_account_import_upload_root', return_value=import_root.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        blocked =self.call('/api/account/import/upload/start', token, 'POST', json={'size': 10})
        self.assertEqual(blocked.status_code, 403)
        self.assertEqual(blocked.json['error'], 'setup_already_completed')
        self.assertEqual(self.call('/api/account/import', token, 'POST', json={'upload_id': 'x'}).status_code, 403)
        replay = self.call('/api/mobile/v1/setup', token, 'PUT', json={'default_model': 'gemini-3.6-flash'})
        self.assertEqual(replay.status_code, 409)

    def test_pairing_review_warns_against_forwarded_codes(self):
        grant = self.device()
        page = self.browser.get('/android/connect?code=' + grant['user_code'], base_url='https://localhost')
        self.assertEqual(page.status_code, 200)
        self.assertIn('他の人から届いたリンクやコードを許可すると', page.get_data(as_text=True))
