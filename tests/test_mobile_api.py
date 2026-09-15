"""Exercise native authentication with isolated SQL and real, ephemeral Redis."""
import json
import io
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

    def test_pairing_token_hash_scope_and_revocation(self):
        token = self.token()
        response = self.call('/api/mobile/v1/me', token)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['username'], 'android-owner')
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

    def test_public_auth_rejects_browser_origin_cookie_and_non_json(self):
        for header in [{'Origin': 'https://evil.example'}, {'Cookie': 'session=test'}]:
            response = self.native.post('/api/mobile/v1/device', base_url='https://localhost', headers=header,
                                        json={'client_id': 'official-android'})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.native.post('/api/mobile/v1/device', base_url='https://localhost', json=[]).status_code, 400)
        self.assertEqual(self.native.post('/api/mobile/v1/device', base_url='https://localhost', data='{}').status_code, 415)

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
        self.assertEqual(self.call('/api/threads', token).status_code, 409)
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
