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
        # Feedback posted by these tests must not land in the real app/feedback/ directory.
        feedback_dir = tempfile.TemporaryDirectory(prefix='mobile-api-feedback-')
        self.addCleanup(feedback_dir.cleanup)
        for patcher in [
            mock.patch.object(target, 'redis_conn', self.redis),
            mock.patch.object(target, '_ensure_temp_chat_monitor_running'),
            mock.patch.object(target, '_bot_turnstile_active', return_value=False),
            mock.patch.object(target, '_bot_lock_info', return_value=(False, '', 0)),
            mock.patch.dict(target.app.config, TESTING=True, MOBILE_API_ENABLED=True,
                            MAINTENANCE_MODE=False, TRUSTED_HOSTS=['localhost'], UPLOAD_FOLDER=upload_dir.name,
                            FEEDBACK_DIR=feedback_dir.name),
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

    def token(self):
        with target.app.test_request_context(base_url='https://localhost'):
            user = target.db.session.get(target.User, self.user_id)
            return target._mobile_issue_token(user, 'Test Android')

    def call(self, path, token, method='GET', **kwargs):
        return self.native.open(path, method=method, base_url='https://localhost',
                                headers={'Authorization': 'Bearer ' + token}, **kwargs)

    def pass_integrity_gate(self):
        # The Integrity/Turnstile gate itself is covered by the integrity tests.
        patcher = mock.patch.object(target, '_mobile_integrity_gate', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def set_password(self, password='correct-horse-1'):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.set_password(password)
            target.db.session.commit()
        return password

    def _totp_user(self, username):
        secret = target.pyotp.random_base32()
        with target.app.app_context():
            user = target.User(username=username, is_setup_completed=True,
                               is_2fa_enabled=True, default_2fa_method='totp')
            user.set_password('correct-password')
            user.totp_secret = target.encrypt_val(secret)
            target.db.session.add(user)
            target.db.session.commit()
            return user.id, secret

    def _native_login(self, username, password='correct-password'):
        return self.native.post('/api/mobile/v1/auth/login', base_url='https://localhost', json={
            'username': username, 'password': password, 'device_name': 'Pixel test',
        })

    def _native_totp(self, transaction_id, code):
        return self.native.post('/api/mobile/v1/auth/totp', base_url='https://localhost', json={
            'transaction_id': transaction_id, 'code': code,
        })
