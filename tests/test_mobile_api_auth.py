from tests.mobile_api_common import *
from tests.mobile_api_common import MobileApiTests


class MobileApiTestsAuth(MobileApiTests):
    def test_signed_in_turnstile_check_issues_an_account_bound_ticket(self):
        token = self.token()
        with mock.patch.object(target, '_bot_turnstile_active', return_value=True), \
             mock.patch.object(target, '_bot_turnstile_verified', return_value=False):
            blocked = self.call('/chat_stream', token, 'POST', json={'message': 'hello'})
            self.assertEqual(blocked.status_code, 403)
            self.assertEqual(blocked.json['error'], 'turnstile_required')
            started = self.call('/api/mobile/v1/security/turnstile', token, 'POST', json={})
            self.assertEqual(started.status_code, 200)
            url = started.json['turnstile_url']
            self.assertTrue(url.startswith('/android/integrity/turnstile?challenge='))
            page = self.browser.get(url, base_url='https://localhost')
            self.assertEqual(page.status_code, 200)
            csrf_token = page.data.decode().split('name="csrf_token" value="', 1)[1].split('"', 1)[0]
            challenge = url.split('=', 1)[1]
            with mock.patch.object(target, 'verify_turnstile', return_value=False):
                failed = self.browser.post('/android/integrity/turnstile/verify', base_url='https://localhost', data={
                    'challenge': challenge, 'csrf_token': csrf_token, 'cf-turnstile-response': 'bad'})
            self.assertEqual(failed.status_code, 403)
            with mock.patch.object(target, 'verify_turnstile', return_value=True):
                verified = self.browser.post('/android/integrity/turnstile/verify', base_url='https://localhost', data={
                    'challenge': challenge, 'csrf_token': csrf_token, 'cf-turnstile-response': 'ok'})
            self.assertEqual(verified.status_code, 303)
            # The return link follows the serving host (the official host in production).
            self.assertTrue(verified.location.startswith('https://localhost/android/auth/callback?turnstile_ticket='))
            ticket = verified.location.split('turnstile_ticket=', 1)[1]
            # The ticket cannot be exchanged for a login and only this account can redeem it.
            self.assertEqual(self.native.post('/api/mobile/v1/auth/login', base_url='https://localhost', json={
                'username': 'android-owner', 'password': 'x', 'device_name': 'Pixel',
                'integrity_enabled': True, 'integrity_turnstile_ticket': ticket,
            }).status_code, 428)
            foreign = 'f' * 43
            self.redis.set('mobile:turnstile:session:' + target._mobile_digest(foreign), str(self.other_id), ex=60)
            self.assertEqual(self.call('/api/mobile/v1/security/turnstile/complete', token, 'POST',
                                       json={'ticket': foreign}).status_code, 410)
            completed = self.call('/api/mobile/v1/security/turnstile/complete', token, 'POST', json={'ticket': ticket})
            self.assertEqual(completed.status_code, 200)
            self.assertEqual(completed.json['status'], 'ok')
            self.assertTrue(self.redis.exists(f'bot:tst:v:{self.user_id}'))
            replay = self.call('/api/mobile/v1/security/turnstile/complete', token, 'POST', json={'ticket': ticket})
            self.assertEqual(replay.status_code, 410)
        # Browsers cannot use the native endpoints with their session cookie.
        self.assertIn(self.browser.post('/api/mobile/v1/security/turnstile', base_url='https://localhost',
                                        json={}).status_code, (400, 401))

    def test_app_reports_rapid_sends_and_taps_to_the_web_bot_detection(self):
        token = self.token()
        with mock.patch.object(target, '_bot_turnstile_active', return_value=False):
            self.assertFalse(self.call('/api/mobile/v1/preferences', token).json['bot_detection_active'])
        with mock.patch.object(target, '_bot_turnstile_active', return_value=True), \
             mock.patch.object(target, '_bot_turnstile_verified', return_value=True), \
             mock.patch.object(target, 'get_bot_detection_global_enabled', return_value=True), \
             mock.patch.object(target, 'verify_turnstile', return_value=False) as verify:
            self.assertTrue(self.call('/api/mobile/v1/preferences', token).json['bot_detection_active'])
            locked = self.call('/api/bot/lock', token, 'POST', json={'reason': 'rapid'})
            self.assertEqual(locked.status_code, 200)
            self.assertEqual(locked.json['status'], 'locked')
            self.assertIn('remaining_seconds', locked.json)
            # setUp stubs the lock lookup; the lock itself is the Web one with its timeout.
            self.assertEqual(self.redis.get(f'bot:lock:{self.user_id}'), b'rapid')
            self.assertGreater(self.redis.ttl(f'bot:lock:{self.user_id}'), 0)
            # The app has no Turnstile token: its reports are scored without one,
            # and it cannot add a Turnstile failure to its own score.
            calm = self.call('/api/bot-telemetry', token, 'POST', json={'clicks': 3, 'window_ms': 4000,
                                                                      'turnstile_failed': True, 'challenged': True})
            self.assertEqual(calm.status_code, 200)
            self.assertEqual(calm.json['score'], 0)
            rapid = self.call('/api/bot-telemetry', token, 'POST', json={
                'window_ms': 400, 'clicks': 13, 'fast_clicks': 12, 'click_burst': 13,
                'avg_click_ms': 30, 'click_cv': 0.01, 'event_rate': 32})
            self.assertEqual(rapid.status_code, 403)
            self.assertEqual(rapid.json['error'], 'banned')
            verify.assert_not_called()
        with target.app.app_context():
            self.assertTrue(target.db.session.get(target.User, self.user_id).is_bot_banned)

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

    def test_native_totp_code_cannot_be_replayed(self):
        self.pass_integrity_gate()
        _, secret = self._totp_user('native-replay')
        code = target.pyotp.TOTP(secret).now()
        first = self._native_login('native-replay')
        self.assertEqual(self._native_totp(first.json['transaction_id'], code).status_code, 200)
        second = self._native_login('native-replay')
        replayed = self._native_totp(second.json['transaction_id'], code)
        self.assertEqual(replayed.status_code, 401)
        self.assertEqual(replayed.json['error'], 'invalid_2fa')

    def test_native_2fa_transaction_ends_after_repeated_wrong_codes(self):
        self.pass_integrity_gate()
        _, secret = self._totp_user('native-tx-fail')
        transaction_id = self._native_login('native-tx-fail').json['transaction_id']
        good = target.pyotp.TOTP(secret).now()
        wrong = '000000' if good != '000000' else '111111'
        for _ in range(target._MOBILE_AUTH_TX_MAX_FAILURES):
            self.assertEqual(self._native_totp(transaction_id, wrong).status_code, 401)
        # The password step must be repeated; the correct code no longer completes this sign-in.
        self.assertEqual(self._native_totp(transaction_id, good).status_code, 401)
        self.assertEqual(self._native_totp(self._native_login('native-tx-fail').json['transaction_id'], good).status_code, 200)

    def test_account_failure_limits_are_shared_and_reset_on_success(self):
        self.pass_integrity_gate()
        user_id, secret = self._totp_user('native-guess')
        self.assertEqual(self._native_login('native-guess', 'wrong-password').status_code, 401)
        self.assertEqual(int(self.redis.get(f'authfail:password:{user_id}')), 1)
        self.assertEqual(self._native_login('native-guess').status_code, 200)
        self.assertIsNone(self.redis.get(f'authfail:password:{user_id}'))
        # Failures from other devices or Web count toward the same account limit.
        self.redis.set(f'authfail:password:{user_id}', target.AUTH_FAILURE_LIMITS['password'][0])
        self.assertEqual(self._native_login('native-guess').status_code, 429)
        self.redis.delete(f'authfail:password:{user_id}')
        transaction_id = self._native_login('native-guess').json['transaction_id']
        self.redis.set(f'authfail:totp:{user_id}', target.AUTH_FAILURE_LIMITS['totp'][0])
        self.assertEqual(self._native_totp(transaction_id, target.pyotp.TOTP(secret).now()).status_code, 429)

    def test_native_totp_disable_limits_guessing(self):
        token = self.token()
        secret = target.pyotp.random_base32()
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.totp_secret = target.encrypt_val(secret)
            user.is_2fa_enabled = True
            target.db.session.commit()
        good = target.pyotp.TOTP(secret).now()
        wrong = '000000' if good != '000000' else '111111'
        for _ in range(8):
            self.assertEqual(self.call('/api/mobile/v1/security/totp/disable', token, 'POST', json={'code': wrong}).status_code, 401)
        self.assertEqual(int(self.redis.get(f'authfail:totp:{self.user_id}')), 8)
        blocked = self.call('/api/mobile/v1/security/totp/disable', token, 'POST', json={'code': good})
        self.assertEqual(blocked.status_code, 429)
        with target.app.app_context():
            self.assertTrue(target.db.session.get(target.User, self.user_id).totp_secret)

    def test_native_reauth_counts_password_failures_per_account(self):
        password = self.set_password()
        token = self.token()
        self.redis.set(f'authfail:password:{self.user_id}', target.AUTH_FAILURE_LIMITS['password'][0])
        blocked = self.call('/api/mobile/v1/reauth', token, 'POST', json={'method': 'password', 'password': password})
        self.assertEqual(blocked.status_code, 429)
        self.redis.delete(f'authfail:password:{self.user_id}')
        self.assertEqual(self.call('/api/mobile/v1/reauth', token, 'POST', json={'method': 'password', 'password': 'wrong'}).status_code, 401)
        self.assertEqual(int(self.redis.get(f'authfail:password:{self.user_id}')), 1)
        self.assertEqual(self.call('/api/mobile/v1/reauth', token, 'POST', json={'method': 'password', 'password': password}).status_code, 200)
        self.assertIsNone(self.redis.get(f'authfail:password:{self.user_id}'))

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

    def test_banned_user_sees_ban_details_and_sends_appeal_like_web(self):
        token = self.token()
        self.assertEqual(self.call('/api/mobile/v1/ban/appeal', token, 'POST', json={'message': 'x' * 20}).status_code, 409)
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.is_bot_banned = True
            user.bot_ban_reason = 'rapid_send'
            user.bot_banned_at = datetime(2026, 10, 12, 1, 2, 3)
            target.db.session.commit()
        self.assertEqual(self.call('/api/threads', token).json['error'], 'banned')
        status = self.call('/api/mobile/v1/ban', token)
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json['reason'], 'rapid_send')
        self.assertEqual(status.json['banned_at'], '2026-10-12 01:02:03')
        self.assertIsNone(status.json['latest_appeal'])
        short = self.call('/api/mobile/v1/ban/appeal', token, 'POST', json={'message': '  short  '})
        self.assertEqual(short.status_code, 400)
        self.assertEqual(short.json['message'], '内容は10文字以上で入力してください。')
        long = self.call('/api/mobile/v1/ban/appeal', token, 'POST', json={'message': 'x' * 3001})
        self.assertEqual(long.json['message'], '内容は3000文字以内で入力してください。')
        sent = self.call('/api/mobile/v1/ban/appeal', token, 'POST', json={'message': '  誤判定だと思います。確認をお願いします。  '})
        self.assertEqual(sent.status_code, 200)
        self.assertEqual(sent.json['latest_appeal']['status'], 'new')
        with target.app.app_context():
            appeal = target.BanAppeal.query.filter_by(user_id=self.user_id).one()
            self.assertEqual(appeal.message, '誤判定だと思います。確認をお願いします。')
            self.assertEqual(appeal.ban_reason, 'rapid_send')
            appeal.admin_reply = '確認しました。'
            user = target.db.session.get(target.User, self.user_id)
            user.appeal_blocked = True
            user.appeal_block_reason = '繰り返しの申し立て'
            target.db.session.commit()
        self.assertEqual(self.call('/api/mobile/v1/ban', token).json['latest_appeal']['admin_reply'], '確認しました。')
        blocked = self.call('/api/mobile/v1/ban/appeal', token, 'POST', json={'message': 'x' * 20})
        self.assertEqual(blocked.status_code, 400)
        self.assertEqual(blocked.json['message'], '繰り返しの申し立て')
        self.assertTrue(blocked.json['appeal_blocked'])
        with mock.patch.object(target, '_bot_lock_info', return_value=(True, '', 300)):
            self.assertEqual(self.call('/api/mobile/v1/ban', token).status_code, 200)
        self.assertEqual(self.call('/api/mobile/v1/ban', token, 'DELETE').json['error'], 'insufficient_scope')

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
