from tests.mobile_api_common import *
from tests.mobile_api_common import MobileApiTests


class MobileApiTestsConfig(MobileApiTests):
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
        self.assertTrue(catalog['gpt-realtime-whisper']['selectable'])
        self.assertEqual(catalog['gemini-embedding-2']['mode'], 'embedding')
        self.assertTrue(catalog['gemini-embedding-2']['selectable'])
        self.assertEqual(catalog['lyria-3.5']['mode'], 'music')
        self.assertTrue(catalog['lyria-3.5']['selectable'])
        self.assertTrue(catalog['lyria-realtime-exp']['selectable'])
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

    def test_browser_csrf_is_preserved(self):
        bad = self.browser.post('/android/connect', base_url='https://localhost', data={'user_code': 'ABCDEF012345', 'decision': 'approve'})
        self.assertEqual(bad.status_code, 403)
        self.assertEqual(self.browser.post('/api/threads', base_url='https://localhost', json={}).status_code, 403)

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

    def test_public_auth_rejects_browser_origin_cookie_and_non_json(self):
        for header in [{'Origin': 'https://evil.example'}, {'Cookie': 'session=test'}]:
            response = self.native.post('/api/mobile/v1/device', base_url='https://localhost', headers=header,
                                        json={'client_id': 'official-android'})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.native.post('/api/mobile/v1/device', base_url='https://localhost', json=[]).status_code, 400)
        self.assertEqual(self.native.post('/api/mobile/v1/device', base_url='https://localhost', data='{}').status_code, 415)

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

    def test_mobile_config_describes_the_server_for_the_login_screen(self):
        with mock.patch.dict(target.os.environ, {
            'GOOGLE_CLIENT_ID': 'cid', 'GOOGLE_CLIENT_SECRET': 'secret-value',
            'ANDROID_APP_LINK_SHA256': 'AA:BB', 'MOBILE_SERVER_NAME': 'Example AI',
        }):
            config = self.native.get('/api/mobile/v1/config', base_url='https://localhost').json
        self.assertEqual(config['api_version'], 1)
        self.assertEqual(config['client_id'], 'official-android')
        self.assertEqual(config['server_name'], 'Example AI')
        self.assertIn('app_scheme', config['auth_callback_modes'])
        self.assertTrue(config['app_links_configured'])
        methods = config['auth_methods']
        self.assertTrue(methods['password'] and methods['passkey'] and methods['google_browser'])
        self.assertNotIn('secret-value', json.dumps(config))
        with mock.patch.dict(target.os.environ, {'ANDROID_APP_LINK_SHA256': '', 'GOOGLE_CLIENT_SECRET': ''}):
            bare = self.native.get('/api/mobile/v1/config', base_url='https://localhost').json
        self.assertFalse(bare['auth_methods']['passkey'])
        self.assertFalse(bare['auth_methods']['google_browser'])
        self.assertEqual(bare['server_name'], 'localhost')

    def test_turnstile_can_return_through_the_app_scheme(self):
        token = self.token()
        with mock.patch.object(target, '_bot_turnstile_active', return_value=True), \
             mock.patch.object(target, '_bot_turnstile_verified', return_value=False):
            url = self.call('/api/mobile/v1/security/turnstile', token, 'POST', json={}).json['turnstile_url']
            page = self.browser.get(url + '&return=app', base_url='https://localhost')
            self.assertIn('name="return" value="app"', page.data.decode())
            csrf_token = page.data.decode().split('name="csrf_token" value="', 1)[1].split('"', 1)[0]
            with mock.patch.object(target, 'verify_turnstile', return_value=True):
                verified = self.browser.post('/android/integrity/turnstile/verify', base_url='https://localhost', data={
                    'challenge': url.split('=', 1)[1], 'csrf_token': csrf_token,
                    'cf-turnstile-response': 'ok', 'return': 'app'})
        self.assertEqual(verified.status_code, 303)
        self.assertTrue(verified.location.startswith('com.minashin1120.aiplayground.auth:/callback?turnstile_ticket='))
        # Unknown return values fall back to the https callback.
        self.assertIsNone(target._mobile_return_mode('https://evil.example'))

    def test_browser_login_app_scheme_requires_pkce(self):
        oauth_redirect = target.redirect('https://accounts.google.com/o/oauth2/auth')
        with mock.patch.object(target.oauth.google, 'authorize_redirect', return_value=oauth_redirect):
            rejected = self.browser.get('/android/auth/google/start?return=app', base_url='https://localhost')
            self.assertEqual(rejected.status_code, 302)
            self.assertEqual(rejected.location, 'com.minashin1120.aiplayground.auth:/callback?error=pkce_required')
            challenge = 'A' * 43
            started = self.browser.get(f'/android/auth/google/start?return=app&code_challenge={challenge}&code_challenge_method=S256',
                                       base_url='https://localhost')
            self.assertEqual(started.location, 'https://accounts.google.com/o/oauth2/auth')
            with self.browser.session_transaction() as sess:
                self.assertEqual(sess.get('mobile_native_return'), 'app')
            with self.browser.session_transaction() as sess:
                sess['mobile_native_return'] = 'app'
            with target.app.test_request_context('/', base_url='https://localhost'):
                target.session['mobile_native_return'] = 'app'
                self.assertEqual(target._mobile_native_redirect(code='abc').location,
                                 'com.minashin1120.aiplayground.auth:/callback?code=abc')
                # Without the flag the App Link callback on this host is used.
                self.assertEqual(target._mobile_native_redirect(error='x').location,
                                 'https://localhost/android/auth/callback?error=x')

    def test_native_gate_can_be_relaxed_only_by_the_operator_without_turnstile(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.set_password('gate-password')
            target.db.session.commit()
        body = {'username': 'android-owner', 'password': 'gate-password', 'device_name': 'Pixel'}
        strict = self.native.post('/api/mobile/v1/auth/login', base_url='https://localhost', json=body)
        self.assertEqual(strict.status_code, 428)
        with mock.patch.dict(target.os.environ, {'MOBILE_NATIVE_GATE': 'turnstile_optional', 'TURNSTILE_SECRET_KEY': 'configured'}):
            still_strict = self.native.post('/api/mobile/v1/auth/login', base_url='https://localhost', json=body)
        self.assertEqual(still_strict.status_code, 428)
        with mock.patch.dict(target.os.environ, {'MOBILE_NATIVE_GATE': 'turnstile_optional', 'TURNSTILE_SECRET_KEY': ''}):
            relaxed = self.native.post('/api/mobile/v1/auth/login', base_url='https://localhost', json=body)
        self.assertEqual(relaxed.status_code, 200)
        self.assertEqual(relaxed.json['status'], 'ok')

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
        self.assertIn('delegate_permission/common.get_login_creds', links.json[0]['relation'])

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

    def test_html_server_failure_is_not_account_restriction(self):
        from flask import g
        with target.app.test_request_context('/api/messages/1', method='DELETE'):
            g.mobile_request = True
            g.mobile_bearer_request = True
            response = target.mobile_response_headers(target.app.make_response(('failure', 500)))
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json['error'], 'server_error')

    def test_retired_pairing_rejects_requests_and_existing_grants(self):
        device_code = 'A' * 43
        grant_key = 'mobile:grant:' + target._mobile_digest(device_code)
        self.redis.set(grant_key, json.dumps({'status': 'approved', 'encrypted_token': 'old'}), ex=60)
        for endpoint, body in [
            ('device', {'client_id': 'official-android', 'device_name': 'Test Android'}),
            ('token', {'client_id': 'official-android', 'device_code': device_code}),
        ]:
            response = self.native.post('/api/mobile/v1/' + endpoint, base_url='https://localhost', json=body)
            self.assertEqual(response.status_code, 410)
            self.assertEqual(response.json['error'], 'legacy_pairing_disabled')
        for client in [self.browser, target.app.test_client()]:
            page = client.get('/android/connect?code=ABCDEF012345', base_url='https://localhost')
            self.assertEqual(page.status_code, 410)
            self.assertIn('連携は終了しました', page.get_data(as_text=True))
            self.assertNotIn('<form', page.get_data(as_text=True))
        response = self.browser.post('/android/connect', base_url='https://localhost', data={
            'csrf_token': 'browser-csrf', 'user_code': 'ABCDEF012345', 'decision': 'approve',
        })
        self.assertEqual(response.status_code, 410)
        with target.app.app_context():
            self.assertEqual(target.UserSession.query.filter(target.UserSession.session_id.startswith('android:')).count(), 0)
        config = self.native.get('/api/mobile/v1/config', base_url='https://localhost').json
        self.assertFalse(config['legacy_auth_enabled'])
        self.assertIsNone(config['legacy_auth_flow'])
