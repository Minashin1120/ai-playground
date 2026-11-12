from tests.mobile_api_common import *
from tests.mobile_api_common import MobileApiTests


class MobileApiTestsAccount(MobileApiTests):
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
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertTrue(user.default_enable_thinking)
            self.assertEqual(user.temp_chat_timeout_seconds, 900)
            self.assertEqual(user.system_prompt, 'Be concise.')

    def test_native_provider_keys_are_write_only_like_web_settings(self):
        token = self.token()
        saved = self.call('/api/mobile/v1/preferences', token, 'PUT', json={
            'openai_key': 'sk-live-secret',
            'model_api_keys': {'gpt-5.6': 'sk-model-secret'},
            'gemini_backend': 'vertex_ai',
            'gemini_vertex_project': 'my-project',
            'google_project': 'tts-project',
        })
        self.assertEqual(saved.status_code, 200)
        body = json.dumps(saved.json)
        self.assertNotIn('sk-live-secret', body)
        self.assertNotIn('sk-model-secret', body)
        self.assertEqual(saved.json['openai_key'], target._SECRET_MASK)
        self.assertEqual(saved.json['gemini_key'], '')
        self.assertEqual(saved.json['model_api_keys'], {'gpt-5.6': target._SECRET_MASK})
        self.assertEqual(saved.json['gemini_backend'], 'vertex_ai')
        self.assertEqual(saved.json['gemini_vertex_project'], 'my-project')
        self.assertEqual(saved.json['google_project'], 'tts-project')
        # Sending the mask back keeps the stored value; an empty string clears it.
        kept = self.call('/api/mobile/v1/preferences', token, 'PUT', json={
            'openai_key': target._SECRET_MASK, 'model_api_keys': {'gpt-5.6': target._SECRET_MASK},
        })
        self.assertEqual(kept.status_code, 200)
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertNotEqual(user.openai_api_key, 'sk-live-secret')
            self.assertEqual(target.decrypt_val(user.openai_api_key), 'sk-live-secret')
            self.assertEqual(target._load_user_model_api_key_map(user), {'gpt-5.6': 'sk-model-secret'})
        cleared = self.call('/api/mobile/v1/preferences', token, 'PUT', json={'openai_key': '', 'model_api_keys': {}})
        self.assertEqual(cleared.json['openai_key'], '')
        self.assertEqual(cleared.json['model_api_keys'], {})
        self.assertEqual(self.call('/api/mobile/v1/preferences', token, 'PUT',
                                   json={'openai_key': 'x' * 5000}).status_code, 400)
        self.assertEqual(self.call('/api/mobile/v1/preferences', token, 'PUT',
                                   json={'gemini_vertex_credentials_json': '{not json'}).status_code, 400)

    def test_native_prompt_settings_match_web_fields(self):
        token = self.token()
        prefs = self.call('/api/mobile/v1/preferences', token).json
        self.assertIn('global_system_prompt_effective', prefs)
        self.assertIn('global_system_prompt_uses_time_fallback', prefs)
        self.assertTrue(prefs['llm_transcribe_prompt_default'])
        config = prefs['auto_system_prompt_notices_config']
        self.assertIn('python', config)
        self.assertTrue(config['python']['default_text'])
        updated = self.call('/api/mobile/v1/preferences', token, 'PUT', json={
            'auto_system_prompt_notices_config': {'python': {'enabled': False, 'text': 'custom'}, 'mcp': {'enabled': False}, 'skills': {'enabled': False}},
            'llm_transcribe_prompt': 'transcribe exactly',
            'enable_latency_metrics': True,
        })
        self.assertEqual(updated.status_code, 200)
        self.assertFalse(updated.json['auto_system_prompt_notices_config']['python']['enabled'])
        self.assertEqual(updated.json['auto_system_prompt_notices_config']['python']['text'], 'custom')
        # MCP / Skills follow their prompt-bar switches and cannot be turned off here (same as Web).
        self.assertTrue(updated.json['auto_system_prompt_notices_config']['mcp']['enabled'])
        self.assertTrue(updated.json['auto_system_prompt_notices_config']['skills']['enabled'])
        self.assertEqual(updated.json['llm_transcribe_prompt'], 'transcribe exactly')
        self.assertTrue(updated.json['enable_latency_metrics'])
        self.assertEqual(self.call('/api/mobile/v1/preferences', token, 'PUT',
                                   json={'auto_system_prompt_notices_config': 'nope'}).status_code, 400)
        self.assertEqual(self.call('/api/mobile/v1/preferences', token, 'PUT',
                                   json={'passkey_only_login': True}).status_code, 400)

    def test_native_sensitive_account_actions_require_recent_reauth(self):
        password = self.set_password()
        token = self.token()
        for path, body in (
            ('/api/mobile/v1/account/delete', {}),
            ('/api/mobile/v1/account/easy-login', {'minutes': 5}),
            ('/api/mobile/v1/sessions/revoke_all', {}),
            ('/api/mobile/v1/security/2fa/disable', {}),
            ('/api/account/export', {}),
            ('/api/account/unlink_google', {}),
        ):
            blocked = self.call(path, token, 'POST', json=body)
            self.assertEqual(blocked.status_code, 403, path)
            self.assertEqual(blocked.json['error'], 'reauth_required', path)
        # Cancelling an easy-login password never needs a re-authentication.
        self.assertEqual(self.call('/api/mobile/v1/account/easy-login', token, 'POST', json={'cancel': True}).status_code, 200)
        options = self.call('/api/mobile/v1/reauth/options', token, 'POST', json={})
        self.assertEqual(options.json['methods'], ['password'])
        self.assertFalse(options.json['reauthenticated'])
        self.assertEqual(self.call('/api/mobile/v1/reauth', token, 'POST', json={'method': 'password', 'password': 'wrong'}).status_code, 401)
        self.assertEqual(self.call('/api/mobile/v1/reauth', token, 'POST', json={'method': 'totp', 'code': '123456'}).status_code, 400)
        self.assertEqual(self.call('/api/mobile/v1/reauth', token, 'POST', json={'method': 'password', 'password': password}).status_code, 200)
        easy = self.call('/api/mobile/v1/account/easy-login', token, 'POST', json={'minutes': 500})
        self.assertEqual(easy.status_code, 200)
        self.assertEqual(easy.json['minutes'], 120)
        self.assertTrue(easy.json['temp_password'])
        # The re-authentication belongs to this device token only.
        other = self.token()
        self.assertEqual(self.call('/api/mobile/v1/account/easy-login', other, 'POST', json={'minutes': 5}).status_code, 403)

    def test_native_admin_thread_encryption_toggle_requires_reauth(self):
        # The chat-screen lock modal: an admin converts their own thread, only after a re-authentication.
        password = self.set_password()
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.is_admin = True
            thread = target.Thread(user_id=self.user_id, public_id=target.generate_thread_public_id())
            foreign = target.Thread(user_id=self.other_id, public_id=target.generate_thread_public_id())
            target.db.session.add_all([thread, foreign])
            target.db.session.flush()
            target.db.session.add(target.Message(thread_id=thread.id, role='user',
                                                 content=target.encrypt_val('secret'), is_encrypted=True))
            target.db.session.commit()
            thread_id, foreign_id = thread.public_id, foreign.public_id
        token = self.token()
        path = f'/api/admin/threads/{thread_id}/encryption'
        blocked = self.call(path, token, 'POST', json={'enable': False})
        self.assertEqual(blocked.status_code, 403)
        self.assertEqual(blocked.json['error'], 'reauth_required')
        self.assertEqual(self.call('/api/mobile/v1/reauth', token, 'POST', json={'method': 'password', 'password': password}).status_code, 200)
        decrypted = self.call(path, token, 'POST', json={'enable': False})
        self.assertEqual(decrypted.status_code, 200, decrypted.get_data(as_text=True))
        self.assertEqual(decrypted.json['changed'], 1)
        with target.app.app_context():
            message = target.db.session.query(target.Message).join(target.Thread).filter(target.Thread.public_id == thread_id).one()
            self.assertFalse(message.is_encrypted)
            self.assertEqual(message.content, 'secret')
        # Only the admin's own threads, and only for admins.
        self.assertEqual(self.call(f'/api/admin/threads/{foreign_id}/encryption', token, 'POST', json={'enable': True}).status_code, 404)
        with target.app.app_context():
            target.db.session.get(target.User, self.user_id).is_admin = False
            target.db.session.commit()
        self.assertEqual(self.call(path, token, 'POST', json={'enable': True}).status_code, 403)
        # Other admin routes stay off the Android token.
        self.assertEqual(self.call('/api/admin/threads', token).json['error'], 'insufficient_scope')

    def test_native_credentials_sessions_and_2fa_reset(self):
        password = self.set_password()
        token = self.token()
        other = self.token()
        renamed = self.call('/api/mobile/v1/account/credentials', token, 'POST', json={'new_username': 'android-renamed'})
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json['username'], 'android-renamed')
        self.assertEqual(self.call('/api/mobile/v1/account/credentials', token, 'POST',
                                   json={'new_username': 'another-owner'}).status_code, 409)
        self.assertEqual(self.call('/api/mobile/v1/account/credentials', token, 'POST',
                                   json={'new_password': 'new-password-1'}).json['error'], 'reauth_required')
        sessions = self.call('/api/mobile/v1/sessions', token).json['sessions']
        self.assertEqual(sum(1 for row in sessions if row['is_current']), 1)
        # Both Android devices are listed with the Web sessions of the same account.
        self.assertGreaterEqual(sum(1 for row in sessions if 'Android' in (row['user_agent'] or '')), 2)
        self.assertEqual(self.call('/api/mobile/v1/reauth', token, 'POST', json={'method': 'password', 'password': password}).status_code, 200)
        changed = self.call('/api/mobile/v1/account/credentials', token, 'POST', json={'new_password': 'new-password-1'})
        self.assertTrue(changed.json['password_changed'])
        # A password change signs out the other devices but keeps this one.
        self.assertEqual(self.call('/api/mobile/v1/me', other).status_code, 401)
        self.assertEqual(self.call('/api/mobile/v1/me', token).status_code, 200)
        disabled = self.call('/api/mobile/v1/security/2fa/disable', token, 'POST', json={})
        self.assertEqual(disabled.status_code, 200)
        self.assertFalse(disabled.json['is_2fa_enabled'])
        foreign = self.call('/api/mobile/v1/sessions/revoke', token, 'POST', json={'id': 999999})
        self.assertEqual(foreign.status_code, 404)

    def test_native_link_grant_requires_reauth_and_is_single_use(self):
        password = self.set_password()
        token = self.token()
        self.assertEqual(self.call('/api/mobile/v1/account/link/google/start', token, 'POST', json={}).json['error'], 'reauth_required')
        self.assertEqual(self.call('/api/mobile/v1/reauth', token, 'POST', json={'method': 'password', 'password': password}).status_code, 200)
        self.assertEqual(self.call('/api/mobile/v1/account/link/github/start', token, 'POST', json={}).status_code, 404)
        started = self.call('/api/mobile/v1/account/link/google/start', token, 'POST', json={})
        self.assertEqual(started.status_code, 200)
        path = started.json['path']
        self.assertTrue(path.startswith('/android/link/google?grant='))
        browser = target.app.test_client()
        grant = path.split('grant=', 1)[1]
        with mock.patch.object(target.oauth.google, 'authorize_redirect', return_value=target.redirect('https://accounts.google.com/o/oauth2')) as authorize:
            # Opening the link only shows which account will be linked.
            opened = browser.get(path, base_url='https://localhost')
            self.assertEqual(opened.status_code, 200)
            self.assertFalse(authorize.called)
            page = opened.get_data(as_text=True)
            self.assertIn('android-owner', page)
            with browser.session_transaction() as sess:
                self.assertNotIn('mobile_native_link_user', sess)
                csrf_token = sess['csrf_token']
            # Without the CSRF token the confirmation is rejected.
            self.assertEqual(browser.post('/android/link/google', data={'grant': grant},
                                          base_url='https://localhost').status_code, 403)
            opened = browser.post('/android/link/google', data={'grant': grant, 'csrf_token': csrf_token},
                                  base_url='https://localhost')
        self.assertEqual(opened.status_code, 302)
        self.assertTrue(authorize.called)
        with browser.session_transaction() as sess:
            self.assertEqual(sess['mobile_native_link_user'], self.user_id)
            self.assertEqual(sess['mobile_native_link_provider'], 'google')
        # The grant cannot be replayed from another browser.
        replay = target.app.test_client().get(path, base_url='https://localhost')
        self.assertIn('error=link_expired', replay.headers['Location'])
        # A later native sign-in in the same browser drops the pending link.
        with mock.patch.object(target.oauth.google, 'authorize_redirect', return_value=target.redirect('https://accounts.google.com/o/oauth2')):
            browser.get('/android/auth/google/start?code_challenge=' + 'a' * 43, base_url='https://localhost')
        with browser.session_transaction() as sess:
            self.assertNotIn('mobile_native_link_user', sess)

    def test_native_google_callback_links_the_granted_user(self):
        browser = target.app.test_client()
        with browser.session_transaction() as sess:
            sess['mobile_native_google'] = True
            sess['mobile_native_link_user'] = self.user_id
            sess['mobile_native_link_provider'] = 'google'
        userinfo = {'sub': 'google-sub-1', 'email': 'owner@example.com', 'email_verified': True}
        with mock.patch.object(target.oauth.google, 'authorize_access_token', return_value={'userinfo': userinfo}):
            linked = browser.get('/android/auth/google/callback', base_url='https://localhost')
        self.assertIn('linked=google', linked.headers['Location'])
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            self.assertEqual(user.google_id, 'google-sub-1')
        # Another user cannot take over an identity that is already linked.
        other = target.app.test_client()
        with other.session_transaction() as sess:
            sess['mobile_native_google'] = True
            sess['mobile_native_link_user'] = self.other_id
            sess['mobile_native_link_provider'] = 'google'
        with mock.patch.object(target.oauth.google, 'authorize_access_token', return_value={'userinfo': userinfo}):
            refused = other.get('/android/auth/google/callback', base_url='https://localhost')
        self.assertIn('error=google_already_linked', refused.headers['Location'])

    def test_native_totp_setup_returns_web_qr_and_scan_is_available(self):
        token = self.token()
        setup = self.call('/api/mobile/v1/security/totp/setup', token, 'POST', json={})
        self.assertEqual(setup.status_code, 200)
        self.assertTrue(setup.json['qr_image'].startswith('data:image/png;base64,'))
        scan = self.call('/api/encryption_scan', token)
        self.assertEqual(scan.status_code, 200)
        self.assertIn('total', scan.json)
        preview = self.call('/api/account/dedupe/preview', token, 'POST', json={})
        self.assertEqual(preview.status_code, 200)
        prefs = self.call('/api/mobile/v1/preferences', token).json
        self.assertEqual(prefs['migration_status'], 'idle')
        self.assertFalse(prefs['is_admin'])

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
        # The enabling code is spent; a code from a later 30-second step disables.
        replayed = self.call('/api/mobile/v1/security/totp/disable', token, 'POST',
                             json={'code': target.pyotp.TOTP(secret).now()})
        self.assertEqual(replayed.status_code, 401)
        self.redis.delete(*self.redis.keys('totp:used:*'))
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

    def test_account_import_after_setup_requires_reauth(self):
        token = self.token()
        # Rejected by the bearer guard before the view runs; should that regress,
        # the view still writes into a throwaway directory, never the real one.
        import_root = tempfile.TemporaryDirectory(prefix='mobile-import-guard-')
        self.addCleanup(import_root.cleanup)
        patcher = mock.patch.object(target, '_account_import_upload_root', return_value=import_root.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        # The settings Data tab imports too, but only right after a re-authentication.
        blocked = self.call('/api/account/import/upload/start', token, 'POST', json={'size': 10})
        self.assertEqual(blocked.status_code, 403)
        self.assertEqual(blocked.json['error'], 'reauth_required')
        self.assertEqual(self.call('/api/account/import', token, 'POST', json={'upload_id': 'x'}).json['error'], 'reauth_required')
        self.assertEqual(self.call('/api/account/export/latest', token).status_code, 200)
        replay = self.call('/api/mobile/v1/setup', token, 'PUT', json={'default_model': 'gemini-3.6-flash'})
        self.assertEqual(replay.status_code, 409)
