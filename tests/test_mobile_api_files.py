from tests.mobile_api_common import *
from tests.mobile_api_common import MobileApiTests


class MobileApiTestsFiles(MobileApiTests):
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

    def test_native_mic_transcription_and_ai_settings_use_bearer(self):
        token = self.token()
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.mic_transcribe_mode = 'llm'
            target.db.session.commit()
        with mock.patch.object(target, '_transcribe_audio_with_llm', return_value='こんにちは') as transcriber:
            response = self.call('/transcribe', token, 'POST', data={
                'file': (io.BytesIO(b'fake-m4a'), 'recording.m4a'), 'llm_model': 'gemini-3.6-flash',
            }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['transcript'], 'こんにちは')
        self.assertEqual(transcriber.call_args.args[2], 'gemini-3.6-flash')
        decision = {'action': 'inspect_settings', 'arguments': {'fields': ['enter_to_send']}}
        with mock.patch.object(target, '_call_llm_for_settings_ai', return_value=(decision, None)):
            response = self.call('/api/settings/apply-ai-prompt', token, 'POST',
                                 json={'prompt': 'Enterで送信は？', 'model': 'gemini-3.6-flash', 'conversation': []})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['mode'], 'inspect')
        self.assertIn('enter_to_send', response.json['current'])

    def test_native_one_shot_sts_is_owner_scoped(self):
        token = self.token()
        own = self.call('/api/threads', token, 'POST', json={})
        self.assertEqual(own.status_code, 200)
        with target.app.app_context():
            foreign = target.Thread(user_id=self.other_id, title='other')
            target.db.session.add(foreign)
            target.db.session.commit()
            foreign_id = foreign.id

        def clip():
            return (io.BytesIO(b'fake-m4a'), 'sts_recording.m4a')

        response = self.call('/sts', token, 'POST', data={'file': clip(), 'model': 'gpt-transcribe', 'thread_id': str(foreign_id)},
                             content_type='multipart/form-data')
        self.assertEqual(response.status_code, 403)
        response = self.call('/sts', token, 'POST', data={'file': clip(), 'model': 'gpt-5.6-sol', 'thread_id': str(own.json['id'])},
                             content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        with mock.patch.object(target, '_get_model_specific_api_key', return_value='sk-test'), \
             mock.patch.object(target, '_convert_audio_to_pcm', return_value=b'\x00\x00' * 240), \
             mock.patch.object(target, '_openai_realtime_transcribe', new=mock.AsyncMock(return_value='こんにちは')):
            response = self.call('/sts', token, 'POST', data={'file': clip(), 'model': 'gpt-transcribe', 'thread_id': str(own.json['id'])},
                                 content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        line = json.loads(response.data.decode().splitlines()[0])
        self.assertTrue(line['final'])
        self.assertEqual(line['transcript'], 'こんにちは')

        # Non-live Grok Voice Transcribe 2.0 goes through the batch /v1/stt helper.
        with mock.patch.object(target, '_get_model_specific_api_key', return_value='xai-test'), \
             mock.patch.object(target, '_xai_stt_transcript', return_value=('おはよう', None)) as stt:
            response = self.call('/sts', token, 'POST', data={'file': clip(), 'model': 'grok-voice-transcribe-2.0-file', 'thread_id': str(own.json['id'])},
                                 content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        line = json.loads(response.data.decode().splitlines()[0])
        self.assertTrue(line['final'])
        self.assertTrue(line['transcription_only'])
        self.assertEqual(line['transcript'], 'おはよう')
        self.assertEqual(stt.call_args.args[2], 'grok-voice-transcribe-2.0')
        self.assertEqual(stt.call_args.args[3], 'xai-test')
        with mock.patch.object(target, '_get_model_specific_api_key', return_value='xai-test'), \
             mock.patch.object(target, '_xai_stt_transcript', return_value=(None, ('xAI STT error 429: slow down', 429))):
            response = self.call('/sts', token, 'POST', data={'file': clip(), 'model': 'grok-voice-transcribe-2.0-file', 'thread_id': str(own.json['id'])},
                                 content_type='multipart/form-data')
        self.assertEqual(response.status_code, 429)

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
        usage = self.call('/api/files/usage?filepath=' + filename, token)
        self.assertEqual(usage.status_code, 200)
        self.assertEqual(usage.json['chats'], [])
        with target.app.app_context():
            foreign = target.User(username='library-other', is_setup_completed=True)
            target.db.session.add(foreign)
            target.db.session.commit()
            foreign_id = foreign.id
        self.assertEqual(self.call('/api/files/favorite', token, 'POST',
                                   json={'filepath': f'{foreign_id}/secret.txt'}).status_code, 403)
        self.assertEqual(self.call(f'/api/files/usage?filepath={foreign_id}/secret.txt', token).status_code, 403)
        deleted = self.call('/api/files/delete', token, 'POST', json={'filenames': [filename]})
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse(self.call('/files/' + filename, token).status_code == 200)
        # Skill CRUD is owner scoped as well.  The legacy /api/gems alias stays.
        created = self.call('/api/skills', token, 'POST', json={'name': 'Android Skill', 'instruction': 'Be brief'})
        self.assertEqual(created.status_code, 200)
        skill_uuid = created.json['uuid']
        listed = self.call('/api/skills', token)
        self.assertEqual(listed.status_code, 200)
        self.assertTrue(any(skill['uuid'] == skill_uuid for skill in listed.json))
        updated = self.call('/api/skills/' + skill_uuid, token, 'PUT', json={'name': 'Renamed Skill'})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json['name'], 'Renamed Skill')
        legacy_created = self.call('/api/gems', token, 'POST', json={'name': 'Legacy Gem', 'instruction': 'Old'})
        self.assertEqual(legacy_created.status_code, 200)
        legacy_uuid = legacy_created.json['uuid']
        legacy_listed = self.call('/api/gems', token)
        self.assertEqual(legacy_listed.status_code, 200)
        self.assertIn('fixed_prompts', next(row for row in legacy_listed.json if row['uuid'] == legacy_uuid))
        with target.app.app_context():
            foreign_skill = target.Skill(uuid='11111111-1111-1111-1111-111111111111',
                                     user_id=self.other_id, name='Foreign', instruction='No')
            target.db.session.add(foreign_skill)
            target.db.session.commit()
        for endpoint in ('/api/skills/', '/api/gems/'):
            for method in ['GET', 'PUT', 'DELETE']:
                payload = {'name': 'No'} if method == 'PUT' else None
                response = self.call(endpoint + '11111111-1111-1111-1111-111111111111', token, method,
                                     **({'json': payload} if payload is not None else {}))
                self.assertEqual(response.status_code, 403, (endpoint, method))
        self.assertEqual(self.call('/api/skills/' + skill_uuid, token, 'DELETE').status_code, 200)
        self.assertEqual(self.call('/api/gems/' + legacy_uuid, token, 'DELETE').status_code, 200)

    def test_native_feedback_and_mcp_list_are_owner_scoped(self):
        token = self.token()
        created = self.call('/api/feedback', token, 'POST', json={'title': 'Android', 'message': 'native note'})
        self.assertEqual(created.status_code, 200)
        self.assertRegex(created.json['public_id'], r'^[0-9a-f]{12}$')
        listed = self.call('/api/feedback', token)
        self.assertEqual(listed.status_code, 200)
        self.assertTrue(any(item['message'] == 'native note' and item['public_id'] == created.json['public_id'] for item in listed.json['items']))
        servers = self.call('/api/mcp/servers', token)
        self.assertEqual(servers.status_code, 200)
        self.assertIn('servers', servers.json)
        # Custom servers with secrets stay on Web; the native client may only add unauthenticated ones.
        for secret in ({'auth_type': 'bearer', 'bearer_token': 't'}, {'auth_type': 'oauth'}, {'bearer_token': 't'}):
            body = {'name': 'x', 'url': 'https://example.com/mcp', **secret}
            self.assertEqual(self.call('/api/mcp/servers', token, 'POST', json=body).status_code, 403)
        self.assertEqual(self.call('/api/mcp/servers/999999/test', token, 'POST', json={}).status_code, 404)
        self.assertEqual(self.call('/api/mcp/oauth-client', token, 'PUT', json={}).status_code, 403)

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
