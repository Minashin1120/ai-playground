from tests.mobile_api_common import *
from tests.mobile_api_common import MobileApiTests


class MobileApiTestsThreads(MobileApiTests):
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

    def test_thread_messages_carry_created_at_for_the_branch_manager(self):
        token = self.token()
        with target.app.app_context():
            thread = target.Thread(user_id=self.user_id, public_id=target.generate_thread_public_id())
            target.db.session.add(thread)
            target.db.session.commit()
            target.db.session.add(target.Message(thread_id=thread.id, role='user', content='hi',
                                                 timestamp=datetime(2026, 9, 27, 3, 4, 5)))
            target.db.session.commit()
            public_id = thread.public_id
        messages = self.call('/api/threads/' + public_id + '?limit=50', token).json['messages']
        self.assertEqual(messages[0]['created_at'], '2026-09-27T03:04:05Z')

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

    def test_native_title_generation_is_owner_scoped(self):
        token = self.token()
        thread_id = self.call('/api/threads', token, 'POST', json={}).json['id']
        with target.app.app_context():
            thread = target.resolve_thread_for_user(thread_id, self.user_id)
            target.db.session.add(target.Message(thread_id=thread.id, role='user', content='A test question', is_encrypted=False))
            foreign = target.Thread(user_id=self.other_id, public_id=target.generate_thread_public_id())
            target.db.session.add(foreign)
            target.db.session.commit()
            foreign_id = foreign.public_id
        with mock.patch.object(target, '_get_model_specific_api_key', return_value=None), \
             mock.patch.object(target, '_admin_env_fallback_enabled', return_value=False), \
             mock.patch.object(target, '_resolve_gemini_runtime', return_value={}):
            response = self.call('/api/generate_title', token, 'POST', json={'thread_id': thread_id})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['title'], 'A test question')
        self.assertEqual(self.call('/api/generate_title', token, 'POST', json={'thread_id': foreign_id}).status_code, 403)
        self.assertEqual(self.call('/api/generate_title', 'invalid', 'POST', json={'thread_id': thread_id}).status_code, 401)

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

    def test_native_thread_gem_is_saved_like_web_settings(self):
        # Web `saveThreadGemUuid`: applying or clearing a Gem stores it on the open chat right away.
        with target.app.app_context():
            mine = target.Skill(uuid='22222222-2222-2222-2222-222222222222', user_id=self.user_id, name='Mine', instruction='Yes')
            foreign = target.Skill(uuid='33333333-3333-3333-3333-333333333333', user_id=self.other_id, name='Foreign', instruction='No')
            thread = target.Thread(user_id=self.user_id, public_id=target.generate_thread_public_id())
            target.db.session.add_all([mine, foreign, thread])
            target.db.session.commit()
            thread_id = thread.public_id
        token = self.token()
        saved = self.call('/api/mobile/v1/preferences', token, 'PUT',
                          json={'last_skill_uuid': '22222222-2222-2222-2222-222222222222', 'thread_id': thread_id})
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(self.call(f'/api/threads/{thread_id}', token).json['last_skill_uuid'], '22222222-2222-2222-2222-222222222222')
        rejected = self.call('/api/mobile/v1/preferences', token, 'PUT',
                             json={'last_skill_uuid': '33333333-3333-3333-3333-333333333333', 'thread_id': thread_id})
        self.assertEqual(rejected.status_code, 403)
        cleared = self.call('/api/mobile/v1/preferences', token, 'PUT', json={'last_skill_uuid': None, 'thread_id': thread_id})
        self.assertEqual(cleared.status_code, 200)
        self.assertIsNone(self.call(f'/api/threads/{thread_id}', token).json['last_skill_uuid'])

    def test_native_thread_legacy_gem_field_is_accepted(self):
        with target.app.app_context():
            skill = target.Skill(uuid='44444444-4444-4444-4444-444444444444',
                                 user_id=self.user_id, name='Legacy Field Skill', instruction='Yes')
            thread = target.Thread(user_id=self.user_id, public_id=target.generate_thread_public_id())
            target.db.session.add_all([skill, thread])
            target.db.session.commit()
            thread_id = thread.public_id
        token = self.token()
        saved = self.call('/api/mobile/v1/preferences', token, 'PUT',
                          json={'last_gem_uuid': '44444444-4444-4444-4444-444444444444', 'thread_id': thread_id})
        self.assertEqual(saved.status_code, 200)
        payload = self.call(f'/api/threads/{thread_id}', token).json
        self.assertEqual(payload['last_skill_uuid'], '44444444-4444-4444-4444-444444444444')
        self.assertEqual(payload['last_gem_uuid'], '44444444-4444-4444-4444-444444444444')

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
