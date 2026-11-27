"""All-chat cost inputs (usage tab): period filtering and legacy fallbacks."""
import json
import unittest
from datetime import datetime, timedelta
from unittest import mock

import app as target


class UsageCostSummaryTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(target.app.config, TRUSTED_HOSTS=['localhost'])
        patcher.start()
        self.addCleanup(patcher.stop)
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            user = target.User(username='usage-owner', is_setup_completed=True)
            target.db.session.add(user)
            target.db.session.commit()
            self.user_id = user.id
        self.client = target.app.test_client()
        with self.client.session_transaction() as session:
            session['_user_id'] = str(self.user_id)
            session['_fresh'] = True
            session['csrf_token'] = 'usage-csrf'

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()

    def add_thread(self, title='Usage chat', messages=()):
        with target.app.app_context():
            thread = target.Thread(user_id=self.user_id, public_id=target.generate_thread_public_id(), title=title)
            target.db.session.add(thread)
            target.db.session.flush()
            for message in messages:
                target.db.session.add(target.Message(thread_id=thread.id, **message))
            target.db.session.commit()
            return thread.id

    def get(self, period):
        response = self.client.get(f'/api/usage/costs?period={period}')
        self.assertEqual(response.status_code, 200, response.data)
        return response.get_json()

    def test_period_limits_answers_and_keeps_saved_calls(self):
        now = datetime.utcnow()
        saved = {'pricing_calls': [{
            'model': 'gpt-6.1-sol', 'input_tokens': 1000, 'output_tokens': 500,
            'created_at': (now - timedelta(hours=1)).isoformat() + 'Z', 'complete': True,
        }]}
        self.add_thread(messages=[
            {'role': 'assistant', 'model': 'gpt-6.1-sol', 'content': 'recent', 'tokens_out': 500,
             'thought_data': json.dumps(saved), 'timestamp': now - timedelta(hours=1)},
            {'role': 'assistant', 'model': 'gpt-6.1-sol', 'content': 'old', 'tokens_out': 500,
             'thought_data': json.dumps(saved), 'timestamp': now - timedelta(days=20)},
        ])
        day = self.get('1d')
        month = self.get('1m')
        self.assertEqual(day['period'], '1d')
        self.assertEqual(day['answers'], 1)
        self.assertEqual(day['messages'][0]['pricing_calls'][0]['input_tokens'], 1000)
        self.assertEqual(month['answers'], 2)

    def test_legacy_answer_infers_input_from_parent_chain(self):
        now = datetime.utcnow()
        self.add_thread(messages=[
            {'role': 'user', 'content': 'first', 'tokens_in': 40, 'timestamp': now - timedelta(minutes=5)},
            {'role': 'assistant', 'model': 'gpt-6.1-sol', 'content': 'answer', 'tokens_out': 100,
             'tokens_thought': 20, 'timestamp': now - timedelta(minutes=4)},
        ])
        with target.app.app_context():
            messages = target.Message.query.order_by(target.Message.id).all()
            parent_id, answer_id = messages[0].id, messages[1].id
            target.db.session.get(target.Message, answer_id).parent_id = parent_id
            target.db.session.commit()
        data = self.get('7d')
        calls = data['messages'][0]['pricing_calls']
        self.assertEqual(calls[0]['input_tokens'], 40)
        self.assertEqual(calls[0]['output_tokens'], 100)
        self.assertTrue(calls[0]['complete'])

    def test_library_threads_are_not_counted(self):
        now = datetime.utcnow()
        self.add_thread(title='[LIBRARY] saved prompt', messages=[
            {'role': 'assistant', 'model': 'gpt-6.1-sol', 'content': 'library', 'tokens_out': 10,
             'tokens_in': 10, 'timestamp': now},
        ])
        self.assertEqual(self.get('1d')['answers'], 0)

    def test_unknown_period_falls_back_to_one_month(self):
        self.assertEqual(self.get('nonsense')['period'], '1m')

    def test_api_key_breakdown_labels_recorded_and_legacy_calls(self):
        now = datetime.utcnow()
        common_key = 'sk-test-common-key-1234'
        special_key = 'sk-test-special-key-9876'
        old_key = 'sk-test-retired-key-5555'
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.openai_api_key = target.encrypt_val(common_key)
            target._save_user_model_api_key_map(user, {'gpt-6.1-sol': special_key})
            target.db.session.commit()
        recorded_old_meta = target._pricing_key_metadata(old_key, 'openai_common', 'openai')
        self.assertNotIn(old_key, json.dumps(recorded_old_meta))
        self.assertEqual(recorded_old_meta['api_key_hint'], '...5555')
        saved_old = {'pricing_calls': [{
            'model': 'gpt-6.1-sol', 'input_tokens': 100, 'output_tokens': 50,
            'created_at': now.isoformat() + 'Z', 'complete': True,
            **recorded_old_meta,
        }]}
        self.add_thread(messages=[
            {'role': 'assistant', 'model': 'gpt-6.1-sol', 'content': 'old-key', 'tokens_out': 50,
             'thought_data': json.dumps(saved_old), 'timestamp': now - timedelta(minutes=2)},
            {'role': 'assistant', 'model': 'gpt-6.1-sol', 'content': 'legacy-inferred', 'tokens_in': 200,
             'tokens_out': 80, 'timestamp': now - timedelta(minutes=1)},
        ])
        data = self.get('1d')
        self.assertEqual(data['answers'], 2)
        old_call = data['messages'][0]['pricing_calls'][0]
        legacy_call = data['messages'][1]['pricing_calls'][0]
        self.assertEqual(old_call['api_key_id'], recorded_old_meta['api_key_id'])
        self.assertIn('...5555', old_call['api_key_label'])
        self.assertIn('...9876', legacy_call['api_key_label'])
        self.assertTrue(legacy_call['api_key_inferred'])
        raw_json = json.dumps(data)
        self.assertNotIn(common_key, raw_json)
        self.assertNotIn(special_key, raw_json)
        self.assertNotIn(old_key, raw_json)


if __name__ == '__main__':
    unittest.main()
