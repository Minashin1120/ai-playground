"""Feedback sent or handled by an AI agent (scripts/send_agent_feedback.py, scripts/update_feedback_status.py)."""
import json
import os
import tempfile
import unittest
from unittest import mock

CSRF = 'csrf-test-token'


class FeedbackAgentTests(unittest.TestCase):
    def setUp(self):
        import app as target
        self.target = target
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=['localhost'])
        target._ensure_temp_chat_monitor_running = lambda: None
        directory = tempfile.TemporaryDirectory(prefix='feedback-agent-')
        self.addCleanup(directory.cleanup)
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            admin = target.User(username='feedback-admin', is_setup_completed=True, is_admin=True)
            admin.set_password('test-password')
            target.db.session.add(admin)
            target.db.session.commit()
            self.admin_id = admin.id
            agent_fb = target.Feedback(user_id=admin.id, title='agent', message='m', agent_name='Claude Code')
            plain_fb = target.Feedback(user_id=admin.id, title='plain', message='m')
            target.db.session.add_all([agent_fb, plain_fb])
            target.db.session.commit()
            self.agent_id, self.plain_id = agent_fb.public_id, plain_fb.public_id
        for patcher in [
            mock.patch.object(target, '_bot_turnstile_active', return_value=False),
            mock.patch.object(target, 'rate_limit', return_value=True),
            mock.patch.dict(target.app.config, FEEDBACK_DIR=directory.name),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = target.app.test_client()
        with self.client.session_transaction() as sess:
            sess['_user_id'] = str(self.admin_id)
            sess['_fresh'] = True
            sess['csrf_token'] = CSRF

    def update(self, public_id, status, **kwargs):
        return self.client.post(f'/api/feedback/{public_id}/update', json={'status': status},
                                headers={'X-CSRF-Token': CSRF, **kwargs.pop('headers', {})},
                                base_url='https://localhost', **kwargs)

    def saved(self, public_id):
        with open(os.path.join(self.target._feedback_dir(public_id), 'feedback.json'), encoding='utf-8') as handle:
            return json.load(handle)

    def test_list_names_the_agent_instead_of_only_the_account(self):
        res = self.client.get('/api/feedback?all=1', base_url='https://localhost')
        items = {item['public_id']: item for item in res.get_json()['items']}
        self.assertEqual(items[self.agent_id]['agent_name'], 'Claude Code')
        self.assertEqual(items[self.plain_id]['agent_name'], '')

    def test_status_update_records_the_agent_from_the_environ(self):
        res = self.update(self.agent_id, 'resolved',
                          environ_base={self.target.FEEDBACK_AGENT_ENVIRON: 'Claude Code'})
        self.assertEqual(res.status_code, 200)
        saved = self.saved(self.agent_id)
        self.assertEqual((saved['status'], saved['handled_by'], saved['agent_name']),
                         ('resolved', 'Claude Code', 'Claude Code'))

    def test_status_update_from_http_keeps_the_account_name(self):
        res = self.update(self.plain_id, 'in_review', headers={'Aiplayground.Feedback-Agent': 'Spoofed'})
        self.assertEqual(res.status_code, 200)
        saved = self.saved(self.plain_id)
        self.assertEqual((saved['status'], saved['handled_by'], saved['agent_name']),
                         ('in_review', 'feedback-admin', ''))

    def test_fixed_feedback_can_wait_for_release_then_be_resolved(self):
        for status in ('resolved_unreleased', 'resolved'):
            with self.subTest(status=status):
                res = self.update(self.agent_id, status,
                                  environ_base={self.target.FEEDBACK_AGENT_ENVIRON: 'Codex'})
                self.assertEqual(res.status_code, 200)
                self.assertEqual(self.saved(self.agent_id)['status'], status)
                listed = self.client.get('/api/feedback?all=1', base_url='https://localhost').get_json()['items']
                item = next(item for item in listed if item['public_id'] == self.agent_id)
                self.assertEqual((item['status'], item['handled_by']), (status, 'Codex'))


if __name__ == '__main__':
    unittest.main()
