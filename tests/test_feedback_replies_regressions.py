"""Reply history of a feedback: replies by administrators and the sender, with the activity log on request (server/routes_admin.py)."""
import json
import os
import tempfile
import unittest
from unittest import mock

CSRF = {'X-CSRF-Token': 'csrf-test-token'}
BASE = 'https://localhost'


class FeedbackRepliesTests(unittest.TestCase):
    def setUp(self):
        import app as target
        self.target = target
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=['localhost'])
        target._ensure_temp_chat_monitor_running = lambda: None
        directory = tempfile.TemporaryDirectory(prefix='feedback-replies-')
        self.addCleanup(directory.cleanup)
        self.root = directory.name
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            for name, admin in (('fb-reply-admin', True), ('fb-reply-sender', False), ('fb-reply-other', False)):
                user = target.User(username=name, is_setup_completed=True, is_admin=admin)
                user.set_password('test-password')
                target.db.session.add(user)
            target.db.session.commit()
            self.admin_id, self.sender_id, self.other_id = [u.id for u in target.User.query.order_by(target.User.id).all()]
        for patcher in [
            mock.patch.object(target, '_bot_turnstile_active', return_value=False),
            mock.patch.object(target, 'rate_limit', return_value=True),
            mock.patch.dict(target.app.config, FEEDBACK_DIR=self.root),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def client(self, user_id):
        client = self.target.app.test_client()
        with client.session_transaction() as sess:
            sess['_user_id'] = str(user_id)
            sess['_fresh'] = True
            sess['csrf_token'] = 'csrf-test-token'
        return client

    def send_feedback(self, client):
        res = client.post('/api/feedback', json={'title': 't', 'message': 'm'}, headers=CSRF, base_url=BASE)
        self.assertEqual(res.status_code, 200)
        return res.get_json()['public_id']

    def reply(self, client, public_id, body):
        return client.post(f'/api/feedback/{public_id}/replies', json=body, headers=CSRF, base_url=BASE)

    def info(self, public_id):
        with open(os.path.join(self.root, public_id, 'feedback.json'), encoding='utf-8') as handle:
            return json.load(handle)

    def listed(self, client, public_id):
        items = client.get('/api/feedback?all=1', base_url=BASE).get_json()['items']
        return next(item for item in items if item['public_id'] == public_id)

    def test_the_sender_and_an_administrator_keep_a_history(self):
        sender, admin = self.client(self.sender_id), self.client(self.admin_id)
        public_id = self.send_feedback(sender)
        self.assertEqual(self.reply(admin, public_id, {'message': 'first'}).status_code, 200)
        self.assertEqual(admin.post(f'/api/feedback/{public_id}/update', json={'status': 'replied'},
                                    headers=CSRF, base_url=BASE).status_code, 200)
        res = self.reply(sender, public_id, {'message': 'still broken'})
        self.assertEqual(res.status_code, 200)
        # A reply by the sender puts a replied feedback back to new.
        self.assertEqual(res.get_json()['feedback_status'], 'new')
        self.assertEqual(self.reply(admin, public_id, {'message': 'second'}).status_code, 200)

        mine = self.listed(sender, public_id)
        self.assertEqual([(r['role'], r['message']) for r in mine['replies']],
                         [('admin', 'first'), ('user', 'still broken'), ('admin', 'second')])
        self.assertEqual({r['author'] for r in mine['replies']}, {''})
        self.assertEqual(mine['admin_reply'], 'second')
        self.assertEqual(mine['status'], 'new')
        self.assertEqual(self.listed(admin, public_id)['replies'][0]['author'], 'fb-reply-admin')
        self.assertEqual([r['message'] for r in self.info(public_id)['replies']], ['first', 'still broken', 'second'])

    def test_only_the_sender_and_administrators_can_reply(self):
        public_id = self.send_feedback(self.client(self.sender_id))
        self.assertEqual(self.reply(self.client(self.other_id), public_id, {'message': 'x'}).status_code, 404)
        self.assertEqual(self.reply(self.client(self.sender_id), public_id, {'message': '  '}).status_code, 400)
        self.assertEqual(self.reply(self.client(self.sender_id), public_id, {'message': 'x', 'client_logs': []}).status_code, 400)

    def test_the_activity_log_goes_with_a_reply_only_when_sent(self):
        sender = self.client(self.sender_id)
        public_id = self.send_feedback(sender)
        res = self.reply(sender, public_id, {'message': 'no log'})
        self.assertNotIn('logs_saved', res.get_json())
        logs = {'client': 'web', 'version': '1', 'window_seconds': 3600, 'entries': [{'t': 1, 'ev': 'click'}]}
        res = self.reply(sender, public_id, {'message': 'with log', 'client_logs': logs})
        self.assertTrue(res.get_json()['logs_saved'])
        reply_id = res.get_json()['reply_id']
        path = os.path.join(self.root, public_id, 'logs', f'reply-{reply_id}.jsonl')
        with open(path, encoding='utf-8') as handle:
            meta, entry = [json.loads(line) for line in handle]
        self.assertEqual((meta['kind'], meta['reply_id'], meta['user_id']), ('meta', reply_id, self.sender_id))
        self.assertEqual(entry['ev'], 'click')
        self.assertFalse(os.path.exists(os.path.join(self.root, public_id, 'logs', 'activity.jsonl')))
        replies = self.info(public_id)['replies']
        self.assertEqual([r['activity_log'] for r in replies], [None, f'logs/reply-{reply_id}.jsonl'])
        admin_view = self.listed(self.client(self.admin_id), public_id)['replies']
        self.assertIsNone(admin_view[0]['log_file'])
        self.assertTrue(admin_view[1]['log_file'].endswith(f'logs/reply-{reply_id}.jsonl'))
        self.assertIsNone(self.listed(sender, public_id)['replies'][1]['log_file'])

    def test_a_reply_saved_before_the_history_stays_first(self):
        sender = self.client(self.sender_id)
        public_id = self.send_feedback(sender)
        with self.target.app.app_context():
            fb = self.target.Feedback.query.filter_by(public_id=public_id).first()
            fb.admin_reply = 'old reply'
            self.target.db.session.commit()
        self.assertEqual(self.reply(sender, public_id, {'message': 'answer'}).status_code, 200)
        # The update API (older apps, scripts/update_feedback_status.py) adds a changed reply to the history.
        admin = self.client(self.admin_id)
        admin.post(f'/api/feedback/{public_id}/update', json={'status': 'replied', 'admin_reply': 'new reply'},
                   headers=CSRF, base_url=BASE)
        admin.post(f'/api/feedback/{public_id}/update', json={'status': 'resolved', 'admin_reply': 'new reply'},
                   headers=CSRF, base_url=BASE)
        self.assertEqual([(r['role'], r['message']) for r in self.listed(sender, public_id)['replies']],
                         [('admin', 'old reply'), ('user', 'answer'), ('admin', 'new reply')])

    def test_deleting_the_feedback_or_the_account_removes_the_replies(self):
        sender = self.client(self.sender_id)
        first, second = self.send_feedback(sender), self.send_feedback(sender)
        self.reply(sender, first, {'message': 'a'})
        self.reply(sender, second, {'message': 'b'})
        self.assertEqual(sender.delete(f'/api/feedback/{first}', headers=CSRF, base_url=BASE).status_code, 200)
        with self.target.app.app_context():
            self.assertEqual([r.message for r in self.target.FeedbackReply.query.all()], ['b'])
            self.target._delete_user_account_immediately(self.target.db.session.get(self.target.User, self.sender_id))
            self.assertEqual(self.target.FeedbackReply.query.count(), 0)


if __name__ == '__main__':
    unittest.main()
