"""Images attached to a feedback (server/feedback_images.py)."""
import io
import json
import os
import tempfile
import unittest
from unittest import mock

PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 32
JPG = b'\xff\xd8\xff\xe0' + b'\x00' * 32
WEBP = b'RIFF\x00\x00\x00\x00WEBPVP8 ' + b'\x00' * 16
GIF = b'GIF89a' + b'\x00' * 16


class FeedbackImagesTests(unittest.TestCase):
    def setUp(self):
        import app as target
        self.target = target
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=['localhost'])
        target._ensure_temp_chat_monitor_running = lambda: None
        directory = tempfile.TemporaryDirectory(prefix='feedback-images-')
        self.addCleanup(directory.cleanup)
        self.root = directory.name
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            for name in ('feedback-image-a', 'feedback-image-b'):
                user = target.User(username=name, is_setup_completed=True)
                user.set_password('test-password')
                target.db.session.add(user)
            target.db.session.commit()
            self.user_ids = [u.id for u in target.User.query.order_by(target.User.id).all()]
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

    def send(self, client):
        res = client.post('/api/feedback', json={'title': 't', 'message': 'm'},
                          headers={'X-CSRF-Token': 'csrf-test-token'}, base_url='https://localhost')
        self.assertEqual(res.status_code, 200)
        return res.get_json()['public_id']

    def upload(self, client, public_id, files):
        data = {'images': [(io.BytesIO(body), name) for name, body in files]}
        return client.post(f'/api/feedback/{public_id}/images', data=data, content_type='multipart/form-data',
                           headers={'X-CSRF-Token': 'csrf-test-token'}, base_url='https://localhost')

    def test_images_are_saved_next_to_the_feedback_and_listed(self):
        client = self.client(self.user_ids[0])
        public_id = self.send(client)
        res = self.upload(client, public_id, [('a.png', PNG), ('b.jpg', JPG)])
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {'saved': 2, 'images': 2})
        res = self.upload(client, public_id, [('c.webp', WEBP)])
        self.assertEqual(res.get_json(), {'saved': 1, 'images': 3})
        folder = os.path.join(self.root, public_id, 'images')
        self.assertEqual(sorted(os.listdir(folder)), ['01.png', '02.jpg', '03.webp'])
        with open(os.path.join(folder, '01.png'), 'rb') as handle:
            self.assertEqual(handle.read(), PNG)
        with open(os.path.join(self.root, public_id, 'feedback.json'), encoding='utf-8') as handle:
            info = json.load(handle)
        self.assertEqual([item['file'] for item in info['images']], ['01.png', '02.jpg', '03.webp'])
        listed = client.get('/api/feedback', base_url='https://localhost').get_json()['items'][0]
        self.assertEqual(listed['image_count'], 3)
        self.assertIsNone(listed['image_dir'])

    def test_rejects_other_types_large_files_too_many_and_other_users(self):
        client = self.client(self.user_ids[0])
        public_id = self.send(client)
        # The header decides: a name or declared type of an image does not make a text file one.
        self.assertEqual(self.upload(client, public_id, [('fake.png', b'<svg></svg>')]).status_code, 415)
        with mock.patch.object(self.target, '_FEEDBACK_IMAGE_MAX_BYTES', 64):
            self.assertEqual(self.upload(client, public_id, [('big.png', PNG + b'\x00' * 64)]).status_code, 413)
        self.assertEqual(self.upload(client, public_id, [('1.png', PNG), ('2.png', PNG), ('3.gif', GIF), ('4.png', PNG)]).status_code, 200)
        self.assertEqual(self.upload(client, public_id, [('5.png', PNG)]).status_code, 400)
        self.assertEqual(len(os.listdir(os.path.join(self.root, public_id, 'images'))), 4)
        other = self.client(self.user_ids[1])
        self.assertEqual(self.upload(other, public_id, [('x.png', PNG)]).status_code, 404)

    def test_deleting_the_feedback_removes_the_images(self):
        client = self.client(self.user_ids[0])
        public_id = self.send(client)
        self.upload(client, public_id, [('a.png', PNG)])
        res = client.delete(f'/api/feedback/{public_id}', headers={'X-CSRF-Token': 'csrf-test-token'}, base_url='https://localhost')
        self.assertEqual(res.status_code, 200)
        self.assertFalse(os.path.exists(os.path.join(self.root, public_id)))


if __name__ == '__main__':
    unittest.main()
