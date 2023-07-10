import unittest
from app import app


class TestSentimentAnalysis(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_prediction(self):
        response = self.client.post('/v1/inference/sentiment_analysis',
                                    json={'sentence': 'this is a bad example'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('sentiment', data)
        self.assertIn('confidence', data)


if __name__ == '__main__':
    unittest.main()
