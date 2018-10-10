import sentiment
import unittest


class TestSentiment(unittest.TestCase):
    def test_flow(self):
        data = [
            ("I like the world", {"pos": 1, "neg": 0}),
            ("I love the world", {"pos": 1, "neg": 0}),
            ("I hate the world", {"pos": 0, "neg": 1}),
        ]

        model = sentiment.build_model(data)

        result = sentiment.predict(model, "I hate you")
        self.assertGreater(result['neg'], result['pos'])

        result = sentiment.predict(model, "I love you")
        self.assertGreater(result['pos'], result['neg'])
