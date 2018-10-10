from sentiment import simple, lstm
import unittest


class TestSentiment(unittest.TestCase):
    def test_simple(self):
        data = [
            ("I like the world", {"pos": 1, "neg": 0}),
            ("I love the world", {"pos": 1, "neg": 0}),
            ("I hate the world", {"pos": 0, "neg": 1}),
        ]

        model = simple.build_model(data)

        result = simple.predict(model, "I hate you")
        self.assertGreater(result['neg'], result['pos'])

        result = simple.predict(model, "I love you")
        self.assertGreater(result['pos'], result['neg'])

    def test_lstm(self):
        data = [
            ("I like the world", 1),
            ("I love the world", 1),
            ("I hate the world", 0),
            ("I hate the world baby", 0),
        ]

        model = lstm.build_model(data, num_classes=2)

        result = lstm.predict(model, "I hate the world")
        self.assertEqual(0, result)

        result = lstm.predict(model, "I love you")
        self.assertEqual(1, result)