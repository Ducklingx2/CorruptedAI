import unittest

from LLM.tokenizer import (
    UNK_ID, InvalidTokenIdError, NotTrainedError, WordTokenizer,
)

CORPUS = "the cat sat on the mat. the dog barked!"


class WordTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tok = WordTokenizer()
        self.tok.train(CORPUS)

    def test_untrained_raises(self):
        with self.assertRaises(NotTrainedError):
            WordTokenizer().encode("x")
        with self.assertRaises(NotTrainedError):
            WordTokenizer().decode([])

    def test_known_round_trip(self):
        text = "the cat sat on the mat."
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_empty_input(self):
        self.assertEqual(self.tok.encode(""), [])
        self.assertEqual(self.tok.decode([]), "")
        self.assertEqual(self.tok.encode("   \n\t "), [])

    def test_unknown_words_become_unk(self):
        ids = self.tok.encode("the zebra")
        self.assertEqual(ids[1], UNK_ID)
        self.assertEqual(self.tok.decode(ids), "the <UNK>")
        self.assertEqual(self.tok.decode(ids, skip_special_tokens=True), "the")

    def test_whitespace_is_normalized_not_preserved(self):
        self.assertEqual(self.tok.encode("the   cat\n\nsat"), self.tok.encode("the cat sat"))

    def test_punctuation_is_separate_token(self):
        self.assertEqual(len(self.tok.encode("mat.")), 2)
        self.assertEqual(self.tok.decode(self.tok.encode("dog barked !")), "dog barked!")

    def test_contractions_stay_whole(self):
        t = WordTokenizer()
        t.train("don't stop")
        self.assertEqual(len(t.encode("don't")), 1)
        self.assertNotIn(UNK_ID, t.encode("don't"))

    def test_unicode(self):
        t = WordTokenizer()
        t.train("café naïve 日本語 🙂")
        ids = t.encode("café 日本語 🙂")
        self.assertNotIn(UNK_ID, ids)
        self.assertEqual(t.decode(ids), "café 日本語 🙂")
        # NFC normalization: decomposed é equals precomposed é
        self.assertEqual(t.encode("cafe\u0301"), t.encode("caf\u00e9"))

    def test_vocab_order_is_deterministic_by_frequency_then_alpha(self):
        t = WordTokenizer()
        t.train("b a a c c c")
        # specials 0-3, then c(3x)=4, a(2x)=5, b(1x)=6
        self.assertEqual(t.encode("c a b"), [4, 5, 6])
        t2 = WordTokenizer()
        t2.train("b a a c c c")
        self.assertEqual(t.to_dict(), t2.to_dict())

    def test_max_vocab_size_and_min_frequency(self):
        t = WordTokenizer(max_vocab_size=6)  # 4 specials + 2 words
        t.train("a a a b b c")
        self.assertEqual(t.vocab_size, 6)
        self.assertEqual(t.encode("c"), [UNK_ID])
        t2 = WordTokenizer(min_frequency=2)
        t2.train("a a b")
        self.assertEqual(t2.encode("b"), [UNK_ID])

    def test_lowercase_option(self):
        t = WordTokenizer(lowercase=True)
        t.train("Hello")
        self.assertEqual(t.encode("HELLO"), t.encode("hello"))
        self.assertNotIn(UNK_ID, t.encode("HELLO"))

    def test_invalid_ids(self):
        for bad in (-1, self.tok.vocab_size, 10**9, "3", 1.5, None, True):
            with self.subTest(bad=bad), self.assertRaises(InvalidTokenIdError):
                self.tok.decode([bad])

    def test_bad_constructor_args(self):
        with self.assertRaises(ValueError):
            WordTokenizer(max_vocab_size=4)
        with self.assertRaises(ValueError):
            WordTokenizer(min_frequency=0)


if __name__ == "__main__":
    unittest.main()
