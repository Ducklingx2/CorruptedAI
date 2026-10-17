import unittest

from LLM.tokenizer import (
    UNK_ID, CharacterTokenizer, InvalidTokenIdError, NotTrainedError,
)


class CharacterTokenizerTests(unittest.TestCase):
    def test_untrained_raises(self):
        with self.assertRaises(NotTrainedError):
            CharacterTokenizer().encode("a")

    def test_exact_round_trip_including_whitespace_and_punctuation(self):
        text = "Hello, World!\n\tTabs  and   spaces.\r\n"
        t = CharacterTokenizer()
        t.train(text)
        self.assertEqual(t.decode(t.encode(text)), text)

    def test_unicode_round_trip(self):
        text = "naïve café 日本語 🙂 Ω"
        t = CharacterTokenizer()
        t.train(text)
        self.assertEqual(t.decode(t.encode(text)), text)

    def test_empty(self):
        t = CharacterTokenizer()
        t.train("")
        self.assertEqual(t.vocab_size, 4)
        self.assertEqual(t.encode(""), [])
        self.assertEqual(t.decode([]), "")
        self.assertEqual(t.encode("abc"), [UNK_ID] * 3)

    def test_vocab_sorted_by_codepoint(self):
        t = CharacterTokenizer()
        t.train("cba")
        self.assertEqual(t.encode("abc"), [4, 5, 6])

    def test_unknown_characters(self):
        t = CharacterTokenizer()
        t.train("ab")
        ids = t.encode("azb")
        self.assertEqual(ids[1], UNK_ID)
        self.assertEqual(t.decode(ids), "a<UNK>b")
        self.assertEqual(t.decode(ids, skip_special_tokens=True), "ab")

    def test_nfc_normalization(self):
        t = CharacterTokenizer()
        t.train("caf\u00e9")
        self.assertEqual(t.encode("cafe\u0301"), t.encode("caf\u00e9"))
        self.assertEqual(t.decode(t.encode("cafe\u0301")), "caf\u00e9")

    def test_invalid_ids(self):
        t = CharacterTokenizer()
        t.train("ab")
        for bad in (-1, 6, "a", 2.0):
            with self.subTest(bad=bad), self.assertRaises(InvalidTokenIdError):
                t.decode([bad])


if __name__ == "__main__":
    unittest.main()
