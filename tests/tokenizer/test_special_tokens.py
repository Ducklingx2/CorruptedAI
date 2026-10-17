import unittest

from LLM.tokenizer import (
    BOS_ID, EOS_ID, PAD_ID, UNK_ID, SPECIAL_TOKENS,
    BPETokenizer, CharacterTokenizer, WordTokenizer,
)


class SpecialTokenTests(unittest.TestCase):
    def test_ids_are_stable(self):
        self.assertEqual((PAD_ID, UNK_ID, BOS_ID, EOS_ID), (0, 1, 2, 3))
        self.assertEqual(SPECIAL_TOKENS, ("<PAD>", "<UNK>", "<BOS>", "<EOS>"))

    def _trained(self):
        text = "hello world < > EOS BOS"
        w, c, b = WordTokenizer(), CharacterTokenizer(), BPETokenizer()
        w.train(text); c.train(text); b.train(text, vocab_size=270)
        return [w, c, b]

    def test_add_special_tokens_wraps_with_bos_eos(self):
        for tok in self._trained():
            with self.subTest(tok=type(tok).__name__):
                ids = tok.encode("hello", add_special_tokens=True)
                self.assertEqual(ids[0], BOS_ID)
                self.assertEqual(ids[-1], EOS_ID)
                self.assertEqual(ids[1:-1], tok.encode("hello"))
                self.assertEqual(tok.encode("", add_special_tokens=True), [BOS_ID, EOS_ID])

    def test_literal_special_token_text_is_ordinary_text(self):
        for tok in self._trained():
            with self.subTest(tok=type(tok).__name__):
                ids = tok.encode("<EOS> <BOS> <PAD>")
                self.assertNotIn(EOS_ID, ids)
                self.assertNotIn(BOS_ID, ids)
                self.assertNotIn(PAD_ID, ids)

    def test_skip_special_tokens_on_decode(self):
        for tok in self._trained():
            with self.subTest(tok=type(tok).__name__):
                ids = tok.encode("hello", add_special_tokens=True)
                self.assertEqual(tok.decode(ids, skip_special_tokens=True), tok.decode(tok.encode("hello")))
                self.assertIn("<BOS>", tok.decode(ids))


if __name__ == "__main__":
    unittest.main()
