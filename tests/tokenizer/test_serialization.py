import json
import os
import tempfile
import unittest
from pathlib import Path

from LLM.tokenizer import (
    BPETokenizer, CharacterTokenizer, NotTrainedError, Tokenizer,
    TokenizerFormatError, WordTokenizer, load_tokenizer, FORMAT_VERSION,
)

TEXT = "the cat sat on the mat. café 日本語 🙂 " * 10


def make_all():
    w = WordTokenizer(lowercase=True); w.train(TEXT)
    c = CharacterTokenizer(); c.train(TEXT)
    b = BPETokenizer(); b.train(TEXT, vocab_size=300)
    return [w, c, b]


class SaveLoadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, obj, name="t.json"):
        p = self.dir / name
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
        return p

    def valid(self, kind):
        tok = {"word": make_all()[0], "character": make_all()[1], "bpe": make_all()[2]}[kind]
        p = self.dir / "ok.json"
        tok.save(p)
        return json.loads(p.read_text(encoding="utf-8"))

    def test_save_load_equivalence(self):
        for tok in make_all():
            with self.subTest(tok=type(tok).__name__):
                p = self.dir / f"{tok.TYPE}.json"
                tok.save(p)
                loaded = load_tokenizer(p)
                self.assertIs(type(loaded), type(tok))
                self.assertEqual(loaded.vocab_size, tok.vocab_size)
                self.assertEqual(loaded.to_dict(), tok.to_dict())
                for s in ["the cat", "café 🙂", "", "unseen zxq"]:
                    self.assertEqual(loaded.encode(s), tok.encode(s))
                    self.assertEqual(loaded.encode(s, add_special_tokens=True),
                                     tok.encode(s, add_special_tokens=True))

    def test_file_is_versioned_plain_json(self):
        d = self.valid("bpe")
        self.assertEqual(d["format_version"], FORMAT_VERSION)
        self.assertEqual(d["special_tokens"], {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3})
        self.assertEqual(d["type"], "bpe")

    def test_classmethod_load_checks_type(self):
        p = self.dir / "b.json"
        make_all()[2].save(p)
        self.assertIsInstance(BPETokenizer.load(p), BPETokenizer)
        with self.assertRaises(TypeError):
            WordTokenizer.load(p)

    def test_save_creates_parent_dirs_and_leaves_no_temp_files(self):
        p = self.dir / "a" / "b" / "t.json"
        make_all()[1].save(p)
        self.assertTrue(p.exists())
        self.assertEqual([f.name for f in p.parent.iterdir()], ["t.json"])

    def test_untrained_word_and_char_cannot_be_saved(self):
        for tok in (WordTokenizer(), CharacterTokenizer()):
            with self.assertRaises(NotTrainedError):
                tok.save(self.dir / "x.json")

    def test_untrained_bpe_round_trips_as_pure_bytes(self):
        p = self.dir / "x.json"
        BPETokenizer().save(p)
        self.assertEqual(load_tokenizer(p).encode("hi"), BPETokenizer().encode("hi"))

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_tokenizer(self.dir / "nope.json")

    def test_malformed_files(self):
        base_bpe = self.valid("bpe")
        base_word = self.valid("word")
        base_char = self.valid("character")

        def mod(base, **changes):
            d = dict(base)
            for k, v in changes.items():
                if v is ...:
                    d.pop(k, None)
                else:
                    d[k] = v
            return d

        cases = {
            "not json": "{ this is not json",
            "empty file": "",
            "json list": "[1, 2, 3]",
            "json string": '"hello"',
            "missing version": mod(base_bpe, format_version=...),
            "future version": mod(base_bpe, format_version=999),
            "string version": mod(base_bpe, format_version="1"),
            "bool version": mod(base_bpe, format_version=True),
            "missing type": mod(base_bpe, type=...),
            "unknown type": mod(base_bpe, type="evil"),
            "missing special": mod(base_bpe, special_tokens=...),
            "wrong special ids": mod(base_bpe, special_tokens={"<PAD>": 1, "<UNK>": 0, "<BOS>": 2, "<EOS>": 3}),
            "missing merges": mod(base_bpe, merges=...),
            "merges not list": mod(base_bpe, merges="abc"),
            "merge wrong arity": mod(base_bpe, merges=[[4, 5, 6]]),
            "merge non-int": mod(base_bpe, merges=[["a", "b"]]),
            "merge bool": mod(base_bpe, merges=[[True, 5]]),
            "merge references future id": mod(base_bpe, merges=[[260, 4]]),
            "merge references special": mod(base_bpe, merges=[[0, 4]]),
            "merge negative": mod(base_bpe, merges=[[-1, 4]]),
            "duplicate merge": mod(base_bpe, merges=[[4, 5], [4, 5]]),
            "word missing vocab": mod(base_word, vocab=...),
            "word duplicate vocab": mod(base_word, vocab=["a", "a"]),
            "word vocab has special": mod(base_word, vocab=["<EOS>"]),
            "word vocab non-string": mod(base_word, vocab=[1, 2]),
            "word vocab empty string": mod(base_word, vocab=[""]),
            "word bad lowercase": mod(base_word, lowercase="yes"),
            "char multi-char entry": mod(base_char, chars=["ab"]),
            "char duplicate": mod(base_char, chars=["a", "a"]),
            "char not list": mod(base_char, chars="abc"),
        }
        for name, content in cases.items():
            with self.subTest(name), self.assertRaises(TokenizerFormatError):
                load_tokenizer(self.write(content, "bad.json"))

    def test_invalid_utf8_file(self):
        p = self.dir / "bin.json"
        p.write_bytes(b"\xff\xfe\x00garbage")
        with self.assertRaises(TokenizerFormatError):
            load_tokenizer(p)

    def test_error_messages_are_specific(self):
        p = self.write({**self.valid("bpe"), "merges": [[260, 4]]})
        with self.assertRaisesRegex(TokenizerFormatError, r"merges\[0\]"):
            load_tokenizer(p)


if __name__ == "__main__":
    unittest.main()
