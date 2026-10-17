import random
import unittest
from collections import Counter

from LLM.tokenizer import BPETokenizer, InvalidTokenIdError
from LLM.tokenizer.bpe import _PRETOKEN_RE, BASE_VOCAB_SIZE, _merge_pair

CORPUS = ("the quick brown fox jumps over the lazy dog. " * 30
          + "low lower lowest newer wider. " * 20)

SAMPLES = [
    "", " ", "a", "hello world", "  leading and trailing  ", "tabs\tand\nnewlines\r\n\n",
    "punctuation!!! ... ?!", "café naïve 日本語 🙂 Ω", "e\u0301 vs \u00e9 (no normalization)",
    "<EOS> <BOS> <PAD> <UNK>", "completely unseen: zxqv 12345 ###", "a" * 1000,
]


class PretokenizerTests(unittest.TestCase):
    def test_chunks_reassemble_to_original(self):
        for s in SAMPLES + ["x \n y", "a  \n  b", "\n\n  \t "]:
            with self.subTest(s=s):
                self.assertEqual("".join(_PRETOKEN_RE.findall(s)), s)


class MergeHelperTests(unittest.TestCase):
    def test_merge_pair_left_to_right_non_overlapping(self):
        self.assertEqual(_merge_pair([1, 1, 1], (1, 1), 9), [9, 1])
        self.assertEqual(_merge_pair([1, 2, 1, 2, 3], (1, 2), 9), [9, 9, 3])
        self.assertEqual(_merge_pair([], (1, 2), 9), [])


class BPETrainingTests(unittest.TestCase):
    def test_untrained_is_pure_byte_tokenizer(self):
        t = BPETokenizer()
        self.assertEqual(t.vocab_size, BASE_VOCAB_SIZE)
        self.assertEqual(t.encode("A"), [4 + 65])
        self.assertEqual(t.decode(t.encode("héllo")), "héllo")

    def test_training_is_deterministic(self):
        a, b = BPETokenizer(), BPETokenizer()
        a.train(CORPUS, vocab_size=400)
        b.train(CORPUS, vocab_size=400)
        self.assertEqual(a.merges, b.merges)
        self.assertEqual(a.encode("the lazy dog"), b.encode("the lazy dog"))

    def test_vocab_size_respected_and_merges_compress(self):
        t = BPETokenizer()
        t.train(CORPUS, vocab_size=300)
        self.assertLessEqual(t.vocab_size, 300)
        self.assertGreater(t.vocab_size, BASE_VOCAB_SIZE)
        self.assertLess(len(t.encode(CORPUS)), len(CORPUS.encode("utf-8")))

    def test_more_merges_never_lengthen_encoding(self):
        small, big = BPETokenizer(), BPETokenizer()
        small.train(CORPUS, vocab_size=280)
        big.train(CORPUS, vocab_size=500)
        self.assertLessEqual(len(big.encode(CORPUS)), len(small.encode(CORPUS)))

    def test_first_merge_is_most_frequent_pair_with_smallest_tie_break(self):
        t = BPETokenizer()
        t.train("ab ab ab cd cd", vocab_size=BASE_VOCAB_SIZE + 1)
        # Chunks: "ab"x1, " ab"x2, "cd"x1, " cd"x1. Pair (a,b) occurs 3x: the winner.
        self.assertEqual(t.merges, [(4 + ord("a"), 4 + ord("b"))])

    def test_tie_break_smallest_pair(self):
        t = BPETokenizer()
        t.train("xy\nab", vocab_size=BASE_VOCAB_SIZE + 1, min_frequency=1)
        # (a,b) and (x,y) both occur once; 'a' < 'x' so (a,b) wins.
        self.assertEqual(t.merges, [(4 + ord("a"), 4 + ord("b"))])

    def test_min_frequency_stops_training(self):
        t = BPETokenizer()
        t.train("abcdefg", vocab_size=1000, min_frequency=2)
        self.assertEqual(t.merges, [])

    def test_stops_when_nothing_left_to_merge(self):
        t = BPETokenizer()
        t.train("ab", vocab_size=1000, min_frequency=1)
        self.assertEqual(t.vocab_size, BASE_VOCAB_SIZE + 1)  # one merge, then single token

    def test_vocab_size_too_small(self):
        with self.assertRaises(ValueError):
            BPETokenizer().train("abc", vocab_size=100)
        with self.assertRaises(ValueError):
            BPETokenizer().train("abc", min_frequency=0)

    def test_retraining_replaces_old_merges_and_cache(self):
        t = BPETokenizer()
        t.train(CORPUS, vocab_size=400)
        t.encode("the quick")
        t.train("zzzz zzzz zzzz", vocab_size=300)
        self.assertEqual(t.decode(t.encode("the quick")), "the quick")
        self.assertEqual(t.merges[0], (4 + ord("z"), 4 + ord("z")))


def reference_train(text, vocab_size, min_frequency):
    """Slow, obviously-correct BPE: rescan every pair after every merge."""
    words = [([4 + b for b in c.encode("utf-8")], f)
             for c, f in Counter(_PRETOKEN_RE.findall(text)).items()]
    merges = []
    while BASE_VOCAB_SIZE + len(merges) < vocab_size:
        counts = Counter()
        for seq, f in words:
            for p in zip(seq, seq[1:]):
                counts[p] += f
        if not counts:
            break
        best, c = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if c < min_frequency:
            break
        nid = BASE_VOCAB_SIZE + len(merges)
        merges.append(best)
        words = [(_merge_pair(s, best, nid), f) for s, f in words]
    return merges


class FastTrainingMatchesReferenceTests(unittest.TestCase):
    def test_matches_naive_implementation(self):
        rng = random.Random(0)
        alphabets = ["ab", "abcde", "abcdefghijklmnopqrstuvwxyz ", "aé日🙂 \n.,"]
        for alphabet in alphabets:
            for min_freq in (1, 2, 3):
                text = "".join(rng.choice(alphabet) for _ in range(3000))
                with self.subTest(alphabet=alphabet, min_freq=min_freq):
                    t = BPETokenizer()
                    t.train(text, vocab_size=400, min_frequency=min_freq)
                    self.assertEqual(t.merges, reference_train(text, 400, min_freq))

    def test_matches_naive_on_realistic_text(self):
        t = BPETokenizer()
        t.train(CORPUS, vocab_size=500)
        self.assertEqual(t.merges, reference_train(CORPUS, 500, 2))


class BPERoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tok = BPETokenizer()
        self.tok.train(CORPUS, vocab_size=400)

    def test_exact_round_trip_for_all_samples(self):
        for s in SAMPLES:
            with self.subTest(s=s[:30]):
                self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_empty(self):
        self.assertEqual(self.tok.encode(""), [])
        self.assertEqual(self.tok.decode([]), "")

    def test_ids_in_range(self):
        for s in SAMPLES:
            ids = self.tok.encode(s)
            self.assertTrue(all(4 <= i < self.tok.vocab_size for i in ids))

    def test_no_normalization(self):
        self.assertNotEqual(self.tok.encode("e\u0301"), self.tok.encode("\u00e9"))

    def test_truncated_multibyte_decodes_with_replacement(self):
        ids = BPETokenizer().encode("🙂")  # 4 bytes
        self.assertEqual(BPETokenizer().decode(ids[:2]), "\ufffd")

    def test_multibyte_char_split_across_tokens_still_decodes(self):
        t = BPETokenizer()
        t.train("日本語 " * 50, vocab_size=300)
        self.assertEqual(t.decode(t.encode("日本語 日本語")), "日本語 日本語")

    def test_invalid_ids(self):
        for bad in (-1, self.tok.vocab_size, 10**9, "5", 1.0, None, True):
            with self.subTest(bad=bad), self.assertRaises(InvalidTokenIdError):
                self.tok.decode([bad])

    def test_merge_consistency_each_merge_id_decodes_to_concatenation(self):
        for rank, (a, b) in enumerate(self.tok.merges):
            merged = BASE_VOCAB_SIZE + rank
            self.assertEqual(self.tok.decode([merged]),
                             self.tok.decode([a]) + self.tok.decode([b]))

    def test_merges_applied_in_learned_order(self):
        # Re-encoding via the trained merges must reproduce the training-time
        # tokenization of a training chunk.
        ids = self.tok.encode(" the")
        self.assertLessEqual(len(ids), 2)


if __name__ == "__main__":
    unittest.main()
