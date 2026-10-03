import copy
import unittest
from pathlib import Path

from tools.concurrent_config import prepare_config
from tools.concurrency_sweep import summarize_wave


class ConcurrencyTools(unittest.TestCase):
    def test_config_preserves_source_and_uses_one_batch_option(self):
        cfg = {"args": ["--native", "model.gguf", "--batch", "8", "--expert-cache", "1234", "--vision"],
               "exe": "old.exe", "tokenizer": "tokens", "vision_exe": "vision.exe", "gpu": [0]}
        original = copy.deepcopy(cfg)
        source = Path("old/model.json").resolve()
        result = prepare_config(cfg, source, 4, 32768, Path("candidate.exe"))
        self.assertEqual(cfg, original)
        self.assertEqual(result["args"].count("--batch"), 1)
        self.assertNotIn("--vision", result["args"])
        self.assertNotIn("vision_exe", result)
        self.assertEqual(result["cwd"], str(source.parent))
        self.assertEqual(result["args"][result["args"].index("--expert-cache") + 1], "auto")

    def test_aggregate_uses_whole_wave_wall_clock(self):
        result = summarize_wave([{"completion_tokens": 80, "ttft_s": 2, "error": None},
                                 {"completion_tokens": 120, "ttft_s": 4, "error": None}], 10)
        self.assertEqual(result["aggregate_tok_s"], 20)
        self.assertEqual(result["ttft_median_s"], 3)

    def test_failed_request_invalidates_wave_rate(self):
        result = summarize_wave([{"completion_tokens": 100, "error": None}, {"error": "EOF"}], 5)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["aggregate_tok_s"])

    def test_sse_wave_counts_actual_usage(self):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        from tools.concurrency_sweep import wave
        root = Path(__file__).resolve().parents[1]
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "abcdefgh"), tok, ChatTemplate(root / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            result = wave(f"http://127.0.0.1:{httpd.server_address[1]}", "", "test", 2, 8, "hi", 3)
            self.assertTrue(result["valid"], result)
            self.assertEqual(result["completion_tokens"], 16)
            self.assertAlmostEqual(result["aggregate_tok_s"], 16 / result["wall_s"])
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
