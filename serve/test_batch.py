"""Batch serving contract tests against a real pipe-driven subprocess, without a GPU."""
import concurrent.futures
import json
import subprocess
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, serve

ROOT = Path(__file__).resolve().parents[1]


class BatchServing(unittest.TestCase):
    def setUp(self):
        real_popen = subprocess.Popen

        def launch(argv, **kwargs):
            # StrataEngine normally inserts --serve immediately after the executable.
            return real_popen([argv[0], argv[2], argv[1], *argv[3:]], **kwargs)

        with mock.patch("serve.server.subprocess.Popen", side_effect=launch):
            self.engine = StrataEngine(sys.executable, [str(ROOT / "serve/fake_batch_engine.py"),
                                                       "--batch", "2", "--max-context", "4096"])
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def tearDown(self):
        self.engine.close()

    def consume(self, token, count=30, started=None):
        got = []
        for value in self.engine.generate([token], count, {}, threading.Event()):
            if value is not None:
                got.append(value)
                if started:
                    started.set()
        return got, dict(self.engine.last)

    def wait_for(self, predicate):
        end = time.monotonic() + 3
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(0.005)
        self.fail("timed out waiting for protocol state")

    def test_solo_promotes_without_duplicating_or_crossing_tokens(self):
        started = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            one = pool.submit(self.consume, 65, 30, started)
            self.assertTrue(started.wait(3))
            two = pool.submit(self.consume, 66, 18)
            self.wait_for(lambda: all(self.engine.slot_busy))
            a, stats_a = one.result(timeout=5)
            b, stats_b = two.result(timeout=5)
        self.assertEqual(a, [65] * 30)
        self.assertEqual(b, [66] * 18)
        self.assertEqual(stats_a["generated"], 30)
        self.assertEqual(stats_b["generated"], 18)
        self.assertFalse(any(self.engine.slot_busy))

    def test_queued_cancel_does_not_send_a_request(self):
        self.engine.ctl.acquire()
        cancel = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(lambda: list(self.engine.generate([67], 5, {}, cancel)))
            self.wait_for(lambda: self.engine.waiting == 1)
            cancel.set()
            self.assertEqual(future.result(timeout=1), [])
        self.engine.ctl.release()
        self.assertEqual(self.consume(68, 4)[0], [68] * 4)

    def test_abandoned_stream_releases_only_its_slot(self):
        started = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            background = pool.submit(self.consume, 65, 30, started)
            self.assertTrue(started.wait(3))
            gen = self.engine.generate([66], 100, {}, threading.Event())
            self.assertEqual(next(gen), 66)
            gen.close()  # also covers leaving during admission, before BADM was read
            self.assertEqual(self.consume(67, 5)[0], [67] * 5)
            self.assertEqual(background.result(timeout=5)[0], [65] * 30)
        self.wait_for(lambda: not any(self.engine.slot_busy))

    def test_history_and_totals_record_every_concurrent_request(self):
        start = threading.Barrier(2)

        def run(token, count):
            start.wait(timeout=2)
            return list(self.svc.run([token], False, [], count, {}, threading.Event()))

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            one = pool.submit(run, 65, 25)
            two = pool.submit(run, 66, 15)
            self.wait_for(lambda: all(self.engine.slot_busy))
            with self.svc.status_lock:
                self.assertEqual(self.svc.status["active"], 2)
            self.assertEqual(self.svc.unload(), "busy")
            one.result(timeout=5)
            two.result(timeout=5)
        self.assertEqual(sorted(h["engine_generated"] for h in self.svc.history), [15, 25])
        self.assertEqual(self.svc.totals["requests"], 2)
        self.assertEqual(self.svc.totals["output_tokens"], 40)
        self.assertFalse(self.svc.status["busy"])
        self.assertFalse(self.svc.active_runs)

    def test_slot_discovery_reports_configured_capacity(self):
        httpd = serve(self.svc, port=0)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/slots", timeout=3) as r:
                slots = json.load(r)
            self.assertEqual([s["id"] for s in slots], [0, 1])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_unsupported_penalty_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not supported"):
            list(self.engine.generate([65], 3, {"repetition_penalty": 1.2}, threading.Event()))

    def test_more_requests_than_slots_queue_and_keep_their_own_tokens(self):
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda t: self.consume(t, 12)[0], [65, 66, 67, 68]))
        self.assertEqual(results, [[t] * 12 for t in [65, 66, 67, 68]])
        self.assertFalse(any(self.engine.slot_busy))


if __name__ == "__main__":
    unittest.main()
