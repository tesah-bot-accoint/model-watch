"""The live server, end to end over real HTTP on a random local port, with a tiny random model. No downloads."""
import json
import time
import unittest
import urllib.error
import urllib.request

from test_model_watch import make_watcher

from model_watch.export import trace_to_html
from model_watch.live import LIVE_PLACEHOLDER, LiveServer


def request(url, data=None, kind="application/json"):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method="GET" if data is None else "POST")
    if data is not None and kind:
        req.add_header("Content-Type", kind)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers, err.read()


class TestLive(unittest.TestCase):
    def setUp(self):
        self.watcher = make_watcher("gemma3", chat=True)
        self.server = LiveServer(self.watcher, max_new_tokens=8, log=lambda *_: None).start(port=0)
        self.api = self.server.url + "api/"

    def tearDown(self):
        self.server.stop()

    def state(self, run=-1, since=0, v=-1):
        status, _, body = request(f"{self.api}state?run={run}&since={since}&v={v}")
        self.assertEqual(status, 200)
        return json.loads(body)

    def wait_until_done(self):
        d = self.state()
        deadline = time.time() + 30
        while (d["busy"] or d["final"] is None) and time.time() < deadline:
            d = self.state(d["run"], 0, d["v"] - 1 if not d["busy"] else d["v"])
        return d

    def test_page_is_live_and_exports_are_not(self):
        status, headers, body = request(self.server.url)
        self.assertEqual(status, 200)
        page = body.decode()
        self.assertIn('const LIVE = {"api": "api/"};', page)
        self.assertNotIn(LIVE_PLACEHOLDER, page)
        trace = self.watcher.trace("hi", 2, stop_at_eos=False)
        self.assertIn("const LIVE = " + LIVE_PLACEHOLDER, trace_to_html(trace))

    def test_before_anything_is_asked(self):
        d = self.state()
        self.assertEqual((d["run"], d["busy"], d["steps"], d["final"]), (0, False, [], None))
        self.assertEqual(request(self.api + "replay.html")[0], 404)

    def test_posts_need_json(self):
        self.assertEqual(request(self.api + "ask", {"message": "hi"}, kind="text/plain")[0], 415)
        self.assertEqual(request(self.api + "ask", {"message": "  "})[0], 400)

    def test_reply_matches_a_direct_trace(self):
        expected = make_watcher("gemma3", chat=True).trace("hello there", 8)
        self.assertEqual(request(self.api + "ask", {"message": "hello there"})[0], 200)
        d = self.wait_until_done()
        final = d["final"]
        self.assertEqual([s["token_id"] for s in final["steps"]], [s["token_id"] for s in expected["steps"]])
        self.assertEqual(final["prompt"], "hello there")
        self.assertEqual(set(final["feature_info"]), set(expected["feature_info"]))
        self.assertEqual(self.server.session.messages[-1], {"role": "assistant", "content": final["reply"]})
        # Steps arrive incrementally: asking from step 2 returns the rest and no header.
        part = self.state(d["run"], 2, -1)
        self.assertIsNone(part["header"])
        self.assertEqual(len(part["steps"]), len(final["steps"]) - 2)
        status, headers, body = request(self.api + "replay.html")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertIn('"prompt": "hello there"', body.decode())
        self.assertEqual(json.loads(request(self.api + "trace.json")[2])["reply"], final["reply"])

    def test_conversation_continues_then_resets(self):
        request(self.api + "ask", {"message": "first"})
        first = self.wait_until_done()
        request(self.api + "ask", {"message": "second"})
        d = self.state(first["run"], 0, first["v"])
        while d["busy"] or d["final"] is None:
            d = self.state(d["run"], 0, d["v"] - 1 if not d["busy"] else d["v"])
        self.assertEqual(d["run"], first["run"] + 1)
        self.assertEqual(len(d["final"]["messages"]), 3)
        self.assertEqual(request(self.api + "reset", {})[0], 200)
        after = self.state()
        self.assertEqual((after["run"], after["header"], after["final"]), (0, None, None))
        request(self.api + "ask", {"message": "fresh"})
        fresh = self.wait_until_done()
        self.assertEqual(len(fresh["final"]["messages"]), 1)
        self.assertGreater(fresh["run"], d["run"])

    def test_busy_and_stop(self):
        self.server.session.delay = 0.05
        self.server.session.max_new_tokens = 200
        self.server.session.watcher.stop_ids = set()
        self.assertEqual(request(self.api + "ask", {"message": "long"})[0], 200)
        self.assertEqual(request(self.api + "ask", {"message": "again"})[0], 409)
        self.assertEqual(request(self.api + "reset", {})[0], 409)
        time.sleep(0.3)
        self.assertEqual(request(self.api + "stop", {})[0], 200)
        d = self.wait_until_done()
        self.assertLess(len(d["final"]["steps"]), 200)
        self.assertTrue(d["status"].startswith("Stopped early"))


if __name__ == "__main__":
    unittest.main()
