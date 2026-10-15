"""The preview page in docs/ must stay a copy of the packaged viewer. Offline, no torch needed."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPreview(unittest.TestCase):
    def test_preview_matches_viewer(self):
        viewer = (ROOT / "model_watch" / "viewer.html").read_text(encoding="utf-8")
        preview = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
        self.assertEqual(preview, viewer, "docs/index.html is out of date: cp model_watch/viewer.html docs/index.html")

    def test_preview_shows_sample(self):
        preview = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
        self.assertIn("const EMBEDDED = /*__TRACE_JSON__*/null;", preview)


if __name__ == "__main__":
    unittest.main()
