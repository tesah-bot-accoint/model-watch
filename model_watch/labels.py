"""Human-readable feature labels from Neuronpedia, cached on disk.

Labels are best-effort. If Neuronpedia is unreachable or its response shape
changes, features show without a label and the viewer still links to the
feature's Neuronpedia page.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

API = "https://www.neuronpedia.org/api/feature/{model}/{source}/{index}"
PAGE = "https://www.neuronpedia.org/{model}/{source}/{index}"


def _cache_dir() -> Path:
    base = os.environ.get("MODEL_WATCH_CACHE") or Path.home() / ".cache" / "model-watch"
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_label(data) -> str | None:
    """Pull the first explanation text out of a Neuronpedia feature response."""
    if not isinstance(data, dict):
        return None
    for key in ("explanations", "explanation"):
        value = data.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    text = item.get("description") or item.get("text")
                    if text:
                        return str(text).strip()
                elif isinstance(item, str) and item.strip():
                    return item.strip()
        elif isinstance(value, str) and value.strip():
            return value.strip()
    return None


class NeuronpediaLabels:
    def __init__(
        self,
        model_id: str = "gemma-2-2b",
        source_id: str = "20-gemmascope-res-16k",
        enabled: bool = True,
        timeout: float = 6.0,
        api_key: str | None = None,
        max_failures: int = 3,
        log=print,
    ):
        self.model_id = model_id
        self.source_id = source_id
        self.enabled = enabled
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("NEURONPEDIA_API_KEY")
        self.max_failures = max_failures
        self.log = log
        self._failures = 0
        self._lock = threading.Lock()
        self._path = _cache_dir() / f"labels-{model_id}-{source_id}.json"
        try:
            self._cache: dict[str, str | None] = json.loads(self._path.read_text())
        except (OSError, ValueError):
            self._cache = {}

    def url(self, index: int) -> str:
        return PAGE.format(model=self.model_id, source=self.source_id, index=index)

    def cached(self, index: int) -> str | None:
        """Label from the disk cache only. Never touches the network."""
        return self._cache.get(str(index))

    def is_cached(self, index: int) -> bool:
        return str(index) in self._cache

    def get(self, index: int, save: bool = True) -> str | None:
        key = str(index)
        if key in self._cache:
            return self._cache[key]
        if not self.enabled:
            return None
        ok, label = self._fetch(index)
        if ok:  # cache real answers, including "no label"; never cache network failures
            with self._lock:
                self._cache[key] = label
            if save:
                self.save()
        return label

    def save(self) -> None:
        self._save()

    def _fetch(self, index: int) -> tuple[bool, str | None]:
        url = API.format(model=self.model_id, source=self.source_id, index=index)
        headers = {"Accept": "application/json", "User-Agent": "model-watch"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self._failures = 0
            return True, parse_label(data)
        except urllib.error.HTTPError as err:
            if err.code == 404:  # feature exists but has no page data: a real "no label"
                return True, None
            return self._failed(err)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as err:
            return self._failed(err)

    def _failed(self, err) -> tuple[bool, None]:
        with self._lock:
            self._failures += 1
            if self._failures >= self.max_failures and self.enabled:
                self.enabled = False
                self.log(f"[labels] Neuronpedia unreachable ({err}); continuing without its labels.")
        return False, None

    def _save(self) -> None:
        with self._lock:
            data = json.dumps(self._cache)
        try:
            self._path.write_text(data)
        except OSError:
            pass
