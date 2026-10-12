"""Give every pytest-xdist worker its own numba cache directory.

numba's on-disk cache (``cache=True``) is not safe to fill from several processes at once: when two processes
compile the same module concurrently, a gufunc's wrapper and its kernel can be cached by different processes with
mismatched internal names, and loading them segfaults (numba/numba#10128, #4807). ``pytest -n auto`` on a fresh
checkout is exactly that situation. This plugin points each xdist worker at its own cache directory, so no two
processes ever write the same cache files.

It only acts inside xdist workers (``PYTEST_XDIST_WORKER`` is set). Serial runs are untouched.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import warnings

ENV = "NUMBA_CACHE_DIR"


def pytest_addoption(parser):
    group = parser.getgroup("numba-cache", "per-worker numba cache directories")
    group.addoption(
        "--numba-cache-shared",
        action="store_true",
        default=False,
        help="let xdist workers share one numba cache directory (disables pytest-numba-cache)",
    )
    parser.addini(
        "numba_cache_isolation",
        type="bool",
        default=True,
        help="give each pytest-xdist worker its own numba cache directory (default: true)",
    )


def _default_base(rootpath) -> str:
    # one persistent base per project, so a worker's cache survives between runs
    digest = hashlib.sha256(str(rootpath).encode()).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), "pytest-numba-cache", digest)


def worker_cache_dir(worker: str, base: str | None, rootpath) -> str:
    return os.path.join(base or _default_base(rootpath), worker)


def pytest_load_initial_conftests(early_config, parser, args):
    """Runs before any conftest.py is imported, which is before most projects import numba."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker:
        return
    if getattr(early_config.known_args_namespace, "numba_cache_shared", False):
        return
    if not early_config.getini("numba_cache_isolation"):
        return
    path = worker_cache_dir(worker, os.environ.get(ENV), early_config.rootpath)
    os.makedirs(path, exist_ok=True)
    os.environ[ENV] = path
    if "numba" in sys.modules:
        # numba was imported before us (e.g. by another plugin): make it re-read the environment
        try:
            from numba.core import config

            config.reload_config()
        except Exception as exc:  # pragma: no cover - depends on numba internals
            warnings.warn(f"pytest-numba-cache: numba was already imported and could not be reconfigured: {exc}")
