import os

import pytest

pytest.importorskip("xdist")

# Each inner test writes what its worker sees, so assertions don't depend on parsing pytest's output.
PROBE = """
import os

def test_probe(tmp_path_factory):
    out = os.path.join({outdir!r}, os.environ.get("PYTEST_XDIST_WORKER", "main") + ".txt")
    with open(out, "w") as f:
        f.write(os.environ.get("NUMBA_CACHE_DIR", "<unset>"))

def test_probe_2():
    pass
"""


def _seen(pytester, *args, env=None):
    outdir = pytester.path / "seen"
    outdir.mkdir(exist_ok=True)
    pytester.makepyfile(test_probe=PROBE.format(outdir=str(outdir)))
    if env is not None:
        for k, v in env.items():
            pytester._monkeypatch.setenv(k, v)
    result = pytester.runpytest_subprocess(*args)
    return result, {p.stem: p.read_text() for p in outdir.glob("*.txt")}


def test_each_worker_gets_its_own_directory(pytester):
    pytester._monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    result, seen = _seen(pytester, "-n", "2")
    result.assert_outcomes(passed=2)
    dirs = [d for d in seen.values() if d != "<unset>"]
    assert dirs and len(set(dirs)) == len(dirs)
    for worker, d in seen.items():
        assert os.path.basename(d) == worker
        assert os.path.isdir(d)


def test_existing_cache_dir_is_used_as_the_base(pytester, tmp_path):
    base = str(tmp_path / "mycache")
    result, seen = _seen(pytester, "-n", "2", env={"NUMBA_CACHE_DIR": base})
    result.assert_outcomes(passed=2)
    for worker, d in seen.items():
        assert d == os.path.join(base, worker)


def test_opt_out_keeps_one_shared_directory(pytester, tmp_path):
    base = str(tmp_path / "shared")
    result, seen = _seen(pytester, "-n", "2", "--numba-cache-shared", env={"NUMBA_CACHE_DIR": base})
    result.assert_outcomes(passed=2)
    assert set(seen.values()) == {base}


def test_ini_opt_out(pytester, tmp_path):
    base = str(tmp_path / "shared")
    pytester.makeini("[pytest]\nnumba_cache_isolation = false\n")
    result, seen = _seen(pytester, "-n", "2", env={"NUMBA_CACHE_DIR": base})
    result.assert_outcomes(passed=2)
    assert set(seen.values()) == {base}


def test_serial_run_is_untouched(pytester):
    pytester._monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    result, seen = _seen(pytester)
    result.assert_outcomes(passed=2)
    assert seen == {"main": "<unset>"}


def test_conftest_that_imports_numba_sees_the_worker_directory(pytester):
    pytest.importorskip("numba")
    pytester._monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    outdir = pytester.path / "seen"
    outdir.mkdir()
    # a conftest importing numba at import time, like many numba-using projects do
    pytester.makeconftest(
        "import os, numba\n"
        "from numba.core import config as nbconfig\n"
        f"_d = {str(outdir)!r}\n"
        "def pytest_configure(config):\n"
        "    w = os.environ.get('PYTEST_XDIST_WORKER', 'main')\n"
        "    open(os.path.join(_d, w + '.txt'), 'w').write(str(nbconfig.CACHE_DIR))\n"
    )
    pytester.makepyfile(test_x="def test_a(): pass\ndef test_b(): pass\n")
    result = pytester.runpytest_subprocess("-n", "2")
    result.assert_outcomes(passed=2)
    seen = {p.stem: p.read_text() for p in outdir.glob("gw*.txt")}
    assert seen and all(os.path.basename(d) == w for w, d in seen.items())
