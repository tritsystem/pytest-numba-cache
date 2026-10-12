# pytest-numba-cache

**Stop `pytest -n auto` from segfaulting in projects that use numba's `cache=True`.**

This pytest plugin gives each [pytest-xdist](https://github.com/pytest-dev/pytest-xdist) worker its own numba cache directory, so parallel workers never write the same cache files.

```bash
pip install pytest-numba-cache
```

That's it. Nothing to configure: it activates only inside xdist workers and leaves serial runs untouched.

## The problem

numba's on-disk cache (`@numba.jit(cache=True)`, `@numba.guvectorize(..., cache=True)`) is not safe to fill from several processes at once. When two processes compile the same module at the same time, a gufunc's wrapper and its kernel can be cached by *different* processes with mismatched internal names, and loading them crashes the process. This is numba's own open bug: [numba/numba#10128](https://github.com/numba/numba/issues/10128), also [#4807](https://github.com/numba/numba/issues/4807).

`pytest -n auto` on a fresh checkout or a fresh install is exactly that situation: every worker imports the library and compiles its cached functions at the same moment. It shows up as workers dying with a segfault or access violation, e.g. librosa's test suite crashing under openSUSE's packaging build ([librosa/librosa#1879](https://github.com/librosa/librosa/issues/1879)).

## Evidence

`tests/race_demo` is a numba-only test suite (12 cached functions in one module, 48 tests), run with `pytest -n 8` from an **empty cache every round**, 10 rounds per row (numba 0.68.0, Python 3.12, Windows 11):

| | rounds with a crash | worker crashes |
|---|---|---|
| without the plugin | 8 / 10 | 138 |
| with the plugin | 0 / 10 | 0 |

With the plugin, every worker wrote its own cache (34–36 cache files per worker per round), so caching was active, just never shared. Reproduce it with `python tests/stress_run.py --plugin off` and `--plugin on`.

The same race in librosa 1.0.0 (8 processes calling `librosa.beat.beat_track` on a fresh cache, 160 processes per row): shared cache, 76 crashed; separate cache per process, 0 crashed.

## How it works

Before any `conftest.py` is imported (pytest's `pytest_load_initial_conftests` hook), each xdist worker sets `NUMBA_CACHE_DIR` to its own subdirectory:

- If you already set `NUMBA_CACHE_DIR`, it's used as the base: `$NUMBA_CACHE_DIR/gw0`, `$NUMBA_CACHE_DIR/gw1`, ...
- Otherwise the base is a per-project folder in your temp directory, so a worker's cache survives between runs and later runs start warm.
- If numba was already imported before the plugin ran, numba's config is reloaded so the new directory takes effect.

**Trade-off:** each worker compiles its own copy the first time, so a cold first run does a little more compiling. In the benchmark above the plugin run was still faster overall (54 s vs 69 s), because crashed workers no longer have to be restarted.

## Options

- `pytest --numba-cache-shared`: turn the plugin off for one run.
- `numba_cache_isolation = false` in your pytest ini: turn it off for a project.

## Limitations

- It fixes the race *between pytest workers*. It does not fix numba itself, or other multi-process programs that share one cache (e.g. several `DataLoader` workers importing a numba library for the first time). For those, warm the cache with one process first, or set a separate `NUMBA_CACHE_DIR` per process.
- The benchmark numbers above are from Windows. The underlying numba bug was originally reported on Linux.

## License

MIT
