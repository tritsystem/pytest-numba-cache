"""Run the race demo under `pytest -n N` from a fresh numba cache, several rounds; count crashed workers.

    python tests/stress_run.py --rounds 10 --workers 8 [--plugin on|off]
"""
import argparse, glob, os, re, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); DEMO = os.path.join(HERE, "race_demo")
ap = argparse.ArgumentParser(); ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--plugin", choices=["on", "off"], default="off"); a = ap.parse_args()
BELOW_NORMAL = 0x00004000 if os.name == "nt" else 0
def clean():
    for f in glob.glob(os.path.join(DEMO, "**", "*.nb[ic]"), recursive=True): os.remove(f)
crashed_rounds = crashed_workers = 0; t0 = time.time()
for r in range(a.rounds):
    clean()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}; env.pop("NUMBA_CACHE_DIR", None)
    if a.plugin == "on":  # fresh, empty base every round, so every round starts cold (the plugin nests one dir per worker)
        env["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="pnc_round_")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-n", str(a.workers), DEMO]
    if a.plugin == "off": cmd[3:3] = ["-p", "no:numba_cache"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, creationflags=BELOW_NORMAL, cwd=DEMO).stdout
    n = len(re.findall(r"worker '?gw\d+'? crashed while running", out))   # crash events (xdist restarts crashed workers)
    summary = (re.findall(r"(\d+ (?:failed|passed)[^\n]*?) in [0-9.]+s", out) or ["?"])[-1]
    crashed_rounds += n > 0; crashed_workers += n
    print(f"round {r+1:2d}: worker crashes {n}  | {summary}", flush=True)
print(f"plugin {a.plugin}: rounds with a crash {crashed_rounds}/{a.rounds}, worker crashes {crashed_workers} total, {time.time()-t0:.0f}s")
