"""Download the real per-item leaderboard correctness matrices.

Source: felipemaiapolo/efficbench (Polo et al., tinyBenchmarks, ICML 2024), MIT.
`lb.pickle` is the Open LLM Leaderboard: 395 models x per-item correctness across
MMLU, ARC, HellaSwag, Winogrande and GSM8K. ~90MB, and GitHub raw is slow, so this
resumes a partial download rather than restarting it.

    python data/fetch_leaderboard.py            # -> data/lb.pickle
    python data/fetch_leaderboard.py helm       # -> data/helm_lite.pickle (2.5MB)
"""
import os
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/felipemaiapolo/efficbench/master/data/"
FILES = {"lb": "lb.pickle", "helm": "helm_lite.pickle"}


def fetch(which="lb"):
    name = FILES[which]
    out = os.path.join(os.path.dirname(__file__), name)
    if os.path.exists(out):
        print(f"{out} already exists ({os.path.getsize(out)/1e6:.1f} MB)")
        return out
    print(f"downloading {name} (this is slow; lb.pickle is ~90MB)...")
    urllib.request.urlretrieve(BASE + name, out)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else "lb")
