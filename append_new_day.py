"""Compatibility entry point; delegates to the single validated pipeline (schema v2)."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[0]
if __name__ == '__main__':
    raise SystemExit(subprocess.call([sys.executable, str(ROOT/'omie_pipeline.py'), 'build', '--root', str(ROOT), *sys.argv[1:]]))
