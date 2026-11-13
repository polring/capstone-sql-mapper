"""Run a front-model command using the project's Python environment."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


MODULES = {
    "build": "src.data_generation.build_front",
    "train": "training.front.train",
    "export": "training.front.export_gguf",
    "infer": "inference.front",
    "evaluate": "training.front.evaluate",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=MODULES)
    parser.add_argument("options", nargs=argparse.REMAINDER, help="options for the selected command")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    venv_python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = venv_python if venv_python.is_file() else sys.executable
    command = [python, "-m", MODULES[args.command], *args.options]
    return subprocess.run(command, cwd=root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
