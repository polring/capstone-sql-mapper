"""Run a back-model command using the project's Python environment."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


MODULES = {
    "generate": "src.data_generation.generalized_pairs.generate",
    "build": "src.data_generation.build_back",
    "train": "training.back.train",
    "export": "training.back.export_gguf",
    "infer": "inference.back",
    "evaluate": "training.back.evaluate",
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
