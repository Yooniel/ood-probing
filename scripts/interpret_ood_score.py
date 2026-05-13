#!/usr/bin/env python3
from __future__ import annotations

from _path import add_src_to_path

add_src_to_path()

from interpretation import main


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")
