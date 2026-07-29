from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: rotate_stream.py PREFIX MAX_BYTES")

    prefix = os.path.abspath(sys.argv[1])
    max_bytes = max(1, int(sys.argv[2]))
    part = 0
    size = 0
    handle = open(f"{prefix}.part{part:03d}.log", "ab", buffering=0)
    try:
        while True:
            chunk = sys.stdin.buffer.readline()
            if not chunk:
                break
            if size and size + len(chunk) > max_bytes:
                handle.close()
                part += 1
                size = 0
                handle = open(f"{prefix}.part{part:03d}.log", "ab", buffering=0)
            handle.write(chunk)
            size += len(chunk)
    finally:
        handle.close()


if __name__ == "__main__":
    main()
