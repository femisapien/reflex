"""Rebuild a Decision Index serving source archive from this checkout and print its hashes.

The GPU jobs ran from a source archive whose layout is

    reflex/build-info.json      (configs/decision-index/<run>-build-info.json)
    reflex/serve_h100.py        (serve_h100.py)
    reflex/src/reflex/**        (src/reflex/**)

built as sorted files, mtime 0, mode 0644, no __pycache__, gzip mtime 0. Running this at a
release tag prints every member's SHA-256 and the archive's, to compare with
docs/decision-index-0.3.md. Member hashes are what pins the code; the archive hash also
depends on the local zlib build, so it can differ on another machine while every member
matches.

    python scripts/decision_index/build_source_archive.py \
        configs/decision-index/pmscr2-build-info.json [--out source.tar.gz]
"""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def members(build_info: Path) -> dict[str, bytes]:
    files = {"reflex/build-info.json": build_info.read_bytes(),
             "reflex/serve_h100.py": (ROOT / "serve_h100.py").read_bytes()}
    for path in (ROOT / "src" / "reflex").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files["reflex/" + path.relative_to(ROOT).as_posix()] = path.read_bytes()
    return dict(sorted(files.items()))


def archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=9) as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    data = bytearray(buffer.getvalue())
    data[4:8] = b"\0\0\0\0"  # gzip header mtime
    return bytes(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_info", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    files = members(args.build_info)
    data = archive(files)
    if args.out:
        args.out.write_bytes(data)
    print(json.dumps({
        "archive_sha256": hashlib.sha256(data).hexdigest(),
        "members": {name: hashlib.sha256(body).hexdigest() for name, body in files.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
