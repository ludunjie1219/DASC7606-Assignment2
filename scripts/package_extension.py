#!/usr/bin/env python3
"""Package/check optional extension evidence; never execute submitted code."""
import argparse
import re
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

MAX_ZIP = 100_000_000
MAX_CONTENTS = 200 * 1024 * 1024
BLOCKED = {'.git', '.venv', 'venv', '__pycache__', 'data', 'private', '.env'}


def validate(path):
    path = Path(path)
    if path.stat().st_size > MAX_ZIP:
        raise ValueError('Compressed extension ZIP exceeds 100 MB')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 2000 or sum(i.file_size for i in entries) > MAX_CONTENTS:
            raise ValueError('Too many files or more than 200 MiB unpacked')
        names = set()
        weights = []
        for item in entries:
            name = item.filename
            parts = PurePosixPath(name).parts
            if (not parts or name.startswith('/') or '\\' in name or ':' in name
                    or '..' in parts or any(p.lower() in BLOCKED for p in parts)
                    or stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 1):
                raise ValueError(f'Unsafe or prohibited entry: {name}')
            if name in names:
                raise ValueError(f'Duplicate entry: {name}')
            names.add(name)
            if name.lower().endswith(('.pt', '.pth', '.safetensors', '.ckpt')):
                weights.append(name)
        if weights and weights != ['extension_best.pt']:
            raise ValueError('Only optional root extension_best.pt is allowed')
        for name in ('README.md', 'extension_report.pdf'):
            if name not in names or archive.getinfo(name).file_size == 0:
                raise ValueError(f'Missing or empty {name}')
        with archive.open('extension_report.pdf') as report:
            if report.read(5) != b'%PDF-':
                raise ValueError('extension_report.pdf is not a PDF')
        for folder in ('code/', 'configs/', 'results/'):
            if not any(n.startswith(folder) and not n.endswith('/')
                       and archive.getinfo(n).file_size for n in names):
                raise ValueError(f'Missing nonempty evidence in {folder}')
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
    return len(entries)


def package(directory, student_id, out):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', student_id):
        raise ValueError('Use a plain student ID, not a path')
    directory = Path(directory).resolve()
    out = Path(out).resolve()
    if not directory.is_dir() or out.exists() or out.is_relative_to(directory):
        raise ValueError('Use an existing input directory and a new output outside it')
    with tempfile.TemporaryDirectory(prefix='a2-extension-') as temp:
        candidate = Path(temp) / 'extension.zip'
        with zipfile.ZipFile(candidate, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(directory.rglob('*')):
                if file.is_symlink():
                    raise ValueError(f'Symlinks are not allowed: {file.name}')
                if file.is_file():
                    archive.write(file, file.relative_to(directory).as_posix())
        validate(candidate)
        # Exclusive creation prevents accidentally replacing an existing submission.
        with out.open('xb') as dest, candidate.open('rb') as source:
            while chunk := source.read(1024 * 1024):
                dest.write(chunk)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--student-id')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        if args.check:
            if any((args.directory, args.student_id, args.out)):
                parser.error('--check cannot be combined with packaging arguments')
            count = validate(args.check)
            print(f'Extension format check passed ({count} entries). This is not a grade.')
        else:
            if not all((args.directory, args.student_id, args.out)):
                parser.error('Packaging requires --directory, --student-id and --out')
            print(f'Created {package(args.directory, args.student_id, args.out)}')
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Extension packaging error: {exc}\n')


if __name__ == '__main__':
    main()
