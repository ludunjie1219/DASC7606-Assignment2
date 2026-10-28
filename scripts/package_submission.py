"""Allowlist-based submission export, validated before the final ZIP is created."""
import argparse
import json
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import sha256
from scripts.validate_submission import ROLES, EVIDENCE, validate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student-id", required=True)
    for role in ROLES:
        p.add_argument("--" + role.replace("_", "-"), required=True, type=Path)
    p.add_argument("--select", required=True, choices=ROLES)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if not args.student_id.isdigit():
        raise ValueError("student ID must contain digits only")
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    if args.out.exists() or partial.exists():
        raise FileExistsError("choose a new output filename")
    runs = {role: getattr(args, role) for role in ROLES}
    checkpoint = runs[args.select] / "best.pt"
    manifest = {"format_version": 1, "student_id": args.student_id, "selected_run": args.select,
                "checkpoint_sha256": sha256(checkpoint), "starter_version": "assignment2-rc1"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as z:
        def add(source, destination):
            if source.is_symlink():
                raise ValueError(f"symlinks are forbidden: {source}")
            z.write(source, destination)
        z.writestr("submission.json", json.dumps(manifest, indent=2))
        add(checkpoint, "best.pt")
        add(args.report, "report.pdf")
        for folder, suffixes in (("src", {".py"}), ("scripts", {".py", ".sh"}),
                                 ("configs", {".yaml"}), ("tests", {".py"})):
            for file in sorted(Path(folder).rglob("*")):
                if file.is_file() and file.suffix in suffixes:
                    add(file, str(file))
        for name in ("README.md", "requirements.txt", "LICENSE", "pytest.ini"):
            add(Path(name), name)
        for role, directory in runs.items():
            for name in EVIDENCE:
                add(directory / name, f"runs/{role}/{name}")
    result = validate(partial)
    partial.replace(args.out)
    print(json.dumps(result, indent=2))
    print(args.out)

if __name__ == "__main__":
    main()
