"""
Builds revision_code.zip: the files that paper.ipynb unpacks into srccv/ on Google Drive.

    python experiments/pack_for_colab.py          # -> <repo>/revision_code.zip
"""

import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The whole training code, not only the files the revision touched: a Drive copy of srccv can hold an older
# configs/default.yaml, data/ or loss/ (seen: embed_dim not divisible by vit_heads, motion loss stuck at ln 16).
FILES = ["train.py", "configs/default.yaml"]
SOURCE_DIRS = ["data", "loss", "models", "utils"]


def main():
    out = REPO / "revision_code.zip"
    members = [REPO / f for f in FILES]
    members += sorted(p for d in SOURCE_DIRS for p in (REPO / d).glob("*.py"))
    members += sorted(p for p in (REPO / "experiments").rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".md", ".docx"})
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in members:
            z.write(p, p.relative_to(REPO).as_posix())
    print(f"[+] {out} ({out.stat().st_size / 1024:.0f} KB, {len(members)} files)")
    for p in members:
        print("   ", p.relative_to(REPO).as_posix())


if __name__ == "__main__":
    main()
