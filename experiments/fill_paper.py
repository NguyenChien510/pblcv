"""
Fills the revised paper template with the aggregated results.

The template (PAPER_SPACETIME_DSCA_v3_template.docx) marks every number that depends on the new experiments with a
Word content control whose tag is a token name (yellow highlight). This script replaces each control by its value
from results/tokens.json, swaps figures 1, 3 and 5 for results/figures/*.png and, once every token has a value,
removes the draft note at the top. Tokens without a value stay highlighted and are listed, so nothing is filled
silently.

    python experiments/fill_paper.py        # experiments/paper/..._template.docx -> results/PAPER_SPACETIME_DSCA_v3.docx

Standard library only. After filling, re-read the discussion paragraphs: the comparison sentences are generated from
the confidence intervals, but the surrounding interpretation is still the authors' responsibility.
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parent.parent
FIGURES = {  # media file in the template -> generated figure
    "word/media/revfig1.png": "fig1_test_r1.png",
    "word/media/revfig3.png": "fig3_delta_mfgf.png",
    "word/media/revfig5.png": "fig5_ablation.png",
}
SDT = re.compile(r'<w:sdt><w:sdtPr><w:alias w:val="([A-Za-z0-9_]+)"/>.*?</w:sdtPr><w:sdtContent>(.*?)</w:sdtContent></w:sdt>',
                 re.S)
NOTE = "DRAFT_NOTE"


def filled_run(content: str, value: str) -> str:
    run = re.search(r"<w:r>.*?</w:r>", content, re.S).group(0)
    run = run.replace('<w:highlight w:val="yellow"/>', "")
    return re.sub(r"<w:t[^>]*>.*?</w:t>", f'<w:t xml:space="preserve">{escape(value)}</w:t>', run, count=1, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=str(REPO / "experiments" / "paper" / "PAPER_SPACETIME_DSCA_v3_template.docx"))
    ap.add_argument("--tokens", default=str(REPO / "results" / "tokens.json"))
    ap.add_argument("--figures", default=str(REPO / "results" / "figures"))
    ap.add_argument("--out", default=str(REPO / "results" / "PAPER_SPACETIME_DSCA_v3.docx"))
    args = ap.parse_args()

    tokens = json.loads(Path(args.tokens).read_text(encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(args.template) as z:
        names = z.namelist()
        z.extractall(tmp)

    doc = tmp / "word" / "document.xml"
    xml = doc.read_text(encoding="utf-8")
    used = {m.group(1) for m in SDT.finditer(xml)} - {NOTE}
    missing = sorted(t for t in used if t not in tokens)

    def repl(m):
        name, content = m.group(1), m.group(2)
        if name == NOTE:
            return m.group(0) if missing else ""
        if name not in tokens:
            return m.group(0)
        return filled_run(content, str(tokens[name]))

    doc.write_text(SDT.sub(repl, xml), encoding="utf-8")

    swapped = []
    for media, fig in FIGURES.items():
        src = Path(args.figures) / fig
        if src.exists() and (tmp / media).exists():
            shutil.copyfile(src, tmp / media)
            swapped.append(fig)

    out = Path(args.out)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:                      # keep the original part order ([Content_Types].xml first)
            z.write(tmp / n, n)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[+] wrote {out}: {len(used) - len(missing)}/{len(used)} tokens filled, figures replaced: {swapped or 'none'}")
    if missing:
        print(f"[!] {len(missing)} tokens have no value yet (still highlighted, draft note kept): {missing}")
        sys.exit(2)
    print("[+] all tokens filled; the draft note was removed")


if __name__ == "__main__":
    main()
