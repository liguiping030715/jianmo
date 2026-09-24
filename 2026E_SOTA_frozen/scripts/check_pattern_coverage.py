from __future__ import annotations
from pathlib import Path
import csv

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'workspace'/'knowledge'/'corpus_manifest.csv'
REPORT=ROOT/'workspace'/'knowledge'/'pattern_coverage_report.md'
REQUIRED=[
    'Problem structure',
    'Question dependency',
    'intermediate mathematical',
    'Model-selection rationale',
    'Validation strategy',
    'Transferable patterns',
    'Non-transferable',
]


def main():
    if not MANIFEST.exists():
        raise SystemExit('Manifest missing. Run build_corpus_manifest.py first.')
    rows=list(csv.DictReader(MANIFEST.open(encoding='utf-8-sig')))
    missing=[]; incomplete=[]; ok=[]
    for r in rows:
        p=ROOT/r['pattern_card_path']
        if not p.exists():
            missing.append((r,p)); continue
        text=p.read_text(encoding='utf-8', errors='ignore')
        absent=[h for h in REQUIRED if h.lower() not in text.lower()]
        if absent:
            incomplete.append((r,p,absent))
        else:
            ok.append((r,p))
    status='COMPLETE' if not missing and not incomplete else 'INCOMPLETE'
    lines=['# Pattern Card Coverage Report','',f'- Status: **{status}**',f'- Manifest papers: {len(rows)}',f'- Complete cards: {len(ok)}',f'- Missing cards: {len(missing)}',f'- Incomplete cards: {len(incomplete)}','']
    if missing:
        lines += ['## Missing'] + [f"- {r['year']}{r['question']} — `{r['relative_path']}` -> `{p.relative_to(ROOT)}`" for r,p in missing]
    if incomplete:
        lines += ['', '## Incomplete'] + [f"- `{p.relative_to(ROOT)}` missing headings/terms: {', '.join(absent)}" for r,p,absent in incomplete]
    REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines[:7]))
    if status != 'COMPLETE': raise SystemExit(2)

if __name__=='__main__': main()
