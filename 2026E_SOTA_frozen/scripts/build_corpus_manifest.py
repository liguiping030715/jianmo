from __future__ import annotations
from pathlib import Path
import csv
import hashlib
import re

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'knowledge_sources' / 'jianmo' / 'Markdown'
OUTDIR = ROOT / 'workspace' / 'knowledge'
MANIFEST = OUTDIR / 'corpus_manifest.csv'
COVERAGE = OUTDIR / 'corpus_coverage.md'


def detect_year(path: Path) -> str:
    m = re.search(r'(20\d{2})', str(path))
    return m.group(1) if m else 'UNKNOWN'


def detect_question(path: Path) -> str:
    s = str(path)
    for q in 'ABCDEF':
        if re.search(rf'[/\\]{q}(?:题优秀论文)?[/\\]', s):
            return q
    stem = path.stem
    m = re.match(r'([A-F])', stem, re.I)
    return m.group(1).upper() if m else 'UNKNOWN'


def card_rel(year: str, q: str, path: Path) -> str:
    safe = re.sub(r'[^0-9A-Za-z\u4e00-\u9fff._-]+', '_', path.stem).strip('_')
    return f'workspace/knowledge/pattern_cards/{year}/{q}/{safe}.md'


def main():
    if not CORPUS.exists():
        raise SystemExit(f'Corpus missing: {CORPUS}. Run scripts/sync_jianmo.py first.')
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows=[]
    for p in sorted(CORPUS.rglob('*.md')):
        rel=p.relative_to(CORPUS)
        raw=p.read_bytes()
        year=detect_year(rel)
        q=detect_question(rel)
        rows.append({
            'paper_id': hashlib.sha1(str(rel).encode('utf-8')).hexdigest()[:12],
            'year': year,
            'question': q,
            'filename': p.name,
            'relative_path': str(rel).replace('\\','/'),
            'size_bytes': len(raw),
            'sha1': hashlib.sha1(raw).hexdigest(),
            'pattern_card_path': card_rel(year,q,p),
        })
    with MANIFEST.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        if rows:
            w.writeheader(); w.writerows(rows)
    by={}
    for r in rows:
        by.setdefault((r['year'],r['question']),0)
        by[(r['year'],r['question'])]+=1
    lines=['# Historical Corpus Coverage','',f'- Total Markdown papers: **{len(rows)}**','', '| Year | Question | Papers |','|---:|:---:|---:|']
    for (y,q),n in sorted(by.items()): lines.append(f'| {y} | {q} | {n} |')
    if len(rows) != 45:
        lines += ['', f'> Note: expected historical corpus count was 45 based on the current 2024-2025 repository snapshot; found {len(rows)}. Re-check repository updates rather than forcing the count.']
    COVERAGE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Wrote {MANIFEST} with {len(rows)} papers')
    print(f'Wrote {COVERAGE}')

if __name__=='__main__': main()
