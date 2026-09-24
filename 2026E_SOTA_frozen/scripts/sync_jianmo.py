from __future__ import annotations
from pathlib import Path
import argparse
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'knowledge_sources' / 'jianmo'
URL = 'https://github.com/liguiping030715/jianmo.git'
LOCAL_LIBRARY = ROOT / 'workspace' / 'knowledge'


def run(cmd):
    print('>', ' '.join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def copy_tree_merge(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob('*'):
        rel = p.relative_to(src)
        q = dst / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)


def main():
    ap = argparse.ArgumentParser(description='Sync jianmo historical corpus and, by default, its derived Pattern Library.')
    ap.add_argument('--no-library', action='store_true', help='Do not merge repo workspace/knowledge into local workspace/knowledge.')
    args = ap.parse_args()

    DEST.parent.mkdir(parents=True, exist_ok=True)
    if (DEST / '.git').exists():
        run(['git', '-C', str(DEST), 'pull', '--ff-only'])
    elif DEST.exists() and any(DEST.iterdir()):
        raise SystemExit(f'{DEST} exists but is not a git repository. Move/remove it first.')
    else:
        run(['git', 'clone', '--depth', '1', URL, str(DEST)])

    md = DEST / 'Markdown'
    if not md.exists():
        raise SystemExit(f'Markdown corpus not found: {md}')
    md_count = sum(1 for _ in md.rglob('*.md'))
    print(f'Markdown papers found: {md_count}')
    if md_count < 1:
        raise SystemExit('No Markdown papers found.')

    source_library = DEST / 'workspace' / 'knowledge'
    if not args.no_library:
        if source_library.exists():
            copy_tree_merge(source_library, LOCAL_LIBRARY)
            cards = sum(1 for _ in (LOCAL_LIBRARY / 'pattern_cards').rglob('*.md')) if (LOCAL_LIBRARY / 'pattern_cards').exists() else 0
            synth = sum(1 for _ in (LOCAL_LIBRARY / 'same_problem_synthesis').rglob('*.md')) if (LOCAL_LIBRARY / 'same_problem_synthesis').exists() else 0
            reusable = sum(1 for _ in (LOCAL_LIBRARY / 'reusable_patterns').rglob('*.md')) if (LOCAL_LIBRARY / 'reusable_patterns').exists() else 0
            print(f'Pattern Library merged into: {LOCAL_LIBRARY}')
            print(f'  pattern cards: {cards}')
            print(f'  same-problem synthesis: {synth}')
            print(f'  reusable patterns: {reusable}')
        else:
            print(f'WARNING: repository Pattern Library not found: {source_library}')
            print('Historical Markdown corpus was synced, but local derived knowledge was not updated.')


if __name__ == '__main__':
    main()
