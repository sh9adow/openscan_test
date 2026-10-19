"""Fetch the reviewed source and make a separate working copy. No install/run."""
import argparse
import shutil
import subprocess
from pathlib import Path

COMMIT = '28a14308a36f6b0d09951b1b2ed3bf4e5a66af95'
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', default='runs/openscan', help='New working copy inside this project')
    args = parser.parse_args()
    source = ROOT / 'code' / 'OpenScan'
    work = (ROOT / args.work).resolve()
    if ROOT not in work.parents or work == source or source in work.parents:
        parser.error('--work must be a new project subdirectory outside code/OpenScan')
    if work.exists():
        parser.error(f'Will not overwrite existing path: {work}')
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/YoujunZhao/OpenScan.git', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', COMMIT], check=True)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    if actual != COMMIT or dirty:
        raise SystemExit('Official checkout must be clean and at the pinned commit; no reset performed.')
    shutil.copytree(source, work, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    (work / 'UPSTREAM_COMMIT.txt').write_text(COMMIT + '\n', encoding='utf-8')
    print(f'Original: {source}\nWorking copy: {work}\nCommit: {COMMIT}')


if __name__ == '__main__':
    main()
