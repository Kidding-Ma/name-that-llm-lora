"""Scan only the explicit public-source release allowlist; no GPU required."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {'README.md', 'AGENTS.md', 'AI_START_HERE.md', 'LICENSE', '.gitignore', '.gitattributes',
              'requirements-cpu.txt', 'requirements-gpu.txt'}
PUBLIC_DIRS = {'ntl', 'scripts', 'configs', 'docs', 'notebooks', 'tests', 'evidence', '.github'}
SECRET_PATTERNS = [r'ghp_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{30,}',
                   r'hf_[A-Za-z0-9]{25,}', r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
FORBIDDEN_SUFFIX = {'.csv', '.tsv', '.parquet', '.npy', '.npz', '.safetensors', '.pt', '.pth', '.bin', '.zip', '.log'}

def release_files(root=ROOT):
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        rel = path.relative_to(root)
        if (len(rel.parts) == 1 and path.name in ROOT_FILES) or rel.parts[0] in PUBLIC_DIRS:
            yield path

def scan(root=ROOT):
    problems = []
    files = list(release_files(root))
    for path in files:
        rel = path.relative_to(root)
        if path.is_symlink():
            problems.append(f'{rel}: symlink not allowed')
        if path.suffix.lower() in FORBIDDEN_SUFFIX or path.name.lower() in {'kaggle.json', '.env'}:
            problems.append(f'{rel}: data/weight/credential artifact')
        if path.stat().st_size > 2 * 1024 * 1024:
            problems.append(f'{rel}: exceeds 2 MiB public source limit')
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeError:
            problems.append(f'{rel}: non-text file in source release')
            continue
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, text):
                problems.append(f'{rel}: possible credential; inspect without printing matched data')
        if path.suffix == '.ipynb':
            try:
                nb = json.loads(text)
                for cell in nb['cells']:
                    if cell.get('outputs') or cell.get('execution_count') is not None:
                        problems.append(f'{rel}: saved notebook execution data')
            except (ValueError, KeyError):
                problems.append(f'{rel}: malformed notebook')
    return files, problems

if __name__ == '__main__':
    files, problems = scan()
    print(json.dumps({'status': 'FAIL' if problems else 'PASS', 'public_files': len(files),
                      'problems': problems, 'scope': 'Explicit source allowlist; runtime dirs excluded'}, indent=2))
    raise SystemExit(bool(problems))
