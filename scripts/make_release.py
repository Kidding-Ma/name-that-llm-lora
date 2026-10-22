"""Build a source-only zip from the scanned release allowlist."""
import argparse
import json
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_release import ROOT, scan
from ntl.common import sha256, write_json

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    files, problems = scan()
    if problems:
        raise ValueError('\n'.join(problems))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, ROOT.name + '/' + path.relative_to(ROOT).as_posix())
    report = {'file': output.name, 'files': len(files), 'bytes': output.stat().st_size,
              'sha256': sha256(output), 'scope': 'source/config/docs/code-only notebooks; no data or weights'}
    write_json(str(output) + '.sha256.json', report)
    print(json.dumps(report, indent=2))
