"""Generate self-contained code-only Kaggle entry notebooks from this repository."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}

def code(text):
    compile(text, '<notebook-cell>', 'exec')
    return {'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(keepends=True),
            'outputs': [], 'execution_count': None}

def build():
    files = {p.relative_to(ROOT).as_posix(): p.read_text(encoding='utf-8')
             for folder, pattern in [('ntl', '*.py'), ('scripts', '*.py'), ('configs', '*.json')]
             for p in sorted((ROOT / folder).glob(pattern)) if p.name != 'build_notebooks.py'}
    bootstrap = ('import json, os, sys, subprocess\nfrom pathlib import Path\n'
        'ROOT = Path("/kaggle/working/ntl_handoff")\nROOT.mkdir(exist_ok=True)\n'
        f'BUNDLE = {files!r}\n'
        'for name, source in BUNDLE.items():\n'
        '    target = ROOT / name\n    target.parent.mkdir(parents=True, exist_ok=True)\n'
        '    target.write_text(source, encoding="utf-8")\n'
        'os.chdir(ROOT)\nsys.path.insert(0, str(ROOT))\n'
        'def run(*args):\n    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)\n')
    inputs = '''# Find the official inputs. If ambiguous, set the two paths explicitly.
inputs = Path('/kaggle/input')
trains = [p for p in inputs.rglob('train.csv') if (p.parent / 'test.csv').is_file()]
bases = []
for path in inputs.rglob('config.json'):
    try:
        value = json.loads(path.read_text())
    except (ValueError, OSError):
        continue
    if (value.get('model_type'), value.get('hidden_size'), value.get('num_hidden_layers')) == ('qwen3', 4096, 36):
        bases.append(path.parent)
if len(trains) != 1 or len(bases) != 1:
    raise RuntimeError(f'Select one competition and one official Qwen3-8B Base: {trains}, {bases}')
TRAIN_CSV, BASE_MODEL = trains[0], bases[0]
'''
    training = [markdown('''# Own LoRA24 training handoff
Upload this notebook to Kaggle. Add Name That LLM competition and Qwen3-8B Base Transformers V1.
Select a CUDA GPU (two T4s can run independent jobs). This is a new, GPU-UNVERIFIED implementation.
Default execution performs a 2-update smoke and reload parity only. Read docs/GPU_RUNBOOK_ZH.md.
No competition data, weights, tokens or old notebook outputs are embedded here.
'''), code(bootstrap), code(inputs), code('''# Change RUN_TAG for every new smoke/full run. Never overwrite a failed run.
RUN_TAG = 'own_lora24_v1'
RUN_FOLDS = False   # enable after smoke passes to obtain honest prompt-held-out validation
RUN_FINAL = False   # enable after smoke passes to obtain the deployable full-data checkpoint
GPU_DEVICES = '0'   # use '0,1' only if both GPUs have enough free memory for independent models
RUN_ROOT = ROOT / 'artifacts' / RUN_TAG
PLAN = RUN_ROOT / 'plan.json'
if not PLAN.exists():
    run('scripts/prepare.py', '--train-csv', TRAIN_CSV, '--base-model', BASE_MODEL, '--output', PLAN)
run('scripts/preflight.py', '--plan', PLAN, '--output', RUN_ROOT / 'preflight.json')
'''), code('''# A smoke checkpoint intentionally has NO COMPLETE.json.
SMOKE = RUN_ROOT / 'smoke'
run('scripts/train.py', '--plan', PLAN, '--job', 'final', '--max-updates', '2', '--output', SMOKE)
run('scripts/validate_checkpoint.py', '--checkpoint', SMOKE, '--allow-smoke')
run('scripts/check_parity.py', '--checkpoint', SMOKE, '--base-model', BASE_MODEL,
    '--train-csv', TRAIN_CSV, '--allow-smoke', '--output', RUN_ROOT / 'smoke_parity.json')
'''), code('''# Every selected job starts from the official base; no reuse of a full-data adapter for CV.
if RUN_FOLDS or RUN_FINAL:
    assert json.loads((RUN_ROOT / 'smoke_parity.json').read_text())['passed']
    jobs = ([f'fold{i}' for i in range(5)] if RUN_FOLDS else []) + (['final'] if RUN_FINAL else [])
    run('scripts/run_jobs.py', '--plan', PLAN, '--output-root', RUN_ROOT / 'checkpoints',
        '--devices', GPU_DEVICES, '--jobs', ','.join(jobs))
else:
    print('Smoke only. Enable RUN_FOLDS / RUN_FINAL when ready, then run this cell onward.')
'''), code('''if RUN_FOLDS:
    run('scripts/evaluate_oof.py', '--plan', PLAN, '--checkpoints', RUN_ROOT / 'checkpoints',
        '--output', RUN_ROOT / 'oof_report.json')
if RUN_FINAL:
    FINAL = RUN_ROOT / 'checkpoints/final'
    run('scripts/validate_checkpoint.py', '--checkpoint', FINAL)
    run('scripts/check_parity.py', '--checkpoint', FINAL, '--base-model', BASE_MODEL,
        '--train-csv', TRAIN_CSV, '--output', RUN_ROOT / 'final_parity.json')
    run('scripts/bundle_checkpoint.py', '--checkpoint', FINAL,
        '--output', RUN_ROOT / 'own_lora24_checkpoint.zip')
''')]
    inference = [markdown('''# Own LoRA24 plain-head control
Add the exported own_lora24 Dataset, competition data and the same Qwen3-8B Base.
This predicts with the independently trained plain head; it is NOT the original .604 ensemble.
Set a CUDA GPU. Save & Run All must regenerate submission.parquet on the hidden test.
'''), code(bootstrap), code(inputs), code('''markers = list(Path('/kaggle/input').glob('**/own_lora24/final/COMPLETE.json'))
if len(markers) != 1:
    raise RuntimeError(f'Expected one own_lora24 checkpoint, found {markers}')
CHECKPOINT = markers[0].parent
run('scripts/validate_checkpoint.py', '--checkpoint', CHECKPOINT)
run('scripts/check_parity.py', '--checkpoint', CHECKPOINT, '--base-model', BASE_MODEL,
    '--train-csv', TRAIN_CSV, '--output', '/kaggle/working/parity.json')
run('scripts/predict.py', '--checkpoint', CHECKPOINT, '--base-model', BASE_MODEL,
    '--test-csv', TRAIN_CSV.parent / 'test.csv', '--output', '/kaggle/working/submission.parquet')
''')]
    for name, cells in [('01_train_lora24.ipynb', training), ('02_predict_plain.ipynb', inference)]:
        nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': cells}
        for index, cell in enumerate(cells):
            cell['id'] = f'cell-{index}'
        (ROOT / 'notebooks' / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
        print(f'Built notebooks/{name}')

if __name__ == '__main__':
    build()
