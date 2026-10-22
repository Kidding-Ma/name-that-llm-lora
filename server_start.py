"""One entry: bundled data -> public base download -> smoke -> training -> prediction."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

def run(*args):
    print('+ python ' + ' '.join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)

def check_data(root=ROOT):
    from ntl.common import read_train, sha256
    import pandas as pd
    manifest = json.loads((root / 'data/manifest.json').read_text(encoding='utf-8'))
    for name, spec in manifest['files'].items():
        path = root / 'data' / name
        if not path.is_file() or sha256(path) != spec['sha256']:
            raise ValueError(f'Missing/changed bundled input: {path}; restore it from the repository')
    frame = read_train(root / 'data/train.csv')
    test = pd.read_csv(root / 'data/test.csv', dtype=str, keep_default_na=False)
    if len(frame) != 2699 or frame.prompt_id.nunique() != 150:
        raise ValueError('Unexpected bundled training set')
    if list(test.columns) != ['id', 'text'] or not test.id.is_unique:
        raise ValueError('Unexpected test schema')
    return {'train_rows': len(frame), 'prompts': int(frame.prompt_id.nunique()),
            'test_rows': len(test), 'kaggle_credential_required_for_training': False}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['check', 'fetch', 'smoke', 'train', 'predict', 'all'], default='all')
    p.add_argument('--devices', default='0', help='Visible GPU indices, e.g. 0 or 0,1')
    p.add_argument('--run', default='server-run1', help='Fresh name when changing inputs/configuration')
    p.add_argument('--base-model', help='Existing complete base model; otherwise downloads public official model')
    p.add_argument('--model-source', choices=['auto', 'modelscope', 'huggingface'], default='auto')
    args = p.parse_args()
    os.chdir(ROOT)
    from ntl.common import write_json, sha256
    summary = check_data()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.phase == 'check':
        run('-m', 'unittest', 'discover', '-s', 'tests', '-v')
        print('Repository + bundled inputs ready. GPU training is the next step.', flush=True)
        return
    if args.phase != 'fetch':
        try:
            import torch
            assert torch.cuda.is_available(), 'CUDA GPU unavailable'
            # A real kernel catches incompatible Blackwell/CUDA installations early.
            x = torch.randn(32, 32, device='cuda', requires_grad=True)
            (x @ x).square().mean().backward(); torch.cuda.synchronize()
            assert bool(torch.isfinite(x.grad).all()), 'GPU kernel produced nonfinite gradients'
            print(f'GPU kernel passed: {torch.cuda.get_device_name(0)}; torch {torch.__version__}', flush=True)
            del x
            torch.cuda.empty_cache()
        except Exception as error:
            raise RuntimeError('Prepare a working CUDA PyTorch environment first; use the supplied Dockerfile '
                               'or the server GPU environment. No Kaggle credential is needed.') from error
    from fetch_model import fetch, validate_model
    if args.base_model:
        base = Path(args.base_model).resolve(); validate_model(base)
    else:
        base = fetch(ROOT / 'models/Qwen3-8B-Base', args.model_source)
    if args.phase == 'fetch':
        return
    folder = ROOT / 'artifacts' / args.run
    plan = folder / 'plan.json'
    if not plan.exists():
        run('scripts/prepare.py', '--train-csv', ROOT / 'data/train.csv', '--base-model', base, '--output', plan)
    else:
        existing = json.loads(plan.read_text())
        if existing['base_model'] != str(base) or existing['train_sha256'] != sha256(ROOT / 'data/train.csv'):
            raise ValueError('Run name already belongs to different inputs; select a fresh --run')
        if existing['config'] != json.loads((ROOT / 'configs/train.json').read_text()):
            raise ValueError('Configuration changed; select a fresh --run')
    run('scripts/preflight.py', '--plan', plan, '--output', folder / 'preflight.json')
    env = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (folder / 'environment.txt').write_text(env.stdout, encoding='utf-8')
    smoke = folder / 'smoke'
    if args.phase in ['smoke', 'train', 'all']:
        if not smoke.exists():
            run('scripts/train.py', '--plan', plan, '--job', 'final', '--max-updates', 2, '--output', smoke)
        run('scripts/validate_checkpoint.py', '--checkpoint', smoke, '--allow-smoke')
        meta = json.loads((smoke / 'training_meta.json').read_text())
        if meta['plan_sha256'] != sha256(plan):
            raise ValueError('Existing smoke belongs to a different plan')
        run('scripts/check_parity.py', '--checkpoint', smoke, '--base-model', base,
            '--train-csv', ROOT / 'data/train.csv', '--allow-smoke', '--output', folder / 'smoke_parity.json')
    if args.phase == 'smoke':
        return
    if args.phase in ['train', 'all']:
        run('scripts/run_jobs.py', '--plan', plan, '--output-root', folder / 'checkpoints',
            '--devices', args.devices, '--skip-complete')
        run('scripts/evaluate_oof.py', '--plan', plan, '--checkpoints', folder / 'checkpoints',
            '--output', folder / 'oof_report.json')
    final = folder / 'checkpoints/final'
    run('scripts/validate_checkpoint.py', '--checkpoint', final)
    run('scripts/check_parity.py', '--checkpoint', final, '--base-model', base,
        '--train-csv', ROOT / 'data/train.csv', '--output', folder / 'final_parity.json')
    bundle = folder / 'own_lora24_checkpoint.zip'
    if not bundle.exists():
        run('scripts/bundle_checkpoint.py', '--checkpoint', final, '--output', bundle)
    if args.phase in ['predict', 'all']:
        submission = folder / 'submission.parquet'
        if not submission.exists():
            run('scripts/predict.py', '--checkpoint', final, '--base-model', base,
                '--test-csv', ROOT / 'data/test.csv', '--output', submission)
    write_json(folder / 'SERVER_RESULT.json', {'status': 'COMPLETE', 'phase': args.phase,
        'checkpoint': str(final), 'checkpoint_zip': str(bundle), 'bundled_test_rows': summary['test_rows'],
        'test_scope': '3 public example rows; Kaggle generates the full hidden test during code submission',
        'kaggle_score': None, 'oof_report': str(folder / 'oof_report.json')})
    print(f'Done. Results: {folder}', flush=True)

if __name__ == '__main__':
    main()
