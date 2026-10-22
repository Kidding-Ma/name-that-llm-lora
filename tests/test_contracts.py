import ast
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from ntl.common import LABELS, POOL_LAYERS, epoch_batches, make_plan, sha256, softmax, write_json
from ntl.checkpoint import check_checkpoint

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('patcher', ROOT / 'scripts/patch_notebook.py')
patcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(patcher)

class Contracts(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([{'id': f'{p}-{c}', 'prompt_id': f'p{p}', 'text': 'synthetic test sentence', 'model': c}
                             for p in range(10) for c in LABELS])

    def test_group_split_is_disjoint_and_covers_validation_once(self):
        frame = self.frame(); plan = make_plan(frame)
        coverage = np.zeros(len(frame), int)
        self.assertEqual(plan, make_plan(frame))
        for name, rows in plan['jobs'].items():
            if name == 'final':
                continue
            fit, valid = rows['fit'], rows['valid']
            self.assertFalse(set(frame.iloc[fit].prompt_id) & set(frame.iloc[valid].prompt_id))
            self.assertFalse(set(fit) & set(valid))
            self.assertEqual(set(fit) | set(valid), set(range(len(frame))))
            coverage[valid] += 1
        np.testing.assert_array_equal(coverage, np.ones(len(frame)))

    def test_sampler_distinct_same_prompt_complete_coverage(self):
        groups = np.array(['a'] * 5 + ['b'] * 7)
        fit = np.arange(len(groups))
        batches, audit = epoch_batches(fit, groups, np.random.default_rng(42))
        self.assertEqual(len(batches) % 6, 0)
        self.assertEqual(set(np.concatenate(batches)), set(fit))
        for batch in batches:
            self.assertEqual(len(set(batch)), 3)
            self.assertEqual(len(set(groups[batch])), 1)
        self.assertEqual(audit['unique_fit_rows'], len(fit))

    def test_sampler_does_not_leak_heldout(self):
        groups = np.array(['a'] * 6 + ['b'] * 6)
        batches, _ = epoch_batches(np.arange(6), groups, np.random.default_rng(1))
        self.assertTrue(set(np.concatenate(batches)) <= set(range(6)))
        with self.assertRaises(ValueError):
            epoch_batches(np.arange(2), groups, np.random.default_rng(1))

    def test_softmax_extreme_logits_is_finite(self):
        prob = softmax(np.array([[1e5, 0, -1e5], [-1e5, -1e5, -1e5]]))
        self.assertTrue(np.isfinite(prob).all())
        np.testing.assert_allclose(prob.sum(1), 1)

    def test_patch_changes_only_embedded_member(self):
        embedded = "W={'lora24_public': .177, 'other': .1}\nLORA_RUNS=[('lora24_public','**/lora24/final/COMPLETE.json')]\n"
        source = 'sentinel = 7\n_GAUSSIAN_SOURCE = ' + repr(embedded) + '\nother = 10\n'
        changed, count = patcher.patch_source(source, '**/own_lora24/final/COMPLETE.json')
        self.assertEqual(count, 1)
        tree = ast.parse(changed)
        value = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == '_GAUSSIAN_SOURCE' for t in n.targets))
        self.assertIn('lora24_own', value)
        self.assertNotIn('lora24_public', value)
        self.assertIn("'other': .1", value)
        ast.parse(value)
        self.assertIn('sentinel = 7', changed)

    def test_patch_rejects_already_removed_member(self):
        with self.assertRaises(ValueError):
            patcher.patch_source("_GAUSSIAN_SOURCE = 'x=1'", '**/own_lora24/final/COMPLETE.json')

    def fixture(self, root, smoke=False):
        config = json.loads((ROOT / 'configs/train.json').read_text())
        for name in ['adapter', 'tokenizer', 'base_config']:
            (root / name).mkdir()
        write_json(root / 'adapter/adapter_config.json', {'peft_type': 'LORA', 'r': 16})
        (root / 'adapter/adapter_model.safetensors').write_bytes(b'synthetic-placeholder-not-a-model')
        write_json(root / 'tokenizer/tokenizer_config.json', {})
        write_json(root / 'base_config/config.json', {})
        write_json(root / 'training_meta.json', {'labels': LABELS, 'config': config, 'job': 'final',
                                                'train_sha256': 'synthetic', 'smoke_only': smoke})
        write_json(root / 'training_report.json', {'status': 'SMOKE_COMPLETE' if smoke else 'COMPLETE',
                                                  'successful_updates': 2, 'planned_updates': 2})
        np.savez_compressed(root / 'readouts.npz', classes=np.asarray(LABELS), pool_layers=POOL_LAYERS,
            plain_coef=np.zeros((18, 32768), np.float32), center_coef=np.zeros((18, 32768), np.float32),
            plain_intercept=np.zeros(18), center_intercept=np.zeros(18))
        for name in ['anchor_features.npz', 'validation_features.npz']:
            np.savez_compressed(root / name, features=np.zeros((0, 32768)))
        header = {f'norms.{i}.{field}': {'shape': [4096]} for i in range(8) for field in ['weight', 'bias']}
        raw = json.dumps(header).encode()
        (root / 'heads.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw)
        files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob('*') if p.is_file()}
        marker = root / ('SMOKE_ONLY.json' if smoke else 'COMPLETE.json')
        write_json(marker, {'job': 'final', 'train_sha256': 'synthetic', 'files': files})
        return marker

    def test_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            self.assertEqual(check_checkpoint(root)['job'], 'final')
            (root / 'tokenizer/tokenizer_config.json').write_text('{"modified":true}')
            with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                check_checkpoint(root)

    def test_manifest_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); marker = self.fixture(root)
            content = json.loads(marker.read_text()); content['files']['../outside'] = 'x'
            write_json(marker, content)
            with self.assertRaisesRegex(ValueError, 'Invalid or missing'):
                check_checkpoint(root)

    def test_smoke_is_not_a_production_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root, smoke=True)
            with self.assertRaisesRegex(ValueError, 'Missing completion'):
                check_checkpoint(root)
            self.assertTrue(check_checkpoint(root, allow_smoke=True)['smoke_only'])

    def test_all_python_and_notebook_cells_compile(self):
        for folder in ['ntl', 'scripts', 'tests']:
            for path in (ROOT / folder).glob('*.py'):
                compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        notebooks = list((ROOT / 'notebooks').glob('*.ipynb'))
        self.assertEqual(len(notebooks), 2)
        for path in notebooks:
            nb = json.loads(path.read_text(encoding='utf-8'))
            for cell in nb['cells']:
                if cell['cell_type'] == 'code':
                    self.assertFalse(cell['outputs'])
                    self.assertIsNone(cell['execution_count'])
                    compile(''.join(cell['source']), str(path), 'exec')

    def test_prepare_cli_and_missing_shard_fail_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base = root / 'base'; base.mkdir()
            train = root / 'train.csv'; self.frame().to_csv(train, index=False)
            write_json(base / 'config.json', {'model_type': 'qwen3', 'hidden_size': 4096, 'num_hidden_layers': 36})
            write_json(base / 'tokenizer_config.json', {})
            write_json(base / 'model.safetensors.index.json', {'weight_map': {'x': 'part.safetensors'}})
            command = [sys.executable, str(ROOT / 'scripts/prepare.py'), '--train-csv', str(train),
                       '--base-model', str(base), '--config', str(ROOT / 'configs/train.json'),
                       '--output', str(root / 'plan.json')]
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('shards are missing', failed.stderr)
            self.assertFalse((root / 'plan.json').exists())
            (base / 'part.safetensors').write_bytes(b'synthetic-test-only')
            passed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            plan = json.loads((root / 'plan.json').read_text())
            self.assertEqual(plan['train_sha256'], sha256(train))
            self.assertEqual(len(plan['jobs']), 6)
            # Re-running must not silently rewrite a plan used by existing folds.
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

if __name__ == '__main__':
    unittest.main()
