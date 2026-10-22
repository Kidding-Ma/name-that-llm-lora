import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('server_start', ROOT / 'server_start.py')
server = importlib.util.module_from_spec(spec); spec.loader.exec_module(server)
spec2 = importlib.util.spec_from_file_location('fetch_model', ROOT / 'scripts/fetch_model.py')
fetcher = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(fetcher)

class ServerInputs(unittest.TestCase):
    def test_real_bundled_dataset_is_complete_without_credentials(self):
        info = server.check_data()
        self.assertEqual(info['train_rows'], 2699)
        self.assertEqual(info['prompts'], 150)
        self.assertEqual(info['test_rows'], 3)
        self.assertFalse(info['kaggle_credential_required_for_training'])

    def test_model_download_validation_rejects_truncated_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {'model_type':'qwen3','hidden_size':4096,'num_hidden_layers':36,
                   'intermediate_size':12288,'vocab_size':151936}
            (root / 'config.json').write_text(json.dumps(cfg))
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a':'part.safetensors'}}))
            for name in ['tokenizer.json','tokenizer_config.json']:
                (root / name).write_text('{}')
            (root / 'part.safetensors').write_bytes(b'partial download')
            with self.assertRaisesRegex(ValueError, 'Missing/incomplete'):
                fetcher.validate_model(root)

if __name__ == '__main__':
    unittest.main()
