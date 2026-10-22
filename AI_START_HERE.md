# 接手后直接执行

目标：在 GPU 服务器上，用比赛训练集训练 LoRA24 权重，接回 Kaggle 推理，生成可提交结果。

## 1. 先看这三个文件

1. `README.md`：项目文件位置。
2. `configs/train.json`：默认训练参数。
3. `docs/GPU_RUNBOOK_ZH.md`：安装依赖和运行命令。

现有代码已通过 CPU 测试，尚未在 GPU 上运行。接手后从短训练开始验证。

## 2. 准备服务器和输入

- 运行 `nvidia-smi`，确认 GPU 型号、数量和空闲显存。
- 准备 CUDA 版 PyTorch，安装 `requirements-gpu.txt` 中的依赖。
- 下载比赛 `name-that-llm` 的 `train.csv`、`test.csv`。
- 下载完整官方 Qwen3-8B Base，Kaggle 标识为 `qwen-lm/qwen-3/Transformers/8b-base/1`。
- 在下面命令中，把 `/mnt/name-that-llm` 和 `/mnt/qwen3-8b-base` 换成真实路径。

## 3. 先运行短训练

从仓库根目录执行：

```bash
python scripts/prepare.py --train-csv /mnt/name-that-llm/train.csv --base-model /mnt/qwen3-8b-base --output artifacts/run1/plan.json
python scripts/preflight.py --plan artifacts/run1/plan.json --output artifacts/run1/preflight.json
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --plan artifacts/run1/plan.json --job final --max-updates 2 --output artifacts/run1/smoke
python scripts/validate_checkpoint.py --checkpoint artifacts/run1/smoke --allow-smoke
CUDA_VISIBLE_DEVICES=0 python scripts/check_parity.py --checkpoint artifacts/run1/smoke --base-model /mnt/qwen3-8b-base --train-csv /mnt/name-that-llm/train.csv --allow-smoke --output artifacts/run1/smoke_parity.json
```

解决报错，确认短训练、保存和重新加载通过，然后继续。

## 4. 完整训练并导出权重

下面使用两张 GPU 分配独立任务；只有一张卡时，把 `--devices 0,1` 改成 `--devices 0`。

```bash
python scripts/run_jobs.py --plan artifacts/run1/plan.json --output-root artifacts/run1/checkpoints --devices 0,1 --jobs fold0,fold1,fold2,fold3,fold4,final
python scripts/evaluate_oof.py --plan artifacts/run1/plan.json --checkpoints artifacts/run1/checkpoints --output artifacts/run1/oof_report.json
python scripts/validate_checkpoint.py --checkpoint artifacts/run1/checkpoints/final
CUDA_VISIBLE_DEVICES=0 python scripts/check_parity.py --checkpoint artifacts/run1/checkpoints/final --base-model /mnt/qwen3-8b-base --train-csv /mnt/name-that-llm/train.csv --output artifacts/run1/final_parity.json
python scripts/bundle_checkpoint.py --checkpoint artifacts/run1/checkpoints/final --output artifacts/run1/own_lora24_checkpoint.zip
```

训练配置是五折验证加一个全量模型，每个训练三轮。原提交缺少的就是这一类 LoRA24 检查点。

## 5. 跑推理并接回原方案

先跑自己的单模型推理：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict.py --checkpoint artifacts/run1/checkpoints/final --base-model /mnt/qwen3-8b-base --test-csv /mnt/name-that-llm/test.csv --output artifacts/run1/submission.parquet
```

然后按照 `docs/INTEGRATE_ORIGINAL.md`，把生成的权重添加到 Kaggle，用 `scripts/patch_notebook.py` 接回原分享，跑完整推理。

原分享：
https://www.kaggle.com/code/kiza123123/ntl-0604-dual-lora-gaussian-hungarian?scriptVersionId=360450415

如果直接在 Kaggle 工作，上传 `notebooks/01_train_lora24.ipynb` 开始训练；`02_predict_plain.ipynb` 用于单模型推理。

## 6. 最后交付

- `own_lora24_checkpoint.zip` 和权重下载位置。
- 训练及验证结果、实际使用的 GPU 和依赖版本。
- 能在 Kaggle 上运行的推理 Notebook 与提交结果。

遇到问题先看日志并修复，具体格式和排查方法按需查阅 `docs/`。
