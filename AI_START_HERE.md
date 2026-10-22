# 接手后直接训练所需权重

目标：用服务器GPU训练当前缺失的LoRA24完整检查点，验证后导出并接回Kaggle推理。任务需要实际产出权重，不停在搭环境。

## 已经提供

- `data/train.csv`：2699条标注数据、150个prompt。
- `data/test.csv`、`data/sample_submission.parquet`：公开测试与格式样例。
- `scripts/fetch_model.py`：从ModelScope或Hugging Face自动下载官方Qwen3-8B Base，不需要Kaggle凭据。
- `requirements-gpu.txt`、`requirements-fetch.txt`、`Dockerfile`：环境入口。
- 训练、验证、推理和Kaggle Notebook。

## 直接执行

有CUDA版PyTorch的服务器（包括B200）从仓库根目录执行：

```bash
python -m pip install -r requirements-gpu.txt -r requirements-fetch.txt
python server_start.py --phase check
python server_start.py --phase all --devices 0
```

两张空闲卡用`--devices 0,1`。已有完整base时加`--base-model /模型目录`。需要搭建环境时按`docs/SERVER_QUICKSTART.md`使用Dockerfile。

入口会检查CUDA实际计算、下载base、执行两步短训练和保存/加载验证，然后训练5fold及final，汇总OOF并打包权重，最后运行样例推理。每个模型默认3轮，配置在`configs/train.json`。遇到报错查看日志并修复后继续。

## 必须产出的文件

`artifacts/server-run1/own_lora24_checkpoint.zip`，内部：

```text
own_lora24/final/
  adapter/
  tokenizer/
  base_config/
  heads.safetensors
  readouts.npz
  anchor_features.npz
  training_meta.json
  training_report.json
  COMPLETE.json
```

另外交付`oof_report.json`、`final_parity.json`、`SERVER_RESULT.json`和可运行的推理Notebook。

## 训练后继续

先跑通自己的单模型推理，再按`docs/INTEGRATE_ORIGINAL.md`把权重替换进原分享。完整原融合的其他模型范围见`docs/DEPENDENCY_AUDIT.md`。

随仓库test是官方3行公开样例；正式完整hidden test由Kaggle重跑提供。接手者的GPU任务从上述命令直接开始，不需要用户补发CSV或kaggle.json。
