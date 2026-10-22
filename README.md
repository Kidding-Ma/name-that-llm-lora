# Name That LLM：直接在 GPU 服务器训练权重

**目标：训练当前缺失的 LoRA24 完整检查点，再用于 Kaggle 推理。**

比赛训练集、测试样例和提交样例已放进 `data/`，不需要 Kaggle 凭据。官方 Qwen3-8B Base 由脚本从 ModelScope / Hugging Face 自动下载（约16GB）。

接手 AI 先读 [AI_START_HERE.md](AI_START_HERE.md)，服务器有可用 CUDA PyTorch 时执行：

```bash
python -m pip install -r requirements-gpu.txt -r requirements-fetch.txt
python server_start.py --phase check
python server_start.py --phase all --devices 0
```

`all` 会实际完成：下载base → 短训练与重新加载验证 → 五折和全量训练 → 验证 → 导出完整权重 → 样例推理。

## 输出

`artifacts/server-run1/`：
- `own_lora24_checkpoint.zip`：adapter、tokenizer、heads、readouts、anchors和完成标记。
- `oof_report.json`、`final_parity.json`：验证结果。
- `submission.parquet`：公开3行test的预测。
- `SERVER_RESULT.json`：结果索引。

## 入口和参考

- [服务器操作与Docker](docs/SERVER_QUICKSTART.md)
- [全部依赖检查](docs/DEPENDENCY_AUDIT.md)
- [模型结构与文件格式](docs/MODEL_CONTRACT.md)
- [接回原Notebook](docs/INTEGRATE_ORIGINAL.md)
- `notebooks/01_train_lora24.ipynb`：Kaggle训练入口。
- `notebooks/02_predict_plain.ipynb`：Kaggle推理入口，隐藏test由Kaggle重跑时提供。

本地已完成CPU测试和数据校验；GPU训练由接手方在服务器执行。
