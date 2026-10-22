# Name That LLM GPU 训练交接

用比赛训练集训练 Qwen3-8B LoRA24 权重，补齐原 Kaggle Notebook 缺失的模型，再运行推理。

**接手 AI 从 [AI_START_HERE.md](AI_START_HERE.md) 开始，里面有完整执行命令。**

当前：CPU 测试已通过；下一步是在 GPU 服务器或 Kaggle 上运行短训练，再做完整训练。

## 主要文件

| 文件 | 用途 |
|---|---|
| `AI_START_HERE.md` | 读什么、准备什么、运行什么、交付什么 |
| `configs/train.json` | 训练参数 |
| `scripts/train.py` | 训练一个 fold 或 final 模型 |
| `scripts/run_jobs.py` | 用一张或多张 GPU 运行训练任务 |
| `scripts/check_parity.py` | 检查权重保存和重新加载 |
| `scripts/evaluate_oof.py` | 汇总五折验证结果 |
| `scripts/predict.py` | 单模型推理 |
| `scripts/patch_notebook.py` | 将权重接回原 Notebook |
| `notebooks/01_train_lora24.ipynb` | 可直接上传 Kaggle 的训练入口 |
| `notebooks/02_predict_plain.ipynb` | 可直接上传 Kaggle 的推理入口 |

## 需要的输入

- 比赛：`name-that-llm` 的训练和测试数据。
- 基础模型：`qwen-lm/qwen-3/Transformers/8b-base/1`。
- 有 CUDA GPU 的服务器或 Kaggle Notebook。

## 按需参考

- [GPU 运行手册](docs/GPU_RUNBOOK_ZH.md)
- [模型与文件格式](docs/MODEL_CONTRACT.md)
- [接回原 Notebook](docs/INTEGRATE_ORIGINAL.md)
- [GitHub 网页上传步骤](docs/GITHUB_UPLOAD_ZH.md)

完整训练产物是 `own_lora24/final/`。源码上传 GitHub；数据与权重保存在服务器或 Kaggle。
