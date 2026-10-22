# AI 执行入口

先读 `AI_START_HERE.md`，按其中步骤在服务器或 Kaggle GPU 上完成训练和推理。

按需查阅：
- `configs/train.json`：训练配置。
- `docs/GPU_RUNBOOK_ZH.md`：环境、命令与故障处理。
- `docs/MODEL_CONTRACT.md`：权重格式与模型实现。
- `docs/INTEGRATE_ORIGINAL.md`：接回原 Kaggle Notebook。

代码修改后执行测试；修改 Python 或配置后重新生成 Kaggle Notebook：

```bash
python scripts/build_notebooks.py
python -m unittest discover -s tests -v
```
