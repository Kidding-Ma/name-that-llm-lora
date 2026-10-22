# AI 执行入口

先读 `AI_START_HERE.md`，用服务器GPU实际训练并导出所需LoRA24权重。

数据已在data/，模型由scripts/fetch_model.py公开下载。运行入口是server_start.py。
按需看docs/SERVER_QUICKSTART.md、docs/MODEL_CONTRACT.md和docs/INTEGRATE_ORIGINAL.md。

修改Python或配置后执行：

```bash
python scripts/build_notebooks.py
python -m unittest discover -s tests -v
```
