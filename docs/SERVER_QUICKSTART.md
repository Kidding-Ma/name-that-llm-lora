# 服务器直接开始

训练和样例推理需要的三份比赛输入已在`data/`，不需要Kaggle凭据。

## 现有 CUDA PyTorch 环境（包括 B200）

```bash
git clone https://github.com/Kidding-Ma/name-that-llm-lora.git
cd name-that-llm-lora
python -m pip install -r requirements-gpu.txt -r requirements-fetch.txt
python server_start.py --phase check
python server_start.py --phase all --devices 0
```

已有克隆用`git pull --ff-only`更新即可。两张空闲卡可用`--devices 0,1`。

脚本先执行真实CUDA矩阵乘法和反向传播，再下载官方`Qwen/Qwen3-8B-Base`，优先ModelScope，失败尝试Hugging Face。完整模型约16GB，放在`models/`。若服务器已有完整模型，加`--base-model /实际模型目录`。

随后执行：输入检查 → 两步短训练 → 保存/重新加载对比 → 五折和final训练 → OOF评估 → 权重打包 → 本地样例推理。完成后的文件位于`artifacts/server-run1/`：

- `own_lora24_checkpoint.zip`：完整自训练权重；
- `oof_report.json`：五折验证；
- `final_parity.json`：重新加载结果；
- `submission.parquet`：随仓库3行样例test的预测；
- `SERVER_RESULT.json`：运行结果索引。

分阶段运行可使用`--phase fetch`、`--phase smoke`、`--phase train`、`--phase predict`；默认`all`。换配置时用新名称，例如`--run server-run2`。

## 用 Docker 准备 GPU 环境

```bash
docker build -t ntl-lora .
docker run --rm --gpus all --shm-size=16g -v "$PWD:/workspace" -w /workspace ntl-lora python server_start.py --phase all --devices 0
```

Dockerfile默认采用NVIDIA PyTorch25.06（CUDA12.9.1）作为Blackwell可用环境起点。若服务器已有正常B200环境，直接使用上一节即可；如宿主驱动与镜像不匹配，按服务器驱动选择官方镜像覆盖`BASE_IMAGE`。脚本会先执行真实CUDA内核，尽早报告环境问题。

官方参考：
- https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-06.html
- https://modelscope.cn/models/Qwen/Qwen3-8B-Base
- https://huggingface.co/Qwen/Qwen3-8B-Base

## 完整 Kaggle 隐藏测试

仓库`test.csv`是官方公开的3行示例，隐藏测试不能提前下载。权重训练完成后，将权重作为Kaggle Input挂载，使用`notebooks/02_predict_plain.ipynb`，由Kaggle重新运行生成完整结果并计分。

## 原分享的完整融合

`server_start.py`完成自训练LoRA与独立推理，原分享还使用多个其他模型。集成路线见`INTEGRATE_ORIGINAL.md`，完整依赖范围见`DEPENDENCY_AUDIT.md`。
