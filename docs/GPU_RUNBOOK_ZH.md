# GPU 执行手册：从空机器到可提交模型

## 0. 先确认目标与环境

第一轮目标是“可训练、可导出、可重新加载、可验证”，不要同时改模型、loss和融合权重。默认3轮训练、5个prompt外折+1个final。独立plain-head先做对照，原复杂融合后接。

本仓库没有GPU实测。请先确认实际GPU、空闲显存、驱动、CUDA版torch和可用运行时长。单卡先跑smoke；双卡只是将两个独立jobs分开。不要将两张16GB卡当作本实现自动支持的32GB单卡。

## A. Kaggle 操作

1. 创建一个自己的训练Notebook，导入`notebooks/01_train_lora24.ipynb`。
2. Add Input → Competition → Name That LLM。
3. Add Input → Models → Qwen3 → Transformers → 8b-base → Version1。标识：`qwen-lm/qwen-3/Transformers/8b-base/1`。
4. 开启CUDA GPU。训练Notebook只需要比赛train和官方base；不需要不可访问的lightningv08权重，也不需要另一个完整推理Notebook的几十GB输入栈。
5. 先执行bootstrap和输入发现。若找到多个train/base，不要使用`hits[0]`瞎选，明确指定正确目录。
6. 保留`RUN_FOLDS=False, RUN_FINAL=False, GPU_DEVICES='0'`，设置独一无二的`RUN_TAG`。
7. 执行smoke：两次成功optimizer更新、导出`SMOKE_ONLY.json`，再执行结构检查与GPU重新加载parity。
8. 如果导入PEFT/Transformers失败，先记录版本和完整异常，再在隔离Notebook里解决。不要照抄旧快照的torchao卸载操作，也不要自动升级所有库。
9. smoke通过后可在同一会话将`RUN_FOLDS/RUN_FINAL`打开，从full-jobs单元格继续。**不要再次运行smoke单元格覆盖已有目录。** 新Save & Run All会从空工作目录按顺序运行；先确认运行时限足够。
10. 双T4且每张卡smoke都可容纳模型时，`GPU_DEVICES='0,1'`，运行5fold+final。每张卡独立加载一个截断8B模型。若host RAM/显存不足，改为单job串行并记录。
11. 末尾生成OOF报告、final parity和`own_lora24_checkpoint.zip`。检查所有输出均成功后Save Version，保存完整运行结果。
12. 把zip解压后的`own_lora24/final/`上传为自己的Kaggle Dataset，或保留训练Notebook Output。确认挂载后能匹配`/kaggle/input/**/own_lora24/final/COMPLETE.json`。不要假设上传zip后Kaggle一定按期望自动解压，必须实际检查。
13. 推理时导入`02_predict_plain.ipynb`，挂载同一个base、比赛数据和自己的checkpoint。输出路径是`/kaggle/working/submission.parquet`。完整Save & Run All及hidden rerun才是提交验证。

Kaggle输出包含代码、计划和日志；若使用整个Notebook Output挂载，选择checkpoint路径时避免误选smoke。生产脚本仅接受final且非smoke的COMPLETE输出。

## B. 有 CUDA GPU 的服务器

以下命令从仓库根目录执行，shell按Linux示例写。`/mnt/...`均需替换为真实路径。Windows有GPU时可用PowerShell运行Python命令；环境变量设置采用PowerShell语法。

### 环境

在独立虚拟环境中，先按官方[PyTorch安装选择器](https://pytorch.org/get-started/locally/)安装匹配驱动的CUDA版torch，然后：

```bash
python -m pip install -r requirements-gpu.txt
python -m unittest discover -s tests -v
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

依赖范围不是已验证lockfile。API实现依据缓存快照的Transformers5风格`dtype=`等接口整理，接收方要在实际环境检查。不能导入时不要直接开始昂贵训练。通过后保存：

```bash
python -m pip freeze > artifacts/run1_environment.txt
nvidia-smi
```

先创建`artifacts`目录再执行freeze。模型和数据从官方授权渠道获取；完整模型shards与tokenizer放同一目录。不要把它们放入Git。

### 准备计划与预检

```bash
python scripts/prepare.py --train-csv /mnt/name-that-llm/train.csv --base-model /mnt/qwen3-8b-base --output artifacts/run1/plan.json
python scripts/preflight.py --plan artifacts/run1/plan.json --output artifacts/run1/preflight.json
```

计划固定train文件哈希、base配置哈希、标签顺序和每折行索引。输入版本需要额外记录：配置哈希不等价于全部模型shards的内容哈希。跨机器迁移同一批fold时要保留原plan字节内容并使用同样挂载路径；或者编写并审计专门的路径迁移工具，不要直接篡改哈希校验。

### 两步更新 smoke

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --plan artifacts/run1/plan.json --job final --max-updates 2 --output artifacts/run1/smoke
python scripts/validate_checkpoint.py --checkpoint artifacts/run1/smoke --allow-smoke
CUDA_VISIBLE_DEVICES=0 python scripts/check_parity.py --checkpoint artifacts/run1/smoke --base-model /mnt/qwen3-8b-base --train-csv /mnt/name-that-llm/train.csv --allow-smoke --output artifacts/run1/smoke_parity.json
```

检查：loss和gradient_norm有限、adapter梯度非零、`status=SMOKE_COMPLETE`、parity passed、没有COMPLETE标记。首次运行包含大量模型加载时间，不能用它直接外推每一步。用report中的update耗时估计完整训练，再加上所有job加载、held-out提取、保存和parity的时间；资源/配额不够时拆分独立jobs。

### 完整训练

单卡先试一个fold：

```bash
python scripts/run_jobs.py --plan artifacts/run1/plan.json --output-root artifacts/run1/checkpoints --devices 0 --jobs fold0
```

确认稳定后继续剩余jobs，单卡串行或双卡并行：

```bash
python scripts/run_jobs.py --plan artifacts/run1/plan.json --output-root artifacts/run1/checkpoints --devices 0,1 --jobs fold0,fold1,fold2,fold3,fold4,final --skip-complete
```

只有一张卡时改成`--devices 0`。`--skip-complete`只允许跳过结构完整且plan/job匹配的已有输出。失败或中断的目录必须保留供诊断，另开run目录重新训练该job；没有optimizer resume功能。

```bash
python scripts/evaluate_oof.py --plan artifacts/run1/plan.json --checkpoints artifacts/run1/checkpoints --output artifacts/run1/oof_report.json
python scripts/validate_checkpoint.py --checkpoint artifacts/run1/checkpoints/final
CUDA_VISIBLE_DEVICES=0 python scripts/check_parity.py --checkpoint artifacts/run1/checkpoints/final --base-model /mnt/qwen3-8b-base --train-csv /mnt/name-that-llm/train.csv --output artifacts/run1/final_parity.json
python scripts/bundle_checkpoint.py --checkpoint artifacts/run1/checkpoints/final --output artifacts/run1/own_lora24_checkpoint.zip
```

### 独立 plain-head 推理对照

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict.py --checkpoint artifacts/run1/checkpoints/final --base-model /mnt/qwen3-8b-base --test-csv /mnt/name-that-llm/test.csv --output artifacts/run1/submission.parquet
```

公开test只有3个示例，生成3行文件只证明接口运行。正式结果要由Kaggle code competition在隐藏test上重新执行。不要把3行样例文件当作完整正式提交文件交付。

## 故障定位

| 报错 | 检查和操作 |
|---|---|
| CUDA unavailable | 是否安装CUDA版torch、驱动是否可见、GPU设置是否开启 |
| 模型shards缺失 | 补齐官方base的全部索引文件，别混入其他尺寸或Instruct模型 |
| OOM | 确认单GPU显存实际空闲；先串行。microbatch3是centered loss的语义，不能直接减为1后假装同一方案 |
| FP16 nonfinite | 保留report，核查dtype/attention/版本；可研究BF16但需新配置及对应推理，T4不按原生BF16假设处理 |
| repeated AMP overflow | 不跳过坏batch伪造完成；诊断scale、梯度和输入，另开实验 |
| parity失败 | 对比base版本、tokenizer、EOS/first-token mask、FP32norms、pool层、adapter加载dtype |
| validation rows mismatch | 数据顺序/内容或plan变了，不能强行拼接OOF |
|原融合继续缺文件 | 查`INTEGRATE_ORIGINAL.md`的其他成员输入，不是自己LoRA训练失败 |

## 停止与交付

收到明确停止请求时停止子进程并保存失败/中断记录。完成后交付：final检查点zip、其hash、GPU parity、OOF报告、运行环境、提交Notebook以及真实提交回执。GitHub仅提交不含私有数据的汇总。
