# 从克隆到交付的依赖检查

| 环节 | 所需内容 | 本仓库提供方式 |
|---|---|---|
| 标注训练 | train.csv，2699行/150prompt/18类 | 已包含`data/train.csv`及SHA-256 |
| 样例推理 | test.csv，3行 | 已包含`data/test.csv` |
| 提交格式参考 | sample_submission.parquet | 已包含`data/sample_submission.parquet` |
| 官方基础模型 | Qwen3-8B-Base完整shards/tokenizer/config | `fetch_model.py`从两个官方公开源下载，无需Kaggle密钥 |
| GPU环境 | CUDA PyTorch、transformers、PEFT等 | requirements + Dockerfile + CUDA实际内核预检 |
| 数据划分和采样 | prompt分折、三答案microbatch | `prepare.py`/`ntl.common`，CPU契约测试 |
| 权重训练 | 5fold+final | `train.py`/`run_jobs.py`，入口自动串联 |
| 文件和模型一致性 | 文件manifest、readouts、anchors | validate_checkpoint + check_parity |
| 验证 | prompt-held-out OOF | evaluate_oof |
| 交付 | final权重zip、报告、样例预测 | server_start自动生成 |
| 正式隐藏测试 | Kaggle评测器动态提供 | 推理Notebook在Kaggle中重新运行 |

模型约16GB，未放入Git普通文件；下载脚本、来源、验证和复用本地路径均已提供。代码运行不需要先拿到旧LoRA检查点。

## 原0.604分享的额外依赖

原分享不是单个LoRA。其其他输入包括：two-stage 8B/14B adapters和public pool、Qwen3-14B Base、Qwen2.5-7B、GLM9B、DeBERTa/Electra/Desklib、seed7/Claude/Qwen2.5 LoRA、两个token-attention head、Qwen3.5 backbone、UltraBERT五折及extra passages。它们不参与本仓库的独立LoRA训练/推理。

这些第三方权重没有被本次检查证明全部可公开下载，也没有被伪装成“仓库已经包含”。需要原完整融合时，在训练自身权重后按`INTEGRATE_ORIGINAL.md`继续恢复这些输入；若其中还有不可访问的权重，应为对应成员寻找其公开训练入口并单独训练。`scripts/patch_notebook.py`只替换此次缺失的LoRA24成员，不能自动生成其他架构的权重。

## 检查边界

本机已执行数据字节校验、完整性检查和CPU测试；没有GPU，未实际下载16GB模型、构建CUDA镜像或运行GPU训练。接手AI可直接从`python server_start.py --phase check`开始，不再向用户索取训练数据和Kaggle凭据。
