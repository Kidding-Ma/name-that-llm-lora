# GPU 运行交付报告（复制到私有运行记录后填写）

- 日期/时区：
- 执行人/AI：
- 仓库commit或发布zip hash：
- 实验名称：
- 平台/Notebook URL/版本ID：
- GPU数量、名称、每卡显存/空闲显存：
- OS/Python/driver/CUDA/torch/transformers/peft/accelerate/safetensors：
- Competition输入版本、train.csv SHA-256：
- 官方base输入版本、config哈希、模型shard版本或哈希：
- plan.json/config/source hash：

## 执行结果

| 阶段 | 状态 | 证据文件/URL |
|---|---|---|
| CPU tests/release scan | NOT_RUN | |
| preflight | NOT_RUN | |
| two-update smoke | NOT_RUN | |
| smoke structure | NOT_RUN | |
| smoke GPU parity | NOT_RUN | |
| fold0..4 | NOT_RUN | |
| plain OOF | NOT_RUN | |
| final3epochs | NOT_RUN | |
| final structure/hash | NOT_RUN | |
| final GPU parity | NOT_RUN | |
| plain inference control | NOT_RUN | |
| ensemble replacement | NOT_RUN | |
| Kaggle hidden rerun | NOT_RUN | |

- 每job成功更新步数/耗时/peak显存/overflow重试：
- smoke/final parity mean relative error：
- 各折及aggregate plain balanced accuracy：
- centered-head或融合新增验证（没有则写未做）：
- checkpoint zip位置、SHA-256、文件大小：
- hidden test行数、ID顺序/唯一性、prediction counts：
- submission hash、Notebook版本、真实提交状态：
- public/private score：null（未评分时保持null）

## 变更与局限

- 与默认配置相比改变了什么、为什么：
- 尚未通过的检查/实际阻塞：
- 下一个可执行步骤：

公开GitHub汇总只保留不敏感的硬件、版本、哈希和指标；不要粘贴训练文本、预测明细、API响应、账号密钥或私有路径。
