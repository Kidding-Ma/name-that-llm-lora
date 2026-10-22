> 当前服务器入口：`python server_start.py --phase all --devices 0`。三份比赛输入已在仓库data/，base由脚本自动下载，无需Kaggle凭据。直接操作见 [SERVER_QUICKSTART.md](SERVER_QUICKSTART.md)。

# 输入与来源

## 训练自己的 LoRA24 所需

| 输入 | 获取方式 | 用途 |
|---|---|---|
| Name That LLM train.csv | [比赛Data](https://www.kaggle.com/competitions/name-that-llm/data) | 标注训练；包含id,prompt_id,text,model |
| Qwen3-8B Base Transformers V1 | [官方Kaggle模型](https://www.kaggle.com/models/qwen-lm/qwen-3/Transformers/8b-base/1) | 36层官方base，截断24层 |
| 本项目代码/config | 此仓库或自包含Notebook | 训练、导出、验证 |

训练阶段不读取test标签，也不需要原来不可访问的LoRA权重。已有的`lora8b_v1`/`lora14b_v1`是另外两个adapter，不能改名冒充缺失成员。

## 与本次错误直接相关的证据

- 原分享：`kiza123123/ntl-0604-dual-lora-gaussian-hungarian`，已查看版本2（scriptVersionId=360450415）。页面显示public score0.604，正常运行日志约3h51m。
- 用户失败版本：`kiddingma/ntl-sota-0604-dual-lora-gaussian-hungarian`，scriptVersionId=360958369，日志AssertionError指向`lora24_public`。
- 原成功日志的来源目录：`/kaggle/input/notebooks/lightningv08/ntl-lora24-fivefold-training/lora24/final`。
- 当前浏览器访问该训练Notebook显示找不到页面；原Input有`[Private Datasource]`。不对其私有化/删除原因作断言。
- 本机已有训练Notebook代码快照及元数据，缓存metadata曾标记public，最后运行时间字段为2026-10-17。该历史字段不证明当前可访问，也不证明本次训练已经完成。
- 缓存source报告自己的训练recipe与更早公开adapter并非完全复现；我们也不承诺当前重写实现与缓存输出相同。

源文件SHA-256保存在`evidence/source_provenance.json`；仓库不携带这些大型原始API响应或嵌套source。

## 原融合的其他输入（不属于本项目从头训练的范围）

用户复制Notebook的Input已经有下列条目，但是否每个版本都能挂载并通过推理，需要接手者实际核验：

- two-stage LoRA checkpoints/public pool（lora8b_v1、lora14b_v1）及Qwen3-14B Base；
- GLM4-9B截断模型、DeBERTa、Electra、Desklib；
- `name-that-llm-train-lora24-s7` Notebook Output；
- `ntl-qwen3-8b-lora-claude-head` Dataset；
- `ntl-train-lora-qwen25-instyle` Notebook Output和Qwen2.5-7B；
- 两个token-attention训练输出和Qwen3.5 backbone；
- UltraBERT五折权重；
- 原源代码引用的extra passages等文本池。

完整原融合依赖远多于本项目的两个训练输入。既有公开可见性不是重新分发许可；数据、base、adapter和第三方代码保留各自条款。接收方应从原渠道挂载并记录版本；此GitHub包只发新代码、配置、文档和摘要。
