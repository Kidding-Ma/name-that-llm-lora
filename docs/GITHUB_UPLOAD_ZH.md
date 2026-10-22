# 用户如何上传并交给别人

工作区目录：`D:\Kaggle\NameThatLLM-LoRA-Handoff`。

## 最简单：网页上传

1. 在GitHub创建一个空仓库，例如`name-that-llm-lora-handoff`。
2. 打开这个工作区，只上传其中的代码、配置、文档、`notebooks`、`tests`、`evidence`和隐藏文件`.gitignore`/`.gitattributes`/`.github`；不要上传父目录`D:\Kaggle`。
3. 两个Notebook已经包含运行所需的新代码，可以直接下载上传Kaggle。
4. 把仓库链接和下面的提示词发给有GPU的人。

如果网页上传不方便，可解压本次交付的源码zip再上传。zip内部的顶层目录就是这个项目。

## 用 Git 命令上传（用户自己执行）

先在GitHub创建空仓库，再在PowerShell运行：

```powershell
Set-Location -LiteralPath 'D:\Kaggle\NameThatLLM-LoRA-Handoff'
python scripts/scan_release.py
git init
git add .
git status --short
git diff --cached --stat
git commit -m "Prepare independent LoRA24 GPU training handoff"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

`YOUR_ACCOUNT/YOUR_REPOSITORY`替换成你自己的仓库。`.gitignore`屏蔽数据、模型和运行产物；add后仍应查看status，确认无意外大文件。

## 直接发给对方 AI 的提示词

> 请先读仓库的 AI_START_HERE.md，使用服务器上的 GPU 按步骤准备数据和 Qwen3-8B Base，运行短训练验证，再训练五折和全量 LoRA24，导出权重并接回 Kaggle 推理。具体命令已经写好，遇到报错查看日志并修复，最后交付权重、验证结果和可运行的推理 Notebook。

## 什么不上传

比赛CSV、模型shards、adapter、checkpoint zip、API Token、kaggle.json、预测文件、原始Notebook/API快照、运行日志均不上传此源码仓库。训练权重通过Kaggle Dataset/Notebook Output等单独交付。发布工具采用白名单打包，可避免把后续运行文件误带入源码zip。
