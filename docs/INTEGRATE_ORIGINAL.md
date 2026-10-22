# 把自己的权重接回原分享

先完成自己的GPU smoke、完整final训练、结构校验与reload parity。集成时保留独立plain-head对照；不要一开始就同时启动几十GB原融合输入。

## 1. 打包与挂载

```bash
python scripts/bundle_checkpoint.py --checkpoint artifacts/run1/checkpoints/final --output artifacts/run1/own_lora24_checkpoint.zip
```

接收方解压后把`own_lora24/final/`上传自己的Kaggle Dataset。或者保持同样目录结构的Notebook Output。手动核实挂载后的这个glob有且只有一个目标：

```python
import glob
hits = glob.glob('/kaggle/input/**/own_lora24/final/COMPLETE.json', recursive=True)
assert len(hits) == 1, hits
print(hits[0])
```

多份同名checkpoint时明确选择唯一版本，避免代码排序取到错误模型。

## 2. 下载原Notebook源码

从原页面的Download菜单下载`.ipynb`，保存到`private_runs/original.ipynb`：

<https://www.kaggle.com/code/kiza123123/ntl-0604-dual-lora-gaussian-hungarian?scriptVersionId=360450415>

使用这个已检查版本，而不是假定作者更新后的源码结构不变。保留原文件及哈希，不提交进本GitHub包。

## 3. 用补丁工具替换一个成员

```bash
python scripts/patch_notebook.py --input private_runs/original.ipynb --output private_runs/own_lora24_ensemble.ipynb
```

补丁工具只解析源代码、不执行它。它会：

- 在`_GAUSSIAN_SOURCE`字符串中将检查点glob改为`**/own_lora24/final/COMPLETE.json`；
- 把成员名称`lora24_public`改为`lora24_own`，同时保留原成员的权重系数作为首轮对照；
- 清除所有旧输出，插入早期缺文件检查；
- 输出补丁文件哈希和修改摘要。

这是新的独立权重替换实验，文件格式兼容不表示权重相同。原固定融合系数未针对新权重优化，新的效果需验证。工具使用`ast.unparse`，源代码排版会变；不能只看文本diff行数判断变化。需要审阅语义和patch报告。

如果上一轮曾手动删除`lora24_public`，请从原始下载版本重新补丁；不要在已经删除成员的Notebook上继续拼修改。源码不再符合预期时工具直接失败，需人工审阅新结构，不得放宽成模糊字符串替换。

## 4. 其余成员仍然必须存在

早期marker检查包括：

```text
**/own_lora24/final/COMPLETE.json
**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json
**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json
**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json
**/tokattn_final.pt
**/tokattn_debl23_final.pt
```

这只是快速缺文件筛查，不覆盖所有base、encoders、UltraBERT、extra pools，也不校验这些第三方文件内容。请对照原Input逐项记录版本，并保留自己的训练checkpoint校验报告。不能拿本项目的COMPLETE来替代其他模型的COMPLETE。

## 5. 集成验收

1. `validate_checkpoint`和`check_parity`先通过；默认FP16契约应与原读取方式对应。
2. 上传补丁后的Notebook，明确命名`own-lora24-replacement-v1`等新实验。
3. 先样例test完整跑通，日志应出现`[lora24_own]`和anchor parity，不出现原缺失错误。
4. 确认末尾真正生成`submission.parquet`。
5. Save Version / Run All，执行Kaggle完整hidden rerun，记录真实结果。
6. 正式提交、发表或共享的最终动作按接收方用户授权执行；不要把交接文件当作对其他账户的无限授权。

原0.604只作为上游对照值。新模型、独立plain-head、缺成员降级版本分别命名和记录，不继承该分数。
