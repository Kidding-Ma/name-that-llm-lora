# 模型与文件契约

## 固定实现

- 基础模型：Qwen3-8B Base，官方原始配置应为 `model_type=qwen3, hidden_size=4096, num_hidden_layers=36`。不是8B Instruct，不是14B，不是两阶段公开adapter。
- 使用 `AutoModel` 而非生成式 LM head，截断为24层；`layer_types`存在时同步截断。
- LoRA：r16、alpha32、dropout0.05，作用于q/k/v/o/gate/up/down七种projection；base冻结，adapter为FP32 trainables。
- 文本：最多447 text tokens，无自动special tokens，再追加EOS。右padding；mask掉第一个token，剩余含EOS做mean pool。
- hidden_states索引 `[2,4,6,8,12,16,20,24]`；每个pool一个FP32 LayerNorm(4096)，拼为32768维。
- 两个18类线性头：plain(z)，center(z - batch_mean(z))。各自dropout0.1。训练loss为两个交叉熵之和。
- 一次microbatch为同一prompt的3个不同答案。6个microbatch累积为一次optimizer update。一个epoch覆盖所有fit样本，尾部补齐有记录。
- AdamW，adapter lr1e-4，head/norm lr1e-3，weight_decay0.01，clip norm1，10%warmup + cosine；默认3轮。
- FP16 CUDA autocast、FP32训练参数、GradScaler；非有限loss直接失败，overflow最多重放3次同一批/RNG。调度器只在成功更新后前进。

这些是当前独立实现的选择，不是“与原权重逐字节相同”的承诺。

## CV 与 final

`prepare.py`把unique prompt排序、seed42打乱后分为五份。每折验证prompt的所有答案完全排除在该折adapter训练之外。每折从官方base重新初始化。固定epoch3导出，不依据验证标签选checkpoint。

`final`在全部标注数据上训练，不产生held-out分数。`evaluate_oof.py`只合并五折plain head。训练期centered loss存在，不等于其在未知prompt分组上已验证有效。进一步验证centered head时，要用隐藏prompt身份不可见的内容聚类，并测单例、部分prompt和混合组；真实prompt分组只能单列oracle诊断。

## 输出结构

```text
own_lora24/final/
  adapter/adapter_config.json
  adapter/adapter_model.safetensors
  tokenizer/...
  base_config/config.json
  heads.safetensors
  readouts.npz
  anchor_features.npz
  validation_features.npz
  training_meta.json
  training_report.json
  COMPLETE.json
```

`heads.safetensors`必须包含`norms.0.weight`至`norms.7.bias`。`readouts.npz`必须包含：

| key | shape / meaning |
|---|---|
| classes | sorted exact18标签，见ntl/common.py |
| pool_layers | [2,4,6,8,12,16,20,24] |
| plain_coef / center_coef | (18,32768) |
| plain_intercept / center_intercept | (18,) |
| train_sha256 | 训练文件字节哈希 |

`anchor_features.npz`为保存前first3训练行特征，用于重新加载后比较；包含ids/classes/features。这是序列化一致性检查，不是泛化准确率。

`validation_features.npz`保存该折held-out行的ids/rows/classes/features。final的validation bank为空。它们属于运行产物，不能上传此GitHub仓库。

## 校验层次

1. Python/CPU测试：只证明分折、采样、补丁等已覆盖逻辑正确。
2. validate_checkpoint：文件、manifest哈希、readout和LayerNorm形状/顺序。
3. check_parity：GPU实际加载base+adapter，复现anchors，平均相对误差<0.02。
4. OOF：held-out性能证据。
5. Kaggle完整hidden rerun：最终执行及计分证据。

前一层通过不能替代后一层。更换任何模型文件都必须重新做GPU parity。
