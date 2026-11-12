# LongCat LoRA 训练参考实现

[English](TRAINING.md) · [推理与示例](README_zh-CN.md)

本实现支持单张 CUDA 显卡上的配对图像 LoRA 微调：先检查数据并缓存冻结编码器/VAE 的结果，再训练 LongCat Transformer 中的 LoRA。无需原研究项目目录、DeepSpeed、W&B 或训练时调用外部 VLM API。

演示图片使用的是历史 `stage2/10000` 权重。这里提供可移植的训练参考实现，保留相应任务提示词、LoRA 层和主要优化设置，但不声称逐像素重训出历史权重或完整复现论文所有结果。

## 1. 安装和数据格式

```bash
python -m pip install -r requirements-train.txt
```

训练清单为 JSONL，每行一组配对数据：

```json
{"input":"marked/example.png","target":"clean/example.png","task":"arrow_edit"}
```

路径相对于清单文件解析。`input` 是完整的带框、箭头或文字的输入，`target` 是已经执行修改且去掉标记的监督目标。`task` 可选 `text_to_image`、`text_edit`、`arrow_edit`、`box_edit`，对应推理时的通用任务提示词；每张图的具体编辑要求仍在图像中。读取器也兼容旧研究清单的 `ref_img_path`（输入）、`img_path`（目标）和 `prompt`。

第一阶段准备 Type I 文字图片清单，第二阶段准备四类混合清单。配对图片应在几何位置上对齐。缓存工具将完整输入与目标缩放到统一训练尺寸，并记录原始尺寸和哈希；不要求单独的框坐标或 mask。默认尺寸为 512 × 512，也可使用 `--width` 与 `--height` 指定尺寸。

`training/examples/` 只复用了演示图来展示格式和测试流程，不是原训练集，也不能作为独立评测集或用来报告模型质量。

## 2. 预缓存

假设自己的清单位于 `data/stage1.jsonl`、`data/stage2.jsonl`：

```bash
python -m siphon.prepare_training_data \
  --manifest data/stage1.jsonl \
  --base-model meituan-longcat/LongCat-Image-Edit \
  --revision 7b54ef423aa7854be7861600024be5c56ab7875a \
  --cache-dir caches/stage1 --resolution 512

python -m siphon.prepare_training_data \
  --manifest data/stage2.jsonl \
  --base-model meituan-longcat/LongCat-Image-Edit \
  --revision 7b54ef423aa7854be7861600024be5c56ab7875a \
  --cache-dir caches/stage2 --resolution 512
```

离线运行时将基座换成本地 snapshot 目录，并加 `--local-files-only`。`--dry-run` 在不加载模型依赖的情况下检查输入和计划；`--limit` 限制烟测样本数。缓存目录包含逐例 safetensors、`manifest.jsonl` 和数据元信息。改变基座版本、配对数据、提示词或有效尺寸后，应重新生成缓存。

## 3. 两阶段训练

```bash
python -m siphon.train --config configs/stage1.json
python -m siphon.train --config configs/stage2.json
```

配置文件中的路径相对于配置文件位置解析；命令行覆盖的路径相对于当前工作目录解析。可用 `--base-model`、`--cache-manifest`、`--output-dir` 覆盖本地设置；`--dry-run` 检查训练请求而不加载模型。

| 配置 | 第一阶段 | 第二阶段 |
| --- | --- | --- |
| 数据 | Type I 文字图片配对 | Type I–IV 混合配对 |
| LoRA 初始化 | 新 adapter | 第一阶段 `checkpoints-2400` |
| 累计步数范围 | 0 → 2500 | 2400 → 10000 |
| 学习率 | `2e-4` | `1e-4` |
| warmup 更新数 | 100 | 250 |
| LoRA rank / alpha | 64 / 8 | 64 / 8 |
| 微批量 × 梯度累积 | 1 × 8 | 1 × 8 |

第一阶段每 200 步及最终步保存，因此虽然结束于 2500，仍会保留供第二阶段初始化的 `checkpoints-2400`。第二阶段从累计 2400 到 10000，共新增 7600 次优化器更新，不是额外训练 10000 步。单卡 `1 × 8` 的有效批量为 8，与历史物理批量 8 不保证数值完全一致。

基座被冻结，只训练 adapter。LoRA 同时注入 attention 和 FFN 投影层，对应本次公开的权重。训练采用 bf16、梯度检查点、AdamW、梯度累积、warmup 和梯度裁剪。`null_text_ratio=0.1` 使用预缓存的空提示词条件；输入图中标注仍然存在。

## 4. 初始化、断点恢复和导出

`init_lora`/`--init-lora` 只初始化 adapter 权重，并重新建立优化器和学习率计划；`initial_step` 定义累计步数标签。跨阶段训练使用这个方式。

真正中断续训用 `--resume`，恢复 checkpoint 中的优化器、scheduler、随机数状态及步数：

```bash
python -m siphon.train --config configs/stage2.json \
  --resume outputs/training/stage2/checkpoints-4000
```

应保持相同基座、缓存数据及兼容配置。`--resume` 优先于第二阶段的初始化设置。

每个 checkpoint 含标准 PEFT adapter，以及单独的 `training_state/`。推理只需 `adapter_config.json` 和 `adapter_model.safetensors`；发布时附模型说明，不必把优化器状态或缓存一起上传。第二阶段默认最终权重位于 `outputs/training/stage2/checkpoints-10000`。

```bash
python -m siphon.inference \
  --lora outputs/training/stage2/checkpoints-10000 \
  --examples examples/examples.json --case box_double_bass \
  --output-dir outputs/after_training
```

## 最小烟测

以下只验证缓存、短训练和导出，不会训练出可用编辑模型：

```bash
python -m siphon.prepare_training_data \
  --manifest training/examples/stage1.jsonl \
  --revision 7b54ef423aa7854be7861600024be5c56ab7875a \
  --cache-dir caches/smoke --resolution 512 --limit 1

python -m siphon.train --config configs/stage1.json \
  --cache-manifest caches/smoke/manifest.jsonl \
  --output-dir outputs/training/smoke \
  --max-steps 2 --batch-size 1 --gradient-accumulation-steps 1
```

会导出 `outputs/training/smoke/checkpoints-2`，可用前面的推理命令加载。两步烟测不能代替示例图使用的历史 `10000` adapter。

## 历史配置与发布范围

历史配置和日志将本次示例 adapter 关联到：第一阶段 4000 条 Type I、512 分辨率、选用 2400 步；第二阶段 20000 条混合数据、训练至累计 10000 步。旧实现每步在线编码 VAE/文本特征，使用多种通用提示词，并在 attention 与 FFN 上训练 LoRA。

新缓存实现使用确定性的 VAE mode latent，并按缓存尺寸编码标注图。旧训练每步对 VAE latent 采样，视觉文本编码器也使用更小的输入图。预缓存、确定性 latent、微批量梯度累积及实现修正都使它不能被称为历史 checkpoint 的精确重训。

本次没有发布原完整训练数据、VLM/teacher 数据合成引擎、论文规模的完整缓存数据流程或 Qwen 基座训练器。数据准备工具读取已有输入/目标配对，不生成 ground truth。实际完成的测试范围见 [验证记录](VALIDATION.md)。
