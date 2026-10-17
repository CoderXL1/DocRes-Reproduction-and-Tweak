# 去畸变坐标场平滑实验

本实验从 `134b614` 的 `inference.py::dewarping()` 复制需要修改的计算步骤，
在本目录内维护。根目录的 `inference.py` 保留普通推理流程和 3×3、15 次平滑的默认设置。

## 运行

从仓库根目录使用 `python -m` 启动，避免同名 `inference.py` 的导入歧义。
沿用根目录的依赖环境和 checkpoint，不需要重新训练。

```bash
# 迁移前的探索设置：20×20 核，扫描 0/1/3/5/10/15/30/60/100 次
python -m experiments.dewarp_smoothing.inference \
  --im_path input/189_origin.png --device mps

# 与原始后处理设置对照：3×3 核，15 次
python -m experiments.dewarp_smoothing.inference \
  --im_path input/189_origin.png --device mps \
  --kernel_sizes 3 --smooth_times 15

# 指定扫描参数和新的输出目录；CPU 环境可改用 --device cpu
python -m experiments.dewarp_smoothing.inference \
  --im_path input/189_origin.png --device mps \
  --kernel_sizes 3 5 20 --smooth_times 0 1 5 15 30 \
  --out_folder restored/dewarp_smoothing/comparison-01 --save_dtsprompt
```

`--im_path` 为必填参数，支持单张图片或目录，不选择默认输入目录。
本地输入可放在 Git 忽略的 `input/local/` 下，或显式指定其他位置。默认输出到
`restored/dewarp_smoothing/<UTC时间戳>/`，指定的输出目录必须尚不存在。
生成图片使用 PNG；每个输入文件有独立子目录，结果名包含核大小和平滑次数。
`restored/` 已被 Git 忽略，早期探索图片归档在 `restored/dewarp_smoothing/legacy/`。
通用研究假设与实验计划见 [研究笔记](../RESEARCH_NOTES.md)。

## 代码边界

- `predict_dewarping()`：复制并维护本实验的预处理与坐标预测，返回原图、prompt 和未平滑的坐标场。
- `render_dewarping()`：复制坐标场后执行指定的平滑与采样，不修改缓存的预测。
- `dewarping()`：组合上面两个函数，供需要单个设置的调用方使用，返回值结构与基线相同。
- `main()`：每张图只做一次模型预测，然后扫描参数。去畸变固定使用 FP32。
- 模型加载、prompt 生成、设备选择和保存工具从基线复用；实验不替换基线模块中的函数。

基线 `inference_one_im()` 内部仍调用基线函数。如果以后需要实验版 end2end，
在本目录显式编排各步骤，调用本实验的 `dewarping()`。

`run.json` 记录实际参数、输入清单、权重路径、设备、PyTorch 版本及 Git 状态。
权重路径不代表内容校验；未提交代码也不能仅凭 commit 完整复现，重要运行应先提交代码并固定权重。
现有探索图片缺少完整运行记录，不据此补写定量结论。

## 验证记录

2026-10-17 已完成真实 checkpoint 的 MPS 端到端验证。
环境为 Python 3.12.13、PyTorch 2.14.0，去畸变使用 FP32。

```bash
python -m experiments.dewarp_smoothing.inference \
  --im_path input/189_origin.png --device mps \
  --kernel_sizes 3 20 --smooth_times 0 15
```

- 四组均生成有效的 4096×3072 PNG，运行参数记录在 `run.json`。
- 独立加载相同的 DocRes 和 MBD 权重，原始推理的 3×3、15 次结果与实验输出逐像素一致。
- 再次运行实验预测，四组结果均与保存的输出逐像素一致；两组零次平滑结果也逐像素一致。
- 合成输入与模拟模型检查覆盖了缓存坐标场不被修改、每图只预测一次、输出目录隔离与参数校验。

这次验证确认入口与结果迁移正确；系统性消融、定量指标与最优参数仍待评估。

## 后续实验

每个实验新增自己的目录，复制确实需要调整的函数，继续复用未修改的工具。
涉及训练时在对应实验目录增加 `train.py`；涉及模型结构时增加 `model.py`，
由该实验的训练和推理入口共同导入。暂不引入通用实验框架。
