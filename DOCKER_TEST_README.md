# iMED-PE 提交镜像测试说明（给师兄）

## 镜像内容

`imedpe-submission:v4-final` — iMED 2026 挑战 Task 1（内窥镜跨相机位姿估计）的提交镜像。
方法：ALIKED + LightGlue 匹配 → 立体三角化 + PnP → 3D-3D Umeyama 精修 → Kalman+RTS 时序平滑。
本地成绩：train mean ATE 1.098 mm / test 1.301 mm（官方 baseline 2.163 / 2.508，-49%/-48%）。

## 导入镜像

```bash
gunzip -c imedpe_v4_image.tar.gz | docker load
```

## 运行方式（模拟评测环境）

评测协议：只读挂载 /input、网络禁用、20G 内存、10 分钟时限、1× RTX 4090。

```bash
# 准备输入：解压后的序列目录（每个序列一个子目录，含 K.txt + 4 路 frame_*.png）
# 例如 /path/to/data/test/<sequence_name>/...

mkdir -p /path/to/output

timeout 660 docker run --rm \
  --gpus all \
  --network=none \
  --memory=20g \
  -v /path/to/data/test:/input:ro \
  -v /path/to/output:/output \
  imedpe-submission:v4-final
```

## 需要记录的结果

1. **总耗时**（`Total: XXXs`，最后一行）— 这是关键：评测限时 10 分钟
2. **每条序列的耗时**（`runtime=XXs`）和 `registered=XX%`
3. 输出文件：`/path/to/output/<sequence>/pose_predictions.txt`

## 评测（有 GT 的话）

如果测试数据带 pose.txt（如官方 test split），可以对照验证精度：

```bash
# 在 official-imedpe 仓库里
python scripts/evaluate_ate.py \
  --data-root /path/to/data \
  --split test \
  --pred-root /path/to/output

# 注意：evaluate_ate 期望 <pred-root>/test/<seq>/pose.txt，
# 容器输出是 pose_predictions.txt，需先改名或软链
```

## 兼容性

- 镜像基于 CUDA 11.8（评测机同版本），4090 及更新 driver（12.x）向下兼容，可直接跑
- 权重已打包在镜像内，运行不需要网络

## 关注点

本地共享 H20 上 ~70s/序列；想知道 4090 独占时每序列几秒、19 条 test 能否在 10 分钟内跑完。
如果总时长 > 600s，告诉我，我这边有备选加速方案（代码已预留 frame_skip 开关）。
