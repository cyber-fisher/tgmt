# GNN 活性子图向量化方法简述

## 1. 问题

原来的活性子图比较方法主要直接比较边集重合度，例如 Jaccard distance。

这种方法实现简单，但对局部扰动比较敏感：

- 少量边缺失会导致相似度明显下降。
- 局部位置轻微偏移会被认为差异较大。
- 两个结构形态相似但边编号不完全一致的子图，可能被误判为不相似。

因此，现在改为先把每个活性子图编码成一个向量，再在向量空间中比较相似性。

核心目标是：

```text
不直接比较边是否完全重合，
而是比较局部活性结构是否相似。
```

## 2. 输入数据

当前使用的是 `dataset_txt` 格式：

```text
dataset_txt/
  datas/<case>.vti
  results/<case>/tree_output.txt
```

其中：

- `.vti` 是原始规则网格标量场。
- `tree_output.txt` 是 merge tree 的文本表示。

`tree_output.txt` 每一行格式为：

```text
nodeId parentId scalar
```

也就是：

- 当前节点编号。
- 父节点编号。
- 当前节点对应的标量值。

## 3. 活性子图提取

对于 merge tree 中的每个节点 `n`：

```text
t = Scalar(n)
```

将该节点的标量值作为阈值，在原始 VTI 规则网格中提取跨越阈值 `t` 的边。

一条网格边 `(u, v)` 如果满足：

```text
scalar(u) <= t < scalar(v)
或
scalar(v) <= t < scalar(u)
```

就认为它是活性边。

由于 `dataset_txt` 中没有 `SegmentationId`，当前局部区域使用 merge tree 节点的子树区域定义：

```text
节点 n 的局部区域 = 以 n 为根的子树覆盖的 VTI 顶点集合
```

最后得到：

```text
节点 n -> 活性子图 G_n
```

## 4. GNN 编码

每个活性子图会输入一个 2 层 GCN 编码器。

每个活性顶点的特征包括：

- 归一化标量值。
- 与当前阈值的距离。
- 中心化、尺度化后的空间坐标。
- 活性子图内局部度数。
- 节点所属区域的简单编码。

GCN 结构：

```text
节点特征
  -> GCN layer 1 + ReLU
  -> GCN layer 2 + ReLU
  -> mean pooling + max pooling
  -> 拼接子图规模信息
  -> 得到固定长度向量 z(n)
```

这样每个 merge tree 节点最终都有一个向量表示：

```text
n -> z(n)
```

## 5. 相似性度量

两个节点的相似性通过 GNN 向量的 cosine distance 计算：

```text
distGNN(i, j) = 1 - cosine(z_i, z_j)
```

含义：

- `distGNN` 越小，两个节点对应的活性子图越相似。
- `distGNN` 越大，说明局部活性结构变化越明显。

在简化时：

```text
distGNN >= epsilon -> 保留该节点
distGNN <  epsilon -> 跳过该节点
```

## 6. GCN 权重训练

最开始的 GCN 权重是随机初始化的，只能作为 baseline。

现在加入了自监督对比学习训练。

训练方式：

1. 对同一个活性子图做两次轻微扰动。
2. 这两个扰动版本作为正样本。
3. batch 中其他子图作为负样本。
4. 使用 InfoNCE / NT-Xent loss 训练 GCN。

正样本构造：

```text
G_n^1 = augment(G_n)
G_n^2 = augment(G_n)
```

其中 `G_n^1` 和 `G_n^2` 来自同一个原始子图，所以应该在向量空间中接近。

扰动方式包括：

- 随机删除少量边。
- 随机遮挡部分节点特征。
- 给标量特征加轻微噪声。
- 给空间坐标特征加轻微噪声。

训练目标：

```text
同一子图的扰动版本更接近，
不同子图的向量更远。
```

## 7. 当前训练结果

当前使用 `dataset_txt` 中 20 个 case 共同训练。

训练样本：

```text
共 5531 个非空活性子图
```

训练到 200 轮后：

```text
loss    = 2.2475
pos_sim = 0.8476
neg_sim = 0.3131
```

说明：

- 正样本仍然保持较高相似性。
- 负样本已经被明显拉开。
- GCN 学到了比随机初始化更有区分度的表示。

## 8. case17 测试结果

使用训练好的模型测试：

```text
case17_12x7x5
```

数据规模：

```text
nodes: 420
arcs: 419
topological chains: 43
```

训练后 `distGNN` 分布：

```text
p50 = 0.0292
p75 = 0.0691
p90 = 0.1428
p95 = 0.4848
```

因此当前选取：

```text
epsilon = 0.15
```

这个阈值接近 p90，表示保留距离最高约 10% 的候选节点。

结果：

```text
候选比较数: 376
keep = 1: 36
keep = 0: 340
```

最终导出的 ParaView 结果中：

```text
kept nodes: 80
GNN 距离触发保留的节点: 36
```

其中 `kept nodes` 包括：

- 拓扑链端点。
- 结构节点。
- GNN 判断变化明显的节点。

## 9. 可视化验证

导出了 ParaView 可视化文件：

```text
case17_gnn_trained_eps015_paraview_volume_for_contour.vti
case17_gnn_trained_eps015_paraview_kept_nodes.vtu
case17_gnn_trained_eps015_paraview_kept_active_edges.vtu
```

其中：

- `.vti` 是原始体数据加上 GNN 结果数组，可以用于等值面 Contour。
- `kept_nodes.vtu` 显示保留下来的 merge tree 节点。
- `kept_active_edges.vtu` 显示 GNN 判断变化明显节点对应的活性边。

ParaView 中使用：

```text
打开 volume_for_contour.vti
Contour By: u
叠加 kept_nodes.vtu 和 kept_active_edges.vtu
```

## 10. 方法总结

当前方法可以概括为：

```text
merge tree 节点
  -> 提取局部活性子图
  -> GCN 编码成向量
  -> cosine distance 比较
  -> 根据阈值决定是否保留
```

相比直接比较边集，该方法的优势是：

- 对轻微局部扰动更稳健。
- 能编码局部结构和空间形状信息。
- 可以通过自监督训练提升区分能力。
- 结果可以导出到 ParaView 中检查。

当前仍然是实验版本，后续可以继续改进：

- 调整阈值 `epsilon`。
- 比较不同 case 上的泛化效果。
- 加入更严格的负样本选择。
- 尝试 GraphSAGE / GAT 等更强的图编码器。
- 与原始 Jaccard 方法做定量对比。

