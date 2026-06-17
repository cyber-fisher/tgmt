# GNN 活性子图向量化实现说明

本文档对应 `GNN` 目录下最新版代码，主要说明两个脚本：

```text
GNN/vectorize_active_subgraphs.py
GNN/export_paraview_results.py
```

第一个脚本负责把每个 merge tree 节点对应的活性子图编码成 GNN 向量，并输出节点间距离。第二个脚本负责把 GNN 结果导出成 ParaView 可视化用的 VTK 文件。

## 1. 实现目标

原始方法直接比较两个活性子图的边集，容易受到局部偏移、轻微扰动、少量边缺失影响。

当前 GNN 方法改为：

```text
merge tree 节点
    -> 提取局部活性子图
    -> 2 层 GCN 编码
    -> 固定长度向量
    -> cosine distance 比较
```

距离越小，表示两个节点对应的活性子图越相似；距离越大，表示局部结构变化越明显。

当前版本是**未训练的确定性随机 GCN baseline**，主要用于先跑通管线和观察距离分布。

## 2. 输入数据

`vectorize_active_subgraphs.py` 需要三个输入：

```bash
-n / --nodes          merge tree nodes.vtu
-a / --arcs           merge tree arcs.vtu
-s / --segmentation   原始体数据 .vti
```

要求字段：

| 文件 | 必要字段 |
|---|---|
| nodes.vtu | `NodeId`, `Scalar` |
| arcs.vtu | `downNodeId`, `upNodeId`, `SegmentationId` |
| data.vti | 标量场，默认 `v`；以及 `SegmentationId` |

标量场名称可以通过 `--scalar` 指定。

## 3. 活性子图提取

对每个 merge tree 节点 `n`：

```text
t = Scalar(n)
```

然后在原始 VTI 规则网格上检查 6 邻接边。若一条边 `(u, v)` 跨越阈值 `t`，则认为它是活性边：

```text
scalar(u) <= t < scalar(v)
或
scalar(v) <= t < scalar(u)
```

同时使用当前节点相邻 arc 的 `SegmentationId` 做局部限制，只保留和相关 segmentation 有关的活性边。

得到的活性边和活性顶点组成当前节点的活性子图。

## 4. GNN 编码

每个活性顶点会构造一组简单特征，包括：

- 标量值归一化结果。
- 与阈值的距离。
- 中心化、尺度化后的空间坐标。
- 活性子图内局部度数。
- `SegmentationId` 的 sin/cos 编码。

编码器结构：

```text
节点特征
  -> GCN layer 1 + ReLU
  -> GCN layer 2 + ReLU
  -> mean pooling + max pooling
  -> 拼接 log(活性点数), log(活性边数)
  -> L2 normalize
```

如果 hidden dimension 为 `H`，最终向量维度是：

```text
2 * H + 2
```

例如：

```text
--hidden 32  -> 66 维
--hidden 64  -> 130 维
```

实现上使用边表消息传递，没有构造稠密邻接矩阵，因此可以处理较大的活性子图。

## 5. GNN 距离和阈值

两个节点的距离定义为：

```text
distGNN = 1 - cosine(z_ref, z_candidate)
```

含义：

| distGNN | 含义 |
|---:|---|
| 越接近 0 | 两个活性子图越相似 |
| 越大 | 局部结构差异越明显 |
| 大于等于 `--epsilon` | 当前候选节点被标记为保留 |

注意：GNN 距离的数值范围和 Jaccard 不一样，不能直接沿用 Jaccard 的阈值。

当前 tangaroa 数据上，使用 `--hidden 32` 时，推荐初始阈值：

```text
--epsilon 0.0025
```

这组数据中该阈值大约保留距离最高的一小部分候选节点。

## 6. 向量化脚本用法

示例：

```bash
python3 GNN/vectorize_active_subgraphs.py \
  -n /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_nodes.vtu \
  -a /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_arcs.vtu \
  -s /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_2.vti \
  -o GNN/out/tangaroa_gnn \
  --hidden 32 \
  --epsilon 0.0025
```

输出：

```text
GNN/out/tangaroa_gnn_embeddings.csv
GNN/out/tangaroa_gnn_chain_distances.csv
```

### 6.1 embeddings.csv

每一行是一个 merge tree 节点的 GNN 向量：

```csv
nodeId,z0,z1,z2,...
0,0.0411,0,0.0344,...
```

其中：

| 字段 | 含义 |
|---|---|
| `nodeId` | merge tree 节点 id |
| `z0, z1, ...` | 该节点活性子图的 GNN embedding |

### 6.2 chain_distances.csv

沿拓扑链比较参考节点和候选节点：

```csv
chainId,refNodeId,candidateNodeId,distGNN,keep
10,10,182,0.004014665198,1
```

字段含义：

| 字段 | 含义 |
|---|---|
| `chainId` | 拓扑链 id |
| `refNodeId` | 当前参考节点 |
| `candidateNodeId` | 当前候选节点 |
| `distGNN` | 两个节点向量之间的 GNN distance |
| `keep` | 是否超过阈值并保留 |

## 7. ParaView 导出

`export_paraview_results.py` 会读取 `chain_distances.csv`，根据 `keep=1` 的结果导出 VTK 文件。

示例：

```bash
python3 GNN/export_paraview_results.py \
  -n /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_nodes.vtu \
  -a /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_arcs.vtu \
  -s /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_2.vti \
  -d GNN/out/tangaroa_gnn_chain_distances.csv \
  -o GNN/out/tangaroa_gnn_paraview \
  --scalar v
```

输出：

```text
GNN/out/tangaroa_gnn_paraview_volume_for_contour.vti
GNN/out/tangaroa_gnn_paraview_kept_nodes.vtu
GNN/out/tangaroa_gnn_paraview_kept_active_edges.vtu
GNN/out/tangaroa_gnn_paraview_contour_values.csv
```

### 7.1 volume_for_contour.vti

这是最重要的 ParaView 文件。

它是 `vtkImageData`，保留了原始体数据，因此可以直接在 ParaView 中做 Contour 等值面。

使用方式：

1. 在 ParaView 中打开 `*_volume_for_contour.vti`。
2. 在 `Pipeline Browser` 中选中这个 `.vti` 数据源。
3. 点击 `Apply`。
4. 使用 `Contour`。
5. `Contour By` 选择原始标量场，例如 `POINTS: v`。

注意：不要选中 `.vtu` 点线文件来做等值面。`.vtu` 辅助文件是点或线数据，Contour 按钮可能会置灰。

该 `.vti` 中额外加入了三个数组：

| 数组 | 含义 |
|---|---|
| `GNNKeptActiveVertex` | 是否属于任意保留节点的活性子图，0/1 |
| `GNNKeptActiveNodeCount` | 被多少个保留节点的活性子图覆盖 |
| `GNNKeptMaxDistGNN` | 覆盖该点的保留节点中最大的 `distGNN` |

### 7.2 kept_nodes.vtu

保留节点点云，用来叠加显示 merge tree 中保留下来的节点。

包含字段：

```text
NodeId
Scalar
GNNKeepReason
distGNN
```

其中：

| 字段 | 含义 |
|---|---|
| `GNNKeepReason = 0` | 拓扑链端点或结构节点保留 |
| `GNNKeepReason = 1` | 因 GNN 距离超过阈值保留 |

### 7.3 kept_active_edges.vtu

`keep=1` 的候选节点对应的活性边线段。

这个文件适合在 ParaView 中作为线框叠加显示，用来观察 GNN 认为变化明显的位置。

### 7.4 contour_values.csv

记录保留节点对应的 scalar 值：

```csv
nodeId,scalar
```

可以从这里挑选 scalar 值，在 ParaView 的 Contour filter 中手动输入等值面值。

## 8. 当前 tangaroa 输出概况

使用：

```text
--hidden 32
--epsilon 0.0025
```

当前结果：

```text
merge tree nodes: 593
merge tree arcs: 592
topological chains: 149
embedding dimension: 66
candidate comparisons: 443
keep = 1: 22
```

导出的 ParaView 文件中：

```text
kept nodes: 172
active points marked in volume: 558562
active edges for GNN-kept candidates: 576
```

其中 `kept nodes` 包括拓扑链端点/结构节点，以及 `keep=1` 的 GNN 候选节点。

## 9. 当前限制

当前版本仍是 baseline：

1. GCN 没有训练，权重由 `--seed` 固定生成。
2. `--epsilon` 需要按数据集距离分布调参。
3. Python 脚本主要用于实验和可视化，尚未直接替换 C++ 简化器的完整输出流程。
4. 后续可以加入对比学习、自监督训练，或者替换为 GraphSAGE / GAT。

