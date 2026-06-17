# TXT Tree ISM Simplifier

这个目录提供 `tree_output.txt + datas/*.vti` 版本输入的 ISM 简化实现，不再依赖 `nodes.vtu / arcs.vtu / segmentation.vti`。

## 输入

- `-t, --tree`：树结构文本，格式为 `nodeId parentId scalar`
- `-s, --scalar-image`：原始体数据 `vti`
- `-o, --output`：输出前缀
- `-e, --epsilon-ism`：ISM 距离阈值，默认 `0.15`
- `-p, --epsilon-pos`：边界质心距离阈值，默认 `1.0`
- `-b, --epsilon-bbox`：包围盒相对差异阈值，默认 `0.10`
- `-n, --hist-bins`：直方图 bins 数，默认 `32`
- `-T, --tree-type`：树方向，`0=JT`，`1=ST`
- `-a, --scalar-array`：VTI 点数据里的标量数组名；不传时优先用 active scalars

## 设计

- 从 `tree_output.txt` 重建父子树和 branch
- 每个节点的区域定义为其 `subtreeVertices`，也就是当前节点及其所有后代节点
- 用该子树集合的包围盒生成局部二值 mask：mask 内只有属于 `subtreeVertices` 的点为 1
- 计算这个二值 mask 的边界距离场，作为 ISM 描述符
- 沿 branch 贪心比较 `distISM / distPos / distBbox`

需要注意：

- 筛选过程是沿单条 branch 做的，候选节点只来自当前 branch。
- 但单个节点的 ISM 描述符不是只取当前 branch 上的一小段，而是取该节点的整棵子树。
- 如果该节点下面挂着其他旁支，这些旁支会一起进入 `subtreeVertices`，从而影响二值 mask、边界距离场、质心和包围盒。
- 当前 TXT 版不会从 `SegmentationId` 取局部区域，也没有显式提取 `f(x)=f(v)` 的等值面；`scalar-image` 主要用于点坐标和标量数组读取，描述符几何来自子树点集本身。

## 输出

- `*_nodes.vtu`
- `*_arcs.vtu`
- `*_node_mask.vtu`
- `*_stats.txt`

## 编译

```bash
cd /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/idea_ism/support_txt
cmake -S . -B build
cmake --build build -j
```

## 示例

```bash
./build/treeTxtIsmSimplifier \
  -t /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/dataset_txt/results/case01_5x5x3/tree_output.txt \
  -s /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/dataset_txt/datas/case01_5x5x3.vti \
  -o /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/idea_ism/support_txt/out/case01 \
  -e 0.15 -p 1.0 -b 0.10 -T 0
```
