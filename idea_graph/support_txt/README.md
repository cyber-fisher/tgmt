# TXT Tree Active Graph Simplifier

这个目录提供 `tree_output.txt + datas/*.vti` 版本输入的 Active Graph 简化实现，不再依赖
`nodes.vtu / arcs.vtu / segmentation.vti`。

## 输入

- `-t, --tree`：树结构文本，格式为 `nodeId parentId scalar`
- `-s, --scalar-image`：原始体数据 `vti`
- `-o, --output`：输出前缀
- `-j, --epsilon-j`：Jaccard 距离阈值，默认 `0.15`
- `-p, --epsilon-pos`：边界质心距离阈值，默认 `0.10`
- `-b, --epsilon-bbox`：包围盒相对差异阈值，默认 `0.10`
- `-T, --tree-type`：树方向，`0=JT`，`1=ST`
- `-a, --scalar-array`：VTI 点数据里的标量数组名；不传时优先用 active scalars

## 设计

- 从 `tree_output.txt` 重建父子树和 branch
- 每个节点的局部区域定义为其子树覆盖的顶点集合
- 在该局部区域包围盒中提取与当前标量阈值相交的 active edges
- 沿 branch 贪心比较 `distJ / distPos / distBbox`

## 输出

- `*_simplified_nodes.vtu`
- `*_simplified_arcs.vtu`
- `*_stats.csv`

## 编译

```bash
cd /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/idea_graph/support_txt
cmake -S . -B build
cmake --build build -j
```

## case17 示例

```bash
./build/treeTxtActiveGraphSimplifier \
  -t /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/dataset_txt/results/case17_12x7x5/tree_output.txt \
  -s /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/dataset_txt/datas/case17_12x7x5.vti \
  -o /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/idea_graph/support_txt/out/case17 \
  -a u -j 0.15 -p 0.10 -b 0.10 -T 0
```
