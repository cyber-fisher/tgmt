# TXT Tree IoU Simplifier

这个实现专门支持 `tree_output.txt + .vti` 这种输入形式，不再要求 `node.vtu` / `arc.vtu`。

## 输入

- `-t`：树结构文本，格式为 `nodeId parentId scalar`
- `-s`：原始标量场 `vti`
- `-o`：输出前缀
- `-e`：简化阈值，范围 `[0, 1]`
- `-T`：树类型，`0=JT`，`1=ST`

## 核心思路

- 从 `txt` 重建父子树
- 对同一条 branch 上的两个节点，先在该 branch 的局部范围内按节点标量构造子水平集体积
- 再计算两个局部体积的 IoU
- 用 `distance = 1 - IoU` 做保点判定

这套定义不依赖 TTK 导出的 `RegionSize/SegmentationId`，因此适合当前 `dataset_txt/case01` 的输入形式。

## 输出

- `*_nodes.vtu`
- `*_arcs.vtu`
- `*_node_mask.vtu`
- `*_stats.txt`

## 示例

```bash
./build/treeTxtIoUSimplifier \
  -t dataset_gyc/gyc/dataset_txt/case01/tree_output.txt \
  -s dataset_gyc/gyc/dataset_txt/case01/case01_5x5x3.vti \
  -o idea_iou/case01_iou/output \
  -e 0.25 \
  -T 0
```
