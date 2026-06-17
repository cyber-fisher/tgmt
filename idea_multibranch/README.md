# 多分支加权等值面简化器 (Multi-Branch Weighted Isosurface Simplifier)

## 概述

本工具实现了基于合并树（Merge Tree）的多分支加权等值面简化。与现有的逐分支独立简化方法不同，本方法在决定是否保留某个等值面时，综合考虑该标量值处所有活跃分支的信息，并根据每个分支的拓扑重要性和几何变化剧烈程度进行加权。

## 核心思想

在合并树中，一个等值面可能同时跨越多个拓扑分支。传统方法对每个分支独立简化，忽略了分支间的关联。本方法的改进在于：

- 对于**变化较小**的分支，给予较低权重（几何变化剧烈程度权重）
- 对于**拓扑上重要**的分支，给予较高权重（拓扑重要性权重）
- 综合加权后决定是否保留该等值面

## 算法流程

1. 解析合并树（txt格式）并构建树结构
2. 提取所有分支，计算每个分支的scalar范围和persistence
3. 确定主链（root到最大persistence叶节点的路径）
4. 沿主链进行贪心简化：
   - 在每个候选节点处，找出所有活跃分支
   - 计算每个分支的IoU距离
   - 计算混合权重并聚合距离
   - 若聚合距离 >= epsilon，保留该节点
5. 非主链分支使用独立IoU简化
6. 重建简化后的树结构并输出

## 权重公式

### 拓扑重要性权重

```
w_topo_i = persistence_i / Σ_j(persistence_j)
```

persistence越大的分支，在加权中占比越高。

### 几何变化剧烈程度权重

```
geo_i = |volSize(ref) - volSize(candidate)| / max(volSize(ref), volSize(candidate))
w_geo_i = geo_i / Σ_j(geo_j)
```

体积变化越大的分支，说明几何变化越剧烈，权重越高。

### 混合权重

```
w_i = alpha * w_topo_i + (1 - alpha) * w_geo_i
```

### 加权聚合距离

```
D(ref, candidate) = Σ_i(w_i * d_i)
```

其中 `d_i = 1 - IoU(volume_ref_i, volume_candidate_i)` 是第i个分支的IoU距离。

## 编译

```bash
cd idea_multibranch
mkdir build && cd build
cmake ..
make
```

依赖：VTK（与项目其他模块相同）

## 使用方法

```bash
./treeTxtMultiBranchSimplifier -t <tree.txt> -s <image.vti> \
    [-o <output_prefix>] [-e <eps>] [-a <alpha>] [-T <0|1>]
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-t` | 合并树txt文件路径（必需） | - |
| `-s` | VTI图像文件路径（必需） | - |
| `-o` | 输出文件前缀 | `idea_multibranch/result/output` |
| `-e` | epsilon阈值，控制简化程度 [0,1] | 0.25 |
| `-a` | alpha权重混合参数 [0,1] | 0.5 |
| `-T` | 树类型：0=Join Tree, 1=Split Tree | 0 |

### alpha参数含义

- `alpha = 0`：纯几何权重，只考虑几何变化剧烈程度
- `alpha = 0.5`：拓扑和几何各占一半（推荐）
- `alpha = 1`：纯拓扑权重，只考虑分支的persistence

## 输入格式

### tree.txt

```
# Augmented Merge Tree
# 格式: [顶点编号] [父节点] [标量值]
0 1 0.277915
1 44 0.277915
2 7 0.280084
...
```

- 每行一个节点：`nodeId parentId scalar`
- `#` 开头的行为注释
- root节点的parentId等于自身或为-1

### image.vti

VTK ImageData格式，提供节点的三维坐标信息。

## 输出文件

| 文件 | 说明 |
|------|------|
| `*_nodes.vtu` | 保留的节点（含Scalar, CriticalType等属性） |
| `*_arcs.vtu` | 简化后的弧（含RegionSize, RegionSpan等） |
| `*_node_mask.vtu` | 所有节点的IsKept标记 |
| `*_stats.txt` | 统计信息（节点数、分支数、主链长度等） |

## 示例

```bash
# 混合权重简化
./treeTxtMultiBranchSimplifier -t tree_output.txt -s data.vti -e 0.25 -a 0.5

# 纯拓扑权重（更激进的简化）
./treeTxtMultiBranchSimplifier -t tree_output.txt -s data.vti -e 0.25 -a 1.0

# 纯几何权重（更保守的简化）
./treeTxtMultiBranchSimplifier -t tree_output.txt -s data.vti -e 0.25 -a 0.0
```

## 与现有方法的关系

本方法是对现有三种基础策略（IoU、ISM、ActiveGraph）的扩展层。当前实现使用IoU作为per-branch距离度量，后续可扩展支持ISM和ActiveGraph。

当树只有单个分支时，本方法退化为标准的逐分支IoU简化，结果与 `idea_iou/TreeTxtIoUSimplifier` 一致。
