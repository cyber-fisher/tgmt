# ISM Merge Tree Simplifier

基于 ISM (Intersection Similarity Measure) 的合并树拓扑简化工具，属于 TTK (Topology Tool Kit) 生态的一部分。

## 用法

```bash
./ism_main <nodes.vtu> <arcs.vtu> <segmentation.vti> [output_prefix]
```

参数：
- `nodes.vtu` — 合并树节点文件，包含关键点及其属性
- `arcs.vtu` — 合并树弧文件，描述节点间连接关系
- `segmentation.vti` — 对应的体分割数据
- `output_prefix` — 输出文件前缀（可选，默认 `ism_simplified`）

## 整体流程

```
加载 VTU/VTI 数据
       ↓
识别结构节点（root / leaf / 分支点）
       ↓
提取拓扑链（结构节点之间的路径）
       ↓
沿链贪心简化：为每个节点构建 ISM 描述符，与前一保留节点比较
       ↓
输出简化结果（节点 VTU、弧 VTU、统计 CSV）
```

---

## 文件详解

### 1. `ism_main.cpp` — 程序入口

最简的命令行入口，负责解析参数并调用 `IsmMergeTreeSimplifier::run()`。

```cpp
int main(int argc, char **argv)
```

- 解析命令行参数，填充 `IsmSimplificationParams` 结构体
- 创建 `IsmMergeTreeSimplifier` 实例并调用 `run()`
- 错误码：`0` 成功，`-1` 加载失败，`-2` 链构建失败，`-3` 简化失败，`-4` 输出失败

---

### 2. `IsmMergeTreeSimplifier.h / .cpp` — 主控制器

#### 数据结构

```cpp
struct TreeNode {
  int nodeId;            // 节点唯一标识
  double scalar;         // 标量值（临界值）
  int criticalType;      // 临界点类型：0=鞍点, 1=极大值, -1=极小值
  int vertexId;          // 对应体数据中的顶点 ID
  double regionSize;     // 区域大小
  double regionSpan;     // 区域跨度
  std::array<double, 3> position;  // 空间坐标
};

struct TreeArc {
  int arcId;             // 弧唯一标识
  int segmentationId;    // 对应分割区域 ID
  int downNodeId;        // 下游节点（标量较小）
  int upNodeId;          // 上游节点（标量较大）
  double regionSize;     // 区域大小
  double regionSpan;     // 区域跨度
};

struct TopologicalChain {
  std::vector<int> nodeIds;  // 链上节点序列
  std::vector<int> arcIds;   // 链上弧序列
};

struct IsmDescriptor {
  int nodeId;                            // 节点 ID
  std::array<int, 6> extent;             // 局部块范围 [xmin, xmax, ymin, ymax, zmin, zmax]
  std::vector<float> distanceField;      // 距离场（到边界的距离）
  std::vector<unsigned char> boundaryMask; // 边界掩码
  std::array<double, 3> centroid;        // 边界质心
  double bboxDiagonal;                   // 包围盒对角线长度
};

struct IsmSimplificationParams {
  std::string treeNodesPath;       // 节点文件路径
  std::string treeArcsPath;        // 弧文件路径
  std::string segmentationPath;    // 分割体数据路径
  std::string scalarFieldName;     // 标量场名称（默认 "v"）
  std::string outputPrefix;        // 输出前缀（默认 "ism_simplified"）
  double epsilonIsm;               // ISM 距离阈值（默认 0.15）
  double epsilonPos;               // 位置距离阈值（默认 0.10）
  double epsilonBbox;              // 包围盒差异阈值（默认 0.10）
  int histogramBins;               // 直方图分箱数（默认 32）
  int paddingVoxels;               // 局部块 padding（默认 3）
  int maxBlockEdge;                // 局部块最大边长（默认 64）
  bool writeDebugVolumes;          // 是否输出调试体数据
};
```

#### 主要方法

| 方法 | 说明 |
|------|------|
| `run(params)` | 主入口，串联四个阶段 |
| `loadData(params)` | 读取 VTU 节点/弧文件和 VTI 分割体数据 |
| `buildTopologicalChains()` | 识别结构节点，提取拓扑链 |
| `simplify()` | 沿链贪心简化，为每个候选节点构建 ISM 描述符并比较 |
| `writeOutputs(params)` | 输出简化后的节点、弧和统计信息 |
| `isStructuralNode(node)` | 判断是否为结构节点（`criticalType != 0`） |

#### 简化算法 (`simplify`)

```
对每条拓扑链：
  保留第一个节点（链起点）
  对链中间的每个候选节点：
    1. 构建候选节点的 ISM 描述符
    2. 构建（或复用缓存的）参考节点描述符
    3. 计算 distIsm、distPos、distBbox 三个指标
    4. 若任一指标超过阈值 → 保留该节点，将其设为新参考点
    5. 否则 → 删除（跳过）
  保留最后一个节点（链终点）
```

---

### 3. `IsmFieldBuilder.h / .cpp` — ISM 描述符构建

负责为单个节点构建完整的 ISM 描述符。

#### 核心方法

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `buildDescriptor` | nodeId, nodes, arcs, imageData | `unique_ptr<IsmDescriptor>` | 总入口 |
| `getLocalSegmentationIds` | nodeId, nodes, arcs | `vector<int>` | 查找节点关联弧的分割 ID |
| `extractLocalBlock` | imageData, segIds, padding | extent, scalarValues, blockSegIds | 裁剪局部体数据 |
| `buildBoundaryMask` | scalarValues, extent, threshold | boundaryMask | 6-连通边界检测 |
| `buildDistanceField` | boundaryMask, extent | distanceField | BFS 距离变换 |
| `computeCentroid` | boundaryMask, extent | `array<double,3>` | 边界质心 |
| `computeBboxDiagonal` | extent | `double` | 包围盒对角线长度 |

#### 构建流程

```
buildDescriptor(nodeId)
  │
  ├─ getLocalSegmentationIds(nodeId)
  │     遍历所有弧，找出 downNodeId 或 upNodeId 等于 nodeId 的弧
  │     收集这些弧的 segmentationId → 去重
  │
  ├─ extractLocalBlock(imageData, segIds, padding=3)
  │     遍历整个体数据，找到所有 segId 在目标集合中的体素
  │     计算这些体素的包围盒（加 padding）
  │     裁剪出子体积的标量值和分割 ID
  │
  ├─ buildBoundaryMask(scalarValues, extent, node->scalar)
  │     以节点标量值为阈值：scalar <= threshold → inside(1), 否则 outside(0)
  │     对每个体素检查 6 个邻居，若邻居值不同则标记为边界
  │
  ├─ buildDistanceField(boundaryMask, extent)
  │     边界体素距离初始化为 0
  │     BFS 从边界向外扩展，每步 +1，得到每个体素到边界的曼哈顿距离
  │
  ├─ computeCentroid(boundaryMask, extent)
  │     对所有边界体素取坐标的算术平均
  │
  └─ computeBboxDiagonal(extent)
        sqrt(dx² + dy² + dz²)
```

---

### 4. `SimilarityMetrics.h / .cpp` — 相似度度量

基于信息论的描述符比较。

#### 核心方法

| 方法 | 说明 |
|------|------|
| `computeDistISM(descA, descB, numBins)` | 计算 ISM 距离 = `1 - NMI` |
| `computeHistogram(field, numBins, minVal, maxVal)` | 将连续场量化为直方图并归一化 |
| `computeEntropy(histogram)` | 计算香农熵 `H = -Σ p·log₂(p)` |
| `computeMutualInformation(fieldA, fieldB, ...)` | 构建联合直方图，计算互信息 |

#### 计算流程

```
computeDistISM(descA, descB, numBins=32)
  │
  ├─ 对 A、B 的距离场分别构建直方图（32 bins，归一化）
  ├─ 计算 H(A)、H(B) — 各自的香农熵
  ├─ 构建联合直方图 jointHist[32×32]
  │     对每对 (fieldA[i], fieldB[i]) 量化到 (binA, binB)
  ├─ 计算互信息 MI = Σ p(a,b) · log₂(p(a,b) / (p(a)·p(b)))
  ├─ NMI = 2·MI / (H(A) + H(B))
  └─ 返回 DistISM = 1 - NMI
```

NMI 取值 [0, 1]，越大表示越相似。DistISM 取值 [0, 1]，越大表示越不相似。

---

### 5. `MergeTreeIO.h / .cpp` — 数据读写

处理 VTK 文件格式的读写。

#### 核心方法

| 方法 | 说明 |
|------|------|
| `readNodesFromVTU(grid)` | 从 VTU 读取节点，需要 `NodeId`、`Scalar`、`CriticalType`、`VertexId`、`RegionSize`、`RegionSpan` 字段 |
| `readArcsFromVTU(grid)` | 从 VTU 读取弧，需要 `downNodeId`、`upNodeId`、`SegmentationId`、`RegionSize`、`RegionSpan` 字段 |
| `writeSimplifiedNodes(originalGrid, keptNodeIds, path)` | 输出简化节点，添加 `KeepNode` 数组标记保留/删除 |
| `writeGrid(grid, path)` | 通用 VTU 写入 |
| `writeStatsCSV(path, chains, keptNodeIds, distToRef)` | 输出统计 CSV（链数量、每链节点数、保留节点数） |

---

### 6. 输出文件

| 文件 | 格式 | 内容 |
|------|------|------|
| `{prefix}_simplified_nodes.vtu` | VTU | 简化后节点，含 `KeepNode` 标记（1=保留, 0=删除） |
| `{prefix}_simplified_arcs.vtu` | VTU | 简化后弧（当前为原始弧的拷贝） |
| `{prefix}_stats.csv` | CSV | 统计信息：总链数、保留节点数、各链节点数 |

---

## 简化决策参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epsilonIsm` | 0.15 | ISM 距离阈值，DistISM 超过此值则保留节点 |
| `epsilonPos` | 0.10 | 质心空间距离阈值（体素单位） |
| `epsilonBbox` | 0.10 | 包围盒对角线相对差异阈值 |

决策逻辑：**三个指标中任一超过阈值即保留**，否则删除该节点。
