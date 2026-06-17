# 基于思路 3 的 TTK `standalone/MergeTree` 落地文档

## 1. 目标

本文档用于指导 AI 在 **TTK 官方仓库的 `standalone/MergeTree`** 目录中实现“思路三：提取原始网格活性子图并比较结构相似性”的工程版本。

目标不是再写一个脱离 TTK 的 demo，而是：

1. 基于 TTK 现有 `standalone/MergeTree` 目录新增一个独立类来承接思路 3 逻辑。
2. 直接读取当前项目 `output/` 目录里的三口输出作为输入。
3. 在 TTK 框架下完成：
   - 合并树读入
   - 拓扑边抽取
   - 活性子图构建
   - 相似性筛选
   - 简化树重建
   - VTK 结果输出

注意：TTK 官方目录名是 **`standalone/MergeTree`**，不是小写 `mergetree`。实现时以官方目录为准。

## 2. 输入数据约定

本实现不再从原始单个标量场文件重新计算 merge tree，而是直接消费已经生成好的三口输出：

- `output/tangaroa_port_0.vtu`
- `output/tangaroa_port_1.vtu`
- `output/tangaroa_port_2.vti`

这三个文件的语义如下。

### 2.1 `port_0.vtu`：树节点表

类型：`vtkUnstructuredGrid`

点数据数组：

- `NodeId`
- `Scalar`
- `VertexId`
- `CriticalType`
- `RegionSize`
- `RegionSpan`

用途：

- `NodeId`：树节点唯一编号
- `Scalar`：节点标量值
- `CriticalType`：临界点类型，`0` 可视为普通节点，非 `0` 视为临界节点
- `VertexId`：原始数据顶点 id，可作为回溯锚点
- `RegionSize` / `RegionSpan`：节点附带统计量，可用于调试和输出

当前样例中：

- 节点数 `150`
- 根节点可视为 `NodeId = 149`

### 2.2 `port_1.vtu`：增广树边表

类型：`vtkUnstructuredGrid`

单元类型：`line`

单元数据数组：

- `SegmentationId`
- `upNodeId`
- `downNodeId`
- `RegionSize`
- `RegionSpan`

用途：

- `downNodeId -> upNodeId` 给出树弧方向
- `SegmentationId` 给出该树弧对应的分割区域 id
- 每条 augmented arc 都有一个独立的 `SegmentationId`

这一步非常关键：**后续活性子图不在全域上比较，而是在对应树弧的局部分割区域中比较。**

### 2.3 `port_2.vti`：原始体数据 + 分割标签

类型：`vtkImageData`

点数据数组：

- `v`
- `w`
- `u`
- `SegmentationId`
- `RegionSize`
- `RegionSpan`
- `RegionType`

当前实现默认使用标量场 `v`。

用途：

- `v`：原始标量值，用于判断体素图中的边是否被阈值 `t` 截穿
- `SegmentationId`：每个网格点所属的分割区域
- `vtkImageData` 的 `Extent / Origin / Spacing` 可恢复规则网格邻接关系

## 3. 算法目标

对每一条拓扑边（即两个结构性临界点之间的一串 augmented 节点）做简化：

- 起点临界节点保留
- 终点临界节点保留
- 中间普通节点根据“活性子图相似性”决定是否保留

核心思想：

1. 对于某个节点阈值 `t = Scalar(node)`，在原始网格上找出所有满足跨阈值的活性边。
2. 只保留与该节点所在树弧局部分割区域相关的活性边，得到局部活性子图。
3. 比较参考节点和当前节点的活性子图差异。
4. 若差异超过阈值，则保留该节点，并将其作为新的参考节点。

## 4. 推荐实现策略

## 4.1 不修改 `main.cpp`，新增独立类

本次实现要求：

- **不要修改 `standalone/MergeTree/main.cpp`**
- 在同目录下新增一个独立类来承接思路 3 后处理逻辑

推荐类名：

- `ActiveGraphMergeTreeSimplifier`

推荐做法：

1. 保留 TTK 官方 `standalone/MergeTree/main.cpp` 原样不动。
2. 新增一个类，专门负责读取三口输出并做思路 3 简化。
3. 该类应可被后续新的 standalone driver、测试程序或其他封装层复用。

这样做的好处：

- 避免污染 TTK 原有 CLI 入口
- 方便单元测试和复用
- 更符合“算法逻辑封装进类、入口只负责调用”的工程习惯
- 后续如果要接入思路一、思路二，也可以复用同样结构

如果后面确实需要命令行入口，应额外新建 driver 文件，例如：

- `activeGraphMain.cpp`

而不是修改现有 `main.cpp`。

## 4.2 第一版只支持 `vtkImageData`

因为当前 `output/tangaroa_port_2.vti` 是规则体数据，第一版实现建议只支持：

- `port_2` 为 `vtkImageData`

不建议第一版就泛化到任意 `vtkUnstructuredGrid`，否则原始网格邻接提取、边枚举、坐标回查都会显著复杂化。

文档里要明确写清：

- 第一阶段目标：支持 `vtkImageData`
- 第二阶段再扩展到非结构网格

## 5. 数据结构设计

建议在 `standalone/MergeTree` 下新增几个本地文件，而不是把所有逻辑都塞进入口文件。

推荐文件：

- `ActiveGraphMergeTreeSimplifier.h`
- `ActiveGraphMergeTreeSimplifier.cpp`
- `ActiveGraphPostProcess.h`
- `ActiveGraphPostProcess.cpp`
- `MergeTreeIO.h`
- `MergeTreeIO.cpp`

如果你希望最小改动，也可以先只增加：

- `ActiveGraphMergeTreeSimplifier.h`
- `ActiveGraphMergeTreeSimplifier.cpp`

第一版由这个类自己封装读取、计算和输出流程。

### 5.1 树节点结构

```cpp
struct TreeNode {
  int nodeId;
  double scalar;
  int criticalType;
  int vertexId;
  double regionSize;
  double regionSpan;
  std::array<double, 3> position;
};
```

### 5.2 树弧结构

```cpp
struct TreeArc {
  int arcId;              // 可直接用 cell id
  int segmentationId;
  int downNodeId;
  int upNodeId;
  double regionSize;
  double regionSpan;
};
```

### 5.3 拓扑边结构

```cpp
struct TopologicalChain {
  std::vector<int> nodeIds;     // 从低到高排序
  std::vector<int> arcIds;      // 与 nodeIds 之间的 augmented arcs 对应
};
```

### 5.4 活性子图结构

第一版没必要上通用图类，直接存“规范化后的网格边集合”即可：

```cpp
using GridEdge = std::pair<vtkIdType, vtkIdType>;

struct ActiveSubgraph {
  std::vector<GridEdge> activeEdges;   // 排序后去重
  std::vector<vtkIdType> activeVertices;
  std::array<double, 3> centroid;
  std::array<double, 3> bboxMin;
  std::array<double, 3> bboxMax;
};
```

`GridEdge` 必须保证小 id 在前，方便做集合交并。

## 6. 拓扑边抽取规则

需要先从 augmented tree 中抽取“两个结构节点之间的链”。

结构节点定义建议如下：

- 根节点
- 叶节点
- 入度或出度不等于 1 的节点
- 或 `CriticalType != 0` 的节点

对每个结构节点出发，沿着“唯一后继”向上走，直到下一个结构节点，形成一条 `TopologicalChain`。

链上的节点顺序必须按 `Scalar` 从小到大排列。

这一步的目的，是把 `port_1` 中很多小弧合并成一条更符合“思路 3”表述的拓扑边。

## 7. 活性子图构建

## 7.1 原始网格边定义

当前 `port_2` 是 `vtkImageData`，所以第一版直接使用规则网格邻接。

推荐用 **6 邻接**：

- `(+x)`
- `(+y)`
- `(+z)`

只枚举正方向，避免重复边。

设点标量为 `f(p)`，节点阈值为 `t`，则一条原始网格边 `(u, v)` 是活性边，当且仅当：

```text
(f(u) <= t && f(v) > t) || (f(v) <= t && f(u) > t)
```

## 7.2 局部分割区域限制

如果只按阈值在整个体数据里找活性边，会混入别的分支上的等值面。

因此必须加局部限制。

对于链上的某个节点 `node_i`，其局部候选分割区域定义为：

- 与 `node_i` 在该链上相邻的 augmented arc 的 `SegmentationId`
- 对内部节点，通常是“前一条 arc + 后一条 arc”的并集
- 对链端点，只取链内相邻的一条 arc

然后只保留满足以下条件的原始网格边：

1. 它本身是阈值活性边
2. 边两端点至少有一个点的 `SegmentationId` 属于该局部分割区域集合

第一版可以使用这个近似规则。不要一开始追求特别复杂的区域追踪。

## 7.3 活性子图附加几何量

为后续“带位置补偿”的相似性准备，构图时顺手计算：

- 活性边数
- 活性点质心 `centroid`
- 轴对齐包围盒 `bbox`
- 包围盒对角线长度

这些量几乎零成本，但可以显著提升筛选稳定性。

## 8. 相似性度量

## 8.1 第一层：边集 Jaccard

这是第一版主判据。

设两个节点的活性边集合为 `E_A, E_B`：

```text
J(A, B) = |E_A ∩ E_B| / |E_A ∪ E_B|
DistJ(A, B) = 1 - J(A, B)
```

实现建议：

- `activeEdges` 先排序去重
- 用双指针求交并
- 避免 `unordered_set<pair<...>>` 的哈希复杂度和实现噪音

## 8.2 第二层：位置补偿项

Jaccard 对“同样数量的边整体平移”不敏感，因此建议加轻量几何项：

- 质心相对位移
- 包围盒对角线相对变化

定义：

```text
DistPos = ||centroid_A - centroid_B|| / max(diag_A, eps)
DistBbox = |diag_A - diag_B| / max(diag_A, eps)
```

总差异度可以先用简单线性组合：

```text
DistTotal = wJ * DistJ + wP * DistPos + wB * DistBbox
```

推荐默认值：

- `wJ = 0.7`
- `wP = 0.2`
- `wB = 0.1`

如果你想更保守，也可以先用 OR 判据：

```text
DistJ >= epsilonJ
or DistPos >= epsilonPos
or DistBbox >= epsilonBbox
```

第一版更推荐 OR 判据，因为更容易调参和解释。

## 9. 简化策略

对每条 `TopologicalChain` 做顺序贪心：

1. 保留链起点
2. `ref = 起点`
3. 从前到后扫描中间节点
4. 计算 `ref` 和 `node_i` 的活性子图差异
5. 若超过阈值，则保留 `node_i`，并更新 `ref = node_i`
6. 链终点强制保留

这是思路三最容易落地、也最稳妥的第一版。

## 10. 输出设计

建议输出四类结果。

### 10.1 简化节点文件

文件名：

- `<prefix>_simplified_nodes.vtu`

内容：

- 从 `port_0` 拷贝所有点
- 新增点数组 `KeepNode`，`0/1`

这样最稳，因为不会破坏原始节点 id 映射。

### 10.2 简化树边文件

文件名：

- `<prefix>_simplified_arcs.vtu`

内容：

- 只保留被选中节点之间重连后的弧
- 单元数据可继承：
  - `downNodeId`
  - `upNodeId`
  - `ChainId`
  - `CollapsedArcCount`

### 10.3 调试报告

文件名：

- `<prefix>_stats.csv`

每行包含：

- `chainId`
- `refNodeId`
- `candidateNodeId`
- `distJ`
- `distPos`
- `distBbox`
- `keep`

### 10.4 活性子图掩码

可选输出：

- `<prefix>_active_mask.vti`

思路：

- 给每个网格点写 `SelectedChainId` 或 `ActiveFrequency`
- 用于 ParaView 可视化

这不是第一版必须项，但有助于调试。

## 11. 类接口设计

由于当前要求是不修改 `main.cpp`，这里优先定义类接口，而不是 CLI。

推荐参数结构：

```cpp
struct ActiveGraphSimplificationParams {
  std::string treeNodesPath;
  std::string treeArcsPath;
  std::string segmentationPath;
  std::string scalarFieldName{"v"};
  std::string outputPrefix{"active_graph"};
  double epsilonJ{0.15};
  double epsilonPos{0.10};
  double epsilonBbox{0.10};
  bool useWeightedDistance{false};
  double wJ{0.7};
  double wP{0.2};
  double wB{0.1};
};
```

推荐类接口：

```cpp
class ActiveGraphMergeTreeSimplifier {
public:
  int run(const ActiveGraphSimplificationParams &params);

private:
  int loadData(const ActiveGraphSimplificationParams &params);
  int buildTopologicalChains();
  int simplify();
  int writeOutputs(const ActiveGraphSimplificationParams &params) const;
};
```

如果后续需要命令行程序，应由新的 driver 负责：

1. 解析参数
2. 填充 `ActiveGraphSimplificationParams`
3. 调用 `ActiveGraphMergeTreeSimplifier::run()`

## 12. 代码组织建议

## 12.1 `ActiveGraphMergeTreeSimplifier.*`

职责：

- 对外提供统一入口 `run()`
- 协调读入、链抽取、活性子图构建、筛选和输出
- 作为思路 3 的主控制类

这一层是核心新增内容。

## 12.2 `MergeTreeIO.*`

职责：

- 读 `port_0`
- 读 `port_1`
- 读 `port_2`
- 构建：
  - `nodeId -> TreeNode`
  - `arcId -> TreeArc`
  - 邻接表
- 写 `simplified_nodes.vtu`
- 写 `simplified_arcs.vtu`
- 写 `stats.csv`

## 12.3 `ActiveGraphPostProcess.*`

职责：

- 抽取拓扑边
- 构建节点局部活性子图
- 计算 Jaccard/位置差异
- 执行贪心筛选
- 生成 `KeepNode` 掩码

## 13. CMake 修改建议

TTK 官方当前 `standalone/MergeTree/CMakeLists.txt` 只有一个 `main.cpp`。

当前要求是不修改 `main.cpp`，但可以修改 `CMakeLists.txt` 以编译新类或新 driver。

如果只是先把类编进现有 target，可改成：

```cmake
add_executable(${PROJECT_NAME}
  main.cpp
  ActiveGraphMergeTreeSimplifier.cpp
  MergeTreeIO.cpp
  ActiveGraphPostProcess.cpp
)
```

并补充 VTK 依赖：

```cmake
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    ttkMergeTree
    VTK::IOXML
    VTK::CommonCore
    VTK::CommonDataModel
)
```

如果写 CSV 用到标准库即可，不需要额外依赖。

如果希望彻底不碰现有可执行程序，建议新增一个 target：

```cmake
add_executable(ttkActiveGraphMergeTreeCmd
  activeGraphMain.cpp
  ActiveGraphMergeTreeSimplifier.cpp
  MergeTreeIO.cpp
  ActiveGraphPostProcess.cpp
)
```

这种方式比修改 `ttkMergeTreeCmd` 更符合当前要求。

## 14. 实现顺序

建议 AI 严格按下面顺序落代码。

### 第一步：新建类骨架

先创建：

- `ActiveGraphMergeTreeSimplifier.h`
- `ActiveGraphMergeTreeSimplifier.cpp`

只放最小可编译接口：

- 参数结构
- `run()`
- 若干私有阶段函数

### 第二步：读三口数据

先确保能正确打印：

- 节点数
- 弧数
- `port_2` 尺寸
- 默认标量场范围

### 第三步：构建树结构和拓扑边

验证：

- 能从 `downNodeId/upNodeId` 重建树
- 能抽取出多条 `TopologicalChain`
- 每条链节点标量单调不减

### 第四步：实现规则网格活性边枚举

验证：

- 给一个节点阈值，能统计出活性边数
- 边端点标量确实跨过阈值

### 第五步：加入局部分割区域约束

验证：

- 相比全局活性边，局部活性边数明显减少
- 不同链上的节点活性图分布不同

### 第六步：实现 Jaccard 与顺序贪心

验证：

- 能输出 `KeepNode`
- 所有链端点保留
- 中间普通节点数量减少

### 第七步：加入位置补偿项

验证：

- 一些 Jaccard 变化不大但空间位置明显变化的节点能被保留

### 第八步：写出 VTK 结果

验证：

- ParaView 能打开
- `KeepNode` 可视化正常
- 简化弧连接关系正确

## 15. 验收标准

AI 写完代码后，至少应满足以下验收条件：

1. 能在 TTK `standalone/MergeTree` 下成功编译。
2. 能直接读取：
   - `output/tangaroa_port_0.vtu`
   - `output/tangaroa_port_1.vtu`
   - `output/tangaroa_port_2.vti`
3. 能输出 `KeepNode` 掩码结果。
4. 能输出重建后的简化树弧文件。
5. 不需要修改原有 `main.cpp`。
6. 若新增可执行程序，也不能破坏原有 `ttkMergeTreeCmd` 的默认行为。

## 16. 不要做的事

为了保证第一版可落地，AI 不要一上来做这些事：

- 不要一开始上精确 GED
- 不要引入图神经网络
- 不要先泛化到任意 `vtkUnstructuredGrid`
- 不要重写 TTK 核心过滤器 `ttkMergeTree`
- 不要修改现有 `main.cpp`
- 不要把所有逻辑硬塞进单个源文件

第一版只需要把“基于活性子图的局部结构相似性简化”跑通。

## 17. 给 AI 的直接任务描述

如果要把这份文档直接喂给 AI，可以用下面这段作为执行指令：

> 请在 TTK 官方仓库的 `standalone/MergeTree` 目录中新增一个独立类 `ActiveGraphMergeTreeSimplifier`，不要修改现有 `main.cpp`。该类直接读取 `port_0.vtu`、`port_1.vtu`、`port_2.vti` 三口输出，不重新计算 merge tree。`port_0` 提供节点信息，`port_1` 提供 augmented tree 弧及 `SegmentationId`，`port_2` 是 `vtkImageData`，包含标量场 `v` 和点级 `SegmentationId`。请先只支持 `vtkImageData`。实现流程是：重建树结构，抽取拓扑边，在每个链上按节点标量阈值构建局部活性子图，活性边定义为规则网格中跨阈值的原始网格边，并限制在该节点相邻弧的 `SegmentationId` 局部区域内。使用边集 Jaccard 距离作为主判据，并增加质心位移和包围盒对角线变化作为轻量位置补偿，采用顺序贪心筛选保留节点。输出 `KeepNode` 点数组、简化弧文件和调试统计 CSV。若需要命令行入口，请额外新建 driver 文件，不要改现有 `main.cpp`。

## 18. 当前仓库中的对应输入文件

本地样例输入路径如下：

- `/Users/guoyichi/Desktop/iou/output/tangaroa_port_0.vtu`
- `/Users/guoyichi/Desktop/iou/output/tangaroa_port_1.vtu`
- `/Users/guoyichi/Desktop/iou/output/tangaroa_port_2.vti`

建议在代码里先用这组路径做开发验证。
