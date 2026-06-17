# ActiveGraph 简化说明

## 1. 项目目录

当前目录主要包含以下内容：

```text
idea_graph/
├── 思路3_TTK落地文档.md
├── 技术文档_ActiveGraph简化流程.md
├── result/
└── active_graph_simplifier/
    ├── main.cpp
    ├── ActiveGraphMergeTreeSimplifier.h
    ├── ActiveGraphMergeTreeSimplifier.cpp
    ├── ActiveGraphPostProcess.h
    ├── ActiveGraphPostProcess.cpp
    ├── MergeTreeIO.h
    ├── MergeTreeIO.cpp
    ├── CMakeLists.txt
    └── build/
```

各部分作用如下。

### `思路3_TTK落地文档.md`

最初的设计文档，说明目标、输入数据约定和算法思路。

### `active_graph_simplifier/`

核心实现目录。

#### `main.cpp`

程序入口，负责：

- 设置默认输入路径
- 解析命令行参数
- 调用简化主流程

#### `ActiveGraphMergeTreeSimplifier.*`

总控模块，负责串联整个流程：

- 读数据
- 抽取拓扑链
- 执行简化
- 写出结果

#### `MergeTreeIO.*`

输入输出基础模块，负责：

- 读取树节点 `port_0`
- 读取树边 `port_1`
- 读取体数据 `port_2`
- 构建树的邻接关系

#### `ActiveGraphPostProcess.*`

后处理模块，负责：

- 提取拓扑链
- 为链上节点构建局部活性子图
- 比较节点之间的差异
- 决定哪些节点保留

#### `build/`

CMake 编译目录，保存可执行文件和构建中间结果。

### `result/`

实验输出目录。当前会保存不同参数下的结果文件，例如：

- `*_simplified_nodes.vtu`
- `*_simplified_arcs.vtu`
- `*_stats.csv`

## 2. 程序流程

程序整体流程很简单：

1. 读取输入数据
2. 构建树邻接
3. 抽取拓扑链
4. 按阈值逐链做简化
5. 输出结果文件

## 3. 输入参数说明

程序支持以下参数。

### `-n <path>`

树节点文件路径，对应 `.vtu` 文件。

里面主要包含：

- `NodeId`
- `Scalar`
- `CriticalType`
- `VertexId`

### `-a <path>`

树边文件路径，对应 `.vtu` 文件。

里面主要包含：

- `downNodeId`
- `upNodeId`
- `SegmentationId`

### `-s <path>`

体数据文件路径，对应 `.vti` 文件。

里面主要使用：

- 标量场 `v`
- 分割标签 `SegmentationId`

### `-o <prefix>`

输出前缀。

例如：

```bash
-o result/test
```

则会生成：

- `result/test_simplified_nodes.vtu`
- `result/test_simplified_arcs.vtu`
- `result/test_stats.csv`

### `-j <value>`

Jaccard 距离阈值。

作用：

- 控制两个节点活性边集合的差异有多大时才保留当前节点

含义：

- 值越小，越容易保留节点
- 值越大，越不容易保留节点

### `-p <value>`

位置距离阈值。

作用：

- 控制两个活性子图质心位置变化达到多大时保留当前节点

含义：

- 值越小，越敏感
- 值越大，越宽松

### `-b <value>`

包围盒距离阈值。

作用：

- 控制两个活性子图尺度变化达到多大时保留当前节点

含义：

- 值越小，越容易触发保留
- 值越大，越不容易触发保留

## 4. 参数判定方式

当前实现使用的是 OR 判据：

```text
distJ >= epsilonJ
or distPos >= epsilonPos
or distBbox >= epsilonBbox
```

也就是说，只要三项里有一项超过阈值，当前节点就会被保留。

因此整体规律是：

- `j / p / b` 调小，结果更细，保留节点更多
- `j / p / b` 调大，结果更粗，保留节点更少

## 5. 输出文件说明

### `*_simplified_nodes.vtu`

简化后的节点结果。

特点：

- 保留原始节点坐标
- 新增 `KeepNode` 数组

其中：

- `KeepNode = 1` 表示保留
- `KeepNode = 0` 表示删除

### `*_simplified_arcs.vtu`

简化后的边结果。

主要包含：

- `ChainId`
- `CollapsedArcCount`
- `downNodeId`
- `upNodeId`

### `*_stats.csv`

调试统计文件，用于查看每次节点比较结果。

列包括：

- `chainId`
- `refNodeId`
- `candidateNodeId`
- `distJ`
- `distPos`
- `distBbox`
- `keep`

## 6. 运行示例

```bash
cd active_graph_simplifier/build
./ActiveGraphMergeTreeSimplifier \
  -o /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/idea_graph/result/test \
  -j 0.15 -p 0.10 -b 0.10
```

## 7. 当前实现特点

当前版本的特点是：

- 输入固定面向 merge tree 三口输出
- 体数据目前只支持 `vtkImageData`
- 标量字段默认使用 `v`
- 阈值参数已经真实接入简化流程
- 输出结果适合直接在 ParaView 中查看

