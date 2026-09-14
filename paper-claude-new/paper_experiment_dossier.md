# 基于 SAT+GPS 的几何过滤增广轮廓树：论文实验资料汇总

**文档用途**：本文档是当前工作区截至 **2026-09-06** 的实验档案，供外部 AI 直接据此撰写论文的实验章节、方法修订说明和结果讨论。它不替代此前的历史文档，也不要求本机安装 LaTeX。

本文档同时承担一个重要作用：区分 `main.tex` 中的早期方法描述、已经完成的历史实验，以及当前真正运行并产生最新结果的代码。论文撰写时应以“当前实现口径”和本文档中的“论文可安全表述”部分为准，避免把旧版随机 GCN、旧版 11 维特征和新版 SAT+GPS 结果混在一起。

## 0. 一页摘要

本文研究的问题是：给定三维规则网格标量场及其增广轮廓树，如何在**保留所有拓扑临界节点**的前提下，删除冗余普通节点，同时尽量保持等值面的几何形状、拓扑截交结构和面积/体积属性曲线。

当前工作流如下：

```text
标量场 + 增广轮廓树
        ↓
按阈值提取活性网格子图
        ↓
ID 位置编码 + RWSE + LapPE
        ↓
Edge-GINE 局部消息传递
        ↓
SAT 风格全局注意力（Q/K=h_local，V=h）
        ↓
图级 130 维 SAT+GPS embedding
        ↓
SAT cosine + Jaccard + λ 连续边残差
        ↓
多分支加权贪心简化
        ↓
保留临界节点的几何过滤增广轮廓树
```

当前最好的已完成配置为：

```text
encoder: SATGraphGPS
metric: satpair
gnn_weight: 0.20
lambda_weight: 0.25
alpha: 0.40
multi_branch: 1
target: 每个 case 保留约 50% 节点
```

在 20 个合成 case、相同约 50% kept 的条件下，该配置相对 Jaccard-multi 的五项平均误差为：

| 指标 | Jaccard-multi | SAT+GPS pairrisk (`gnn_weight=0.20`) | 相对变化 |
|---|---:|---:|---:|
| NNAE-C / Chamfer | 0.002585332 | **0.002460194** | **−4.840%** |
| NNAE-H / Hausdorff | 0.030300650 | **0.029002995** | **−4.283%** |
| NNAE-J | 0.053823448 | **0.052859914** | **−1.790%** |
| PCE-A | 0.004973312 | **0.004765015** | **−4.188%** |
| PCE-V | 0.024553904 | **0.024426102** | **−0.520%** |

五项指标均低于 Jaccard；逐 case 优于 Jaccard 的数量分别为 `15/20、14/20、15/20、13/20、12/20`。

必须严格使用以下表述：

> 当前已经完成的是 **SAT+GPS + Jaccard + λ 对齐残差**，也可称为 SAT-enhanced pair-aligned risk。SAT+GPS encoder 确实参与了删点排序，但最终距离仍保留 Jaccard 和 λ，且在当前推荐配置中 SAT cosine 的权重为 0.20。因此该结果不是严格意义上的纯 SAT+GPS-only 结果。

## 1. 文件和结果的来源

### 1.1 当前代码

| 内容 | 文件 |
|---|---|
| 原始 GraphGPS 训练与导出 | [`train_gnn_graphgps.py`](../../scripts/train_gnn_graphgps.py) |
| SAT+GPS 训练与导出 | [`train_sat_graphgps.py`](../../scripts/train_sat_graphgps.py) |
| 统一简化器 | [`python_simplifier.py`](../../scripts/python_simplifier.py) |
| SAT pairrisk 全量评估 | [`eval_sat_pairrisk_all.py`](../../scripts/eval_sat_pairrisk_all.py) |
| SAT cosine 权重扫描 | [`eval_sat_gnn_weight_sweep.py`](../../scripts/eval_sat_gnn_weight_sweep.py) |
| 权重扫描汇总 | [`weight_sweep_summary.md`](../sat_gnn_weight_final/weight_sweep_summary.md) |

### 1.2 模型和 embedding

| 产物 | 说明 |
|---|---|
| [`gnn_graphgps.pt`](../../models/gnn_graphgps.pt) | 原始 GraphGPS checkpoint；不要覆盖 |
| [`gnn_satlite_lambda.pt`](../../models/gnn_satlite_lambda.pt) | 当前 SAT+GPS/SAT-lite checkpoint；不要覆盖 |
| [`embeddings/`](../../data/embeddings/) | 原始 GraphGPS 的 case embedding |
| [`embeddings_sat_lambda/`](../../data/embeddings_sat_lambda/) | SAT+GPS 的 case embedding |

### 1.3 结果目录

| 结果 | 文件/目录 |
|---|---|
| 原始 GraphGPS 全 20 case、约 50% kept | [`graphgps_all50/metrics.md`](../graphgps_all50/metrics.md) |
| 原始 SAT+lambda 全 20 case | [`sat_lambda_all/metrics.md`](../sat_lambda_all/metrics.md) |
| SAT pairrisk，旧权重 0.05 | [`sat_pairrisk_all/metrics.md`](../sat_pairrisk_all/metrics.md) |
| `gnn_weight` 完整扫描 | [`sat_gnn_weight_sweep/metrics_all.json`](../sat_gnn_weight_sweep/metrics_all.json) |
| `gnn_weight=0.20/0.35` 严格复跑 | [`sat_gnn_weight_final/`](../sat_gnn_weight_final/) |
| Tangaroa 零样本 | [`tangora/zero_shot/`](../tangora/zero_shot/) |
| 原有历史横向表 | [`compare_all.md`](./compare_all.md) |
| SAT 前后历史对照 | [`sat_before_after_comparison.md`](./sat_before_after_comparison.md) |
| 当前简要 SAT+GPS 结果 | [`sat_gps_current_results.md`](./sat_gps_current_results.md) |

## 2. 论文方法与当前实现的统一描述

### 2.1 输入：标量场、增广轮廓树和活性子图

设规则三维网格为 `Ω=(V_Ω,E_Ω)`，每个网格顶点 `p` 携带标量值 `f(p)`。增广轮廓树记为 `T=(V_T,E_T,f)`。树中：

- 根、叶、鞍点及其他非度 2 节点属于结构/临界节点；
- 度为 2 的中间节点属于普通节点；
- 相邻临界节点之间的普通节点链构成拓扑分支。

对轮廓树节点 `v`，阈值为 `t=f(v)`。网格边 `(p,q)` 在阈值 `t` 下是活性边，当且仅当：

```text
(f(p) ≤ t < f(q)) 或 (f(q) ≤ t < f(p))
```

所有活性边端点构成活性顶点集，活性边及其端点组成局部活性子图 `G(v)=(V_act(v),E_act(v))`。该子图的语义是：

- 顶点是等值面附近的原始网格采样点；
- 边是被阈值等值面穿过的网格边；
- 活性边的空间分布近似表达等值面的截交形状；
- 活性边数量表达局部几何规模。

这个构造无需先生成三角化等值面，就能为每一个轮廓树普通节点构造可比较的图输入。最终评价时仍然使用 Marching Cubes 等值面重建，避免筛选度量与评价指标完全同源。

### 2.2 当前节点特征：基础特征从 11 维缩减，但模型输入为 28 维

这是论文中最容易混淆的地方。

早期版本的原始节点特征为 11 维；当前代码已经去掉标量、相对坐标、拓扑度和分割编码，只保留 4 维节点 ID 位置编码：

```text
[sin(id/17), cos(id/17), sin(id/53), cos(id/53)]
```

随后为每个活性子图拼接：

```text
4 维 ID 位置编码
+ 16 维 RWSE（随机游走结构编码）
+ 8 维 LapPE（拉普拉斯位置编码）
= 28 维送入模型第一层的最终输入
```

因此论文应写成：

> 原始 11 维语义特征中的基础节点特征被压缩为 4 维 ID 正余弦位置编码；在 GraphGPS 输入端再拼接 16 维 RWSE 与 8 维 LapPE，模型实际输入维度为 28。

不能写成“当前模型输入只有 4 维”，也不能继续写成“模型输入是 11 维”。

### 2.3 原始 GraphGPS encoder

原始 GraphGPS 位于 `train_gnn_graphgps.py`，每一层由两个并行分支组成：

1. 无边属性的 GINE/GIN 风格局部消息传递；
2. 图内多头自注意力全局分支。

原始 GPS 层可概括为：

```text
h_local = GINE(h, E)
h_global = MultiheadAttention(h, h, h)
h_next = FFN(LN(h + h_local) + LN(h + h_global))
```

当前原始 GraphGPS 的超参数为：

| 参数 | 值 |
|---|---:|
| 输入维度 | 28 |
| 隐藏维度 | 64 |
| 层数 | 3 |
| attention heads | 4 |
| 图级 embedding | 130 |
| 训练目标 | NT-Xent |

图级读出为：

```text
mean_pool(64) || max_pool(64) || log(1+|V_act|) || log(1+|E_act|)
= 130 维
```

### 2.4 当前 SAT+GPS encoder

SAT+GPS 位于 [`train_sat_graphgps.py`](../../scripts/train_sat_graphgps.py) 的 `SATGraphGPS` 和 `SATGPSLayer`。

每个 `SATGPSLayer` 包含：

- 带 5 维边属性的 `EdgeGINELayer` 局部分支；
- 4 头 `MultiheadAttention` 全局分支；
- 残差连接、LayerNorm 和两层 FFN。

原始 GraphGPS 的注意力是：

```text
Q = h, K = h, V = h
```

当前 SAT 风格注意力是：

```text
Q = h_local
K = h_local
V = h
```

代码中对应的逻辑是：

```python
q = h_local
kv = h
out, _ = self.attn(q, q, kv, ...)
```

这意味着当前确实使用了 GPS 的局部消息传递 + 全局注意力框架；SAT 的作用是让局部结构提取器控制全局注意力中的 query/key 匹配，而由原始节点状态提供 value。它不是“完整 SAT 编码器串联一个未修改 GPS 编码器”，而是 SAT 风格注意力对 GraphGPS 的融合改造。

### 2.5 SAT+GPS 的边特征

对方向为 `p→q` 的活跃网格边，当前 Edge-GINE 使用 5 维属性：

```text
[lambda, |f(q)-f(p)|, axis_x, axis_y, axis_z]
```

其中：

```text
lambda = (t - f(p)) / (f(q) - f(p))
```

并截断到 `[0,1]`。反向边 `q→p` 使用 `1-lambda`，标量差和方向 one-hot 保持一致。λ 表示等值面在网格边上的线性插值位置，提供了 Jaccard 二值边集没有的连续信息。

### 2.6 训练方式

当前 SAT+GPS 不是未训练的随机网络，而是通过自监督 NT-Xent 训练得到 checkpoint。

正样本来自两类配对：

1. 同一轮廓树分支上相邻的普通节点；
2. 同一普通节点对应的轻微标量扰动副本。

训练时对 LapPE 随机进行符号翻转，以处理拉普拉斯特征向量的符号不确定性。NT-Xent 目标使正样本图级 projection embedding 相似。

当前 SAT+GPS 训练记录：

| 项目 | 当前值 |
|---|---|
| 训练 case | 14 个 synthetic case |
| 验证 case | `case09`、`case13`、`case18` |
| 训练 pair | 约 6,383 |
| 验证 pair | 约 1,793 |
| optimizer | AdamW |
| 初始学习率 | `5e-4` |
| weight decay | `1e-4` |
| batch size | 16 |
| temperature | `τ=0.1` |
| jitter | `0.01` |
| dropout | `0.1` |
| checkpoint 训练设备 | NVIDIA RTX 3080 Ti |
| 参数量 | 200,079 |
| checkpoint | `gnn_satlite_lambda.pt` |

当前 embedding exporter 只为普通节点生成向量；临界节点没有 embedding 时，简化器回退到活性边 Jaccard 距离。因此 `gnn_weight=1.0` 也不能解释成所有节点 pair 都是纯 cosine，除非另行补齐临界节点 embedding。

## 3. 当前删点距离和贪心算法

### 3.1 Jaccard 距离

两个阈值节点 `a,b` 的活性边集合分别为 `E_a,E_b`，当前基础 Jaccard 距离为：

```text
D_J(a,b) = 1 - |E_a ∩ E_b| / |E_a ∪ E_b|
```

空集合按实现返回 0。

### 3.2 λ 对齐残差

对共同活跃边 `E_a∩E_b`，计算线性插值位置差：

```text
R_lambda(a,b) = mean_{e∈E_a∩E_b} |lambda_a(e)-lambda_b(e)|
```

没有共同活跃边时残差为 0。当前 pairrisk 的基础部分是：

```text
D_Jλ = D_J + lambda_weight × R_lambda
```

推荐配置 `lambda_weight=0.25`。

### 3.3 SAT+GPS cosine 距离

SAT+GPS encoder 输出已 L2 归一化的图级 embedding `z_a,z_b`，余弦距离为：

```text
D_SATGPS(a,b) = 1 - z_a · z_b
```

### 3.4 当前最终混合距离

对 `satpair`，代码实际执行：

```text
D_pair(a,b)
  = gnn_weight × D_SATGPS(a,b)
  + (1-gnn_weight) × [D_J(a,b) + lambda_weight × R_lambda(a,b)]
```

当前推荐配置：

```text
gnn_weight = 0.20
lambda_weight = 0.25
```

即：

```text
D_pair = 0.20 × D_SATGPS + 0.80 × (D_J + 0.25R_lambda)
```

这也是为什么必须称为“SAT+GPS 增强的 pair-aligned 方法”，而不是纯 SAT+GPS-only。

### 3.5 多分支加权的当前代码口径

对候选标量层级 `t`，代码找出满足分支标量范围的活跃分支集合。每条分支先计算：

```text
topo_weight_l = persistence_l / Σ persistence
```

代码中的几何权重并不是简单的“当前活性边数归一化”，而是参考节点和候选节点的活性边规模差：

```text
g_l = |s_ref,l - s_cand,l| / max(s_ref,l, s_cand,l)
geo_weight_l = g_l / Σ g_l
```

其中 `s` 是活性边数量。随后按 `alpha` 混合：

```text
w_l = alpha × topo_weight_l + (1-alpha) × geo_weight_l
```

实际实现会对最终权重再归一化。候选节点的整体距离为：

```text
D = Σ_l w_l × d_l
```

当前评估固定 `alpha=0.40`、`multi_branch=1`。这与 `main.tex` 中“几何规模权重直接按 s_l 归一化”的早期公式不完全一致，论文需要按当前代码修正或明确将旧公式作为理论版本、当前实现作为实验版本。

### 3.6 结构约束和贪心判定

简化器首先无条件保留所有结构节点。对主链和非主链分支，按标量顺序扫描普通候选节点：

```text
如果 D(reference, candidate) >= eps：保留 candidate，并更新 reference
否则：删除/跳过 candidate
```

分支末端始终保留。被删除节点跨越的弧随后合并，输出合法的过滤增广轮廓树。临界节点保留率是硬约束，不属于“平均误差越小越好”的连续指标。

## 4. 数据集、划分和实验协议

### 4.1 Synthetic 20-case 数据

当前数据包为 `experiments/data/dataset_txt/`，每个 case 提供一个 `.vti` 和一个 `tree_output.txt`。20 个 case 的网格和树规模如下：

| case | 网格 | 树节点总数 | 分支数 | 50% 目标 kept |
|---|---|---:|---:|---:|
| case01 | 5×5×3 | 75 | 17 | 38 |
| case02 | 6×6×3 | 108 | 7 | 54 |
| case03 | 7×5×4 | 140 | 13 | 70 |
| case04 | 6×6×4 | 144 | 13 | 72 |
| case05 | 7×7×4 | 196 | 17 | 98 |
| case06 | 8×6×4 | 192 | 19 | 96 |
| case07 | 9×6×4 | 216 | 27 | 108 |
| case08 | 10×5×4 | 200 | 25 | 100 |
| case09 | 8×8×4 | 256 | 33 | 128 |
| case10 | 9×7×4 | 252 | 17 | 126 |
| case11 | 10×6×4 | 240 | 14 | 120 |
| case12 | 11×6×4 | 264 | 23 | 132 |
| case13 | 8×8×5 | 320 | 41 | 160 |
| case14 | 9×8×5 | 360 | 39 | 180 |
| case15 | 10×8×5 | 400 | 39 | 200 |
| case16 | 11×8×5 | 440 | 39 | 220 |
| case17 | 12×7×5 | 420 | 43 | 210 |
| case18 | 12×8×5 | 480 | 44 | 240 |
| case19 | 9×9×6 | 486 | 27 | 243 |
| case20 | 10×10×5 | 500 | 19 | 250 |

总节点数为 `5,689`，约 50% 目标合计 `2,845`。最终严格复跑的 kept 合计为 `2,845`，总压缩率约 `49.97%`（按每 case 平均约 50% 口径）。

### 4.2 训练/验证/测试划分

当前 encoder 训练划分为：

```text
训练：14 个 case
验证：case09, case13, case18
预留测试：case01, case17, case20
```

注意：当前“20-case 平均表”是工程阶段的全量评估，包含训练 case、验证 case 和预留测试 case，并不等价于严格的独立测试集结果。论文应优先报告三种结果：

1. 20-case 全量结果：展示总体工程效果；
2. `case01/case17/case20` 测试子集：展示未用于 encoder 训练的合成 case 泛化；
3. Tangaroa 零样本或裁剪留出结果：展示跨域能力。

当前尚未完成多随机种子和多压缩率的统计显著性验证，因此不能写成“已证明普遍优于 Jaccard”。

### 4.3 kept 对齐和阈值校准

不同方法使用同一个 `eps` 并不公平，因为不同距离的数值尺度不同。当前评估对每个 case、每个方法单独二分 `eps`：

```text
eps 搜索区间：[0, 1.5]
目标：round(total_nodes × 0.5)
最终严格复跑：18 次二分迭代
```

这样比较的是“相同压缩强度下选择的节点质量”。如果只把 `eps` 调大而不重新对齐 kept，得到的主要是“删除更多节点”的结果，不能作为算法优劣结论。

## 5. 评价指标定义

所有连续误差指标均为越小越好。

### 5.1 kept 和压缩率

```text
compression = (total_nodes - kept_nodes) / total_nodes
```

`kept` 只用于控制比较公平性，不是质量指标。

### 5.2 NNAE-C / Chamfer

对每个被删除的普通节点，按标量值找到最近的保留节点；分别重建两张等值面，计算双向平均点集距离，并除以 VTI 包围盒对角线：

```text
Chamfer(A,B) = 0.5 × [mean_{a∈A} min_{b∈B} ||a-b||
                       + mean_{b∈B} min_{a∈A} ||b-a||]
```

所有被删节点的平均值是 NNAE-C。它反映平均几何偏差，对少数极端点比 Hausdorff 更鲁棒。

### 5.3 NNAE-H / Hausdorff

同一对等值面计算双向最大点距离，再除以包围盒对角线：

```text
Hausdorff(A,B) = max(max_{a∈A} min_{b∈B} ||a-b||,
                     max_{b∈B} min_{a∈A} ||b-a||)
```

它反映最坏几何偏差，对局部异常和漏采样更敏感。

### 5.4 NNAE-J

比较被删节点和最近保留节点的活性网格边集合：

```text
NNAE-J = 1 - |E_a∩E_b| / |E_a∪E_b|
```

它衡量活性截交结构的变化。由于它与 Jaccard 筛选度量有一定同源性，论文应把 NNAE-C/H 和 PCE-A/V 作为更重要的外部质量指标，把 NNAE-J 作为拓扑/截交一致性指标。

### 5.5 PCE-A 和 PCE-V

在所有原始节点和保留节点上计算等值面面积曲线 `A(t)` 与体积曲线 `V(t)`。用保留节点值对原始采样点做分段线性插值，计算相对 L2 误差：

```text
PCE_A = ||A - A_hat||_2 / ||A||_2
PCE_V = ||V - V_hat||_2 / ||V||_2
```

PCE-A 衡量面积属性保持，PCE-V 衡量体积属性保持。

### 5.6 临界节点保留率

```text
R_critical = critical_kept / critical_total
```

当前简化器的设计要求为 100%。在所有当前合成和 Tangaroa 运行中，临界节点均全部保留。该指标应作为硬约束报告，而不是与连续误差简单加权平均。

## 6. 全部方法的当前结果

### 6.1 20-case 平均结果

下面的 Jaccard-multi、原始 GraphGPS、原始 SAT+lambda 使用同一轮约 50% kept 协议。`pairrisk` 的 `gnn_weight=0.05` 是此前冻结版本，`0.20/0.35` 是最新权重调大后的严格复跑。

| 方法 | NNAE-C | NNAE-H | NNAE-J | PCE-A | PCE-V |
|---|---:|---:|---:|---:|---:|
| Jaccard-multi | 0.002585332 | 0.030300650 | 0.053823448 | 0.004973312 | 0.024553904 |
| 原始 GraphGPS，`gw=0.05` | 0.002578358 | 0.030028810 | 0.053831490 | 0.004950258 | 0.024589877 |
| SAT+lambda，`gw=0.05` | 0.002586238 | 0.030381414 | 0.053945701 | 0.004961568 | 0.024565712 |
| SAT pairrisk，`gw=0.05,lw=0.25` | 0.002528812 | 0.029578690 | 0.053161054 | 0.004861074 | 0.024353390 |
| SAT pairrisk，`gw=0.20,lw=0.25` | **0.002460194** | **0.029002995** | **0.052859914** | **0.004765015** | **0.024426102** |
| SAT pairrisk，`gw=0.35,lw=0.25` | **0.002431678** | **0.028555956** | **0.053145265** | **0.004855099** | **0.024522795** |

相对 Jaccard 的变化：

| 方法 | NNAE-C | NNAE-H | NNAE-J | PCE-A | PCE-V |
|---|---:|---:|---:|---:|---:|
| 原始 GraphGPS，`gw=0.05` | −0.270% | −0.897% | +0.015% | −0.463% | +0.147% |
| SAT+lambda，`gw=0.05` | +0.035% | +0.267% | +0.227% | −0.236% | +0.048% |
| SAT pairrisk，`gw=0.05` | −2.186% | −2.383% | −1.231% | −2.257% | −0.817% |
| SAT pairrisk，`gw=0.20` | **−4.840%** | **−4.283%** | **−1.790%** | **−4.188%** | **−0.520%** |
| SAT pairrisk，`gw=0.35` | **−5.943%** | **−5.758%** | **−1.260%** | **−2.377%** | **−0.127%** |

### 6.2 原始 SAT+lambda 与 pairrisk 的差别

原始 SAT+lambda 只把 SAT+GPS embedding 以 `gnn_weight=0.05` 混入 Jaccard 距离，没有共同活跃边上的 λ residual。其相对 Jaccard 平均变化为：

```text
NNAE-C  +0.035%
NNAE-H  +0.267%
NNAE-J  +0.227%
PCE-A   −0.236%
PCE-V   +0.048%
```

它没有全面超过 Jaccard。加入 common-edge λ residual 后，`gw=0.05` 已经五项均改善；再把 `gnn_weight` 提高到 `0.20`，几何指标和面积属性指标进一步改善。

这说明当前收益不是“只要换 SAT 注意力就自动超过 Jaccard”，而是来自三部分共同作用：

1. SAT+GPS 的局部/全局结构 embedding；
2. 共同活跃边的显式对应关系；
3. 活跃边上的连续 λ 位移信息。

### 6.3 权重扫描结论

固定 `lambda_weight=0.25` 后扫描 `gnn_weight`：

| `gnn_weight` | NNAE-C 相对 Jaccard | NNAE-H 相对 Jaccard | NNAE-J 相对 Jaccard | PCE-A 相对 Jaccard | PCE-V 相对 Jaccard | 解释 |
|---:|---:|---:|---:|---:|---:|---|
| 0.05 | −2.041% | −2.040% | −1.323% | −2.260% | −0.768% | 保守，SAT 作用较小 |
| 0.10 | −3.184% | −3.681% | −1.727% | −1.346% | −0.315% | 几何改善，属性改善较弱 |
| **0.20** | **−4.840%** | **−4.283%** | **−1.790%** | **−4.188%** | **−0.520%** | 综合最均衡，当前推荐 |
| 0.35 | −5.943% | −5.758% | −1.260% | −2.377% | −0.127% | Chamfer/Hausdorff 最好，但整体平衡略弱 |
| 0.50 | −5.433% | −4.154% | **+1.766%** | **+0.591%** | **+0.683%** | 拓扑和属性开始退化 |
| 0.75 | −0.183% | −0.966% | +4.378% | −0.832% | +6.865% | 权重过高，PCE-V 明显退化 |
| 1.00 | +36.646% | +22.817% | +29.843% | +20.037% | +29.209% | 纯 cosine 当前明显不稳定 |

`gw=0.20` 和 `gw=0.35` 的严格复跑均精确命中每个 case 的目标 kept。当前综合归一化误差约为：

```text
gw=0.20: 0.97497
gw=0.35: 0.97601
```

因此建议论文主结果采用 `gw=0.20`，同时把 `gw=0.35` 作为几何指标更激进的敏感性结果。

### 6.4 训练/验证/测试子集结果

为了避免 20-case 全量平均掩盖数据划分问题，下面给出最新 `gw=0.20` 与 Jaccard 的子集结果。测试子集是 `case01/case17/case20`，没有参与 SAT encoder 的训练。

#### 训练 14 case

| 方法 | NNAE-C | NNAE-H | NNAE-J | PCE-A | PCE-V |
|---|---:|---:|---:|---:|---:|
| Jaccard-multi | 0.002534010 | 0.030602555 | 0.053572990 | 0.004708996 | 0.029211139 |
| SAT+GPS pairrisk `gw=0.20` | 0.002443930 | 0.029409462 | 0.052137147 | 0.004606232 | 0.028511688 |

#### 验证 3 case：case09/case13/case18

| 方法 | NNAE-C | NNAE-H | NNAE-J | PCE-A | PCE-V |
|---|---:|---:|---:|---:|---:|
| Jaccard-multi | 0.001561744 | 0.023976892 | 0.039570015 | 0.002443961 | 0.004760641 |
| SAT+GPS pairrisk `gw=0.20` | 0.001541474 | 0.023582264 | 0.038923047 | 0.002269645 | 0.004763443 |

#### 测试 3 case：case01/case17/case20

| 方法 | NNAE-C | NNAE-H | NNAE-J | PCE-A | PCE-V |
|---|---:|---:|---:|---:|---:|
| Jaccard-multi | 0.003848421 | 0.035215518 | 0.069245686 | 0.008736135 | 0.022613403 |
| SAT+GPS pairrisk `gw=0.20` | **0.003454812** | **0.032526878** | 0.070169697 | **0.008001373** | 0.025022694 |

测试子集上，`gw=0.20` 的 Chamfer、Hausdorff 和面积误差更低，但 NNAE-J 和体积误差略高。这是一个重要的诚实结论：当前方法在全量平均和大多数 case 上表现良好，但还不能宣称所有留出测试指标都稳定优于 Jaccard。

## 7. `gnn_weight=0.20` 逐 case 主结果

下表是当前建议的论文主配置：`SAT+GPS pairrisk`、`gnn_weight=0.20`、`lambda_weight=0.25`、`alpha=0.40`、多分支、约 50% kept。每个 `target/kept` 均精确命中。

| case | 网格 | total | target/kept | eps | Jaccard C | SAT C | Jaccard H | SAT H | Jaccard J | SAT J | Jaccard A | SAT A | Jaccard V | SAT V |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case01 | 5×5×3 | 75 | 38/38 | 0.220922 | 0.009866 | 0.008737 | 0.071100 | 0.064104 | 0.147879 | 0.151180 | 0.022970 | 0.020802 | 0.042502 | 0.051053 |
| case02 | 6×6×3 | 108 | 54/54 | 0.127344 | 0.004819 | 0.004482 | 0.050589 | 0.045575 | 0.083909 | 0.072762 | 0.011254 | 0.009155 | 0.055273 | 0.046666 |
| case03 | 7×5×4 | 140 | 70/70 | 0.113960 | 0.004147 | 0.003921 | 0.051576 | 0.046198 | 0.081736 | 0.081270 | 0.009120 | 0.009470 | 0.039955 | 0.042100 |
| case04 | 6×6×4 | 144 | 72/72 | 0.118498 | 0.004202 | 0.004005 | 0.049766 | 0.045317 | 0.069898 | 0.066014 | 0.007659 | 0.008009 | 0.072071 | 0.077498 |
| case05 | 7×7×4 | 196 | 98/98 | 0.089464 | 0.002588 | 0.002486 | 0.029581 | 0.030029 | 0.065860 | 0.065667 | 0.005781 | 0.006430 | 0.015191 | 0.014726 |
| case06 | 8×6×4 | 192 | 96/96 | 0.090197 | 0.002967 | 0.002736 | 0.035968 | 0.031434 | 0.069462 | 0.069567 | 0.003893 | 0.003611 | 0.053719 | 0.042292 |
| case07 | 9×6×4 | 216 | 108/108 | 0.074266 | 0.001980 | 0.001985 | 0.023331 | 0.024285 | 0.046388 | 0.047279 | 0.005252 | 0.005494 | 0.041181 | 0.040545 |
| case08 | 10×5×4 | 200 | 100/100 | 0.085121 | 0.002369 | 0.002442 | 0.033111 | 0.032672 | 0.055086 | 0.053941 | 0.003642 | 0.003513 | 0.046127 | 0.049988 |
| case09 | 8×8×4 | 256 | 128/128 | 0.079462 | 0.001528 | 0.001534 | 0.024394 | 0.024155 | 0.045466 | 0.045247 | 0.003268 | 0.003195 | 0.003609 | 0.003712 |
| case10 | 9×7×4 | 252 | 126/126 | 0.054623 | 0.002607 | 0.002628 | 0.022976 | 0.023760 | 0.043462 | 0.043783 | 0.002697 | 0.002470 | 0.015289 | 0.015136 |
| case11 | 10×6×4 | 240 | 120/120 | 0.066645 | 0.003036 | 0.003149 | 0.031702 | 0.036305 | 0.044533 | 0.043777 | 0.003278 | 0.003090 | 0.027687 | 0.028285 |
| case12 | 11×6×4 | 264 | 132/132 | 0.053209 | 0.001431 | 0.001416 | 0.023715 | 0.021998 | 0.042572 | 0.042089 | 0.002713 | 0.002681 | 0.015697 | 0.015575 |
| case13 | 8×8×5 | 320 | 160/160 | 0.071526 | 0.002346 | 0.002338 | 0.028898 | 0.029602 | 0.044343 | 0.042963 | 0.002661 | 0.002333 | 0.006240 | 0.006313 |
| case14 | 9×8×5 | 360 | 180/180 | 0.072355 | 0.001143 | 0.001088 | 0.016395 | 0.016310 | 0.042610 | 0.042570 | 0.003387 | 0.003374 | 0.007027 | 0.006735 |
| case15 | 10×8×5 | 400 | 200/200 | 0.068338 | 0.001901 | 0.001891 | 0.022919 | 0.022396 | 0.041004 | 0.039929 | 0.002749 | 0.002877 | 0.006597 | 0.006535 |
| case16 | 11×8×5 | 440 | 220/220 | 0.054749 | 0.001153 | 0.001102 | 0.016588 | 0.016357 | 0.035620 | 0.033909 | 0.002419 | 0.002430 | 0.009831 | 0.009853 |
| case17 | 12×7×5 | 420 | 210/210 | 0.056826 | 0.000935 | 0.000932 | 0.018681 | 0.017462 | 0.033451 | 0.032817 | 0.001858 | 0.001862 | 0.005985 | 0.005806 |
| case18 | 12×8×5 | 480 | 240/240 | 0.043196 | 0.000811 | 0.000752 | 0.018639 | 0.016990 | 0.028902 | 0.028559 | 0.001403 | 0.001281 | 0.004433 | 0.004266 |
| case19 | 9×9×6 | 486 | 243/243 | 0.038475 | 0.000935 | 0.000885 | 0.020217 | 0.019098 | 0.027883 | 0.027363 | 0.002083 | 0.001884 | 0.003312 | 0.003231 |
| case20 | 10×10×5 | 500 | 250/250 | 0.039419 | 0.000745 | 0.000696 | 0.015865 | 0.016014 | 0.026407 | 0.026512 | 0.001380 | 0.001341 | 0.019353 | 0.018209 |

注：`case19` 的逐 case 值以严格复跑的 `metrics_gw0.2.json` 为准；如果外部 AI 只需要论文正文表格，建议使用第 6.1 节平均表，逐 case 表放附录或补充材料。

## 8. 与历史 ISM、GCN 和 GraphGPS 结果的关系

`compare_all.md` 中的 case17/case20 历史表使用 `kept≈100`，而当前主实验使用每个 case 约 50% kept。因此两者不能逐位比较，也不能把历史表中的 ISM 行直接与最新 20-case 平均表混成一张严格公平表。

历史 kept≈100 表可用于说明方法谱系：

- ISM 是经典距离场互信息基线；
- Jaccard 是当前活性边集合基线；
- GCN-random/GCN-trained 是早期图表示实验；
- 原始 GraphGPS 是标准 GPS 结构；
- 当前 SAT+GPS 是 SAT 风格注意力和 λ 边特征的融合版本。

论文中建议将历史结果标为“早期 kept≈100 对照”或放入补充材料；主结论应使用同一轮、同 kept 的 20-case 结果。

## 9. Tangaroa 真实数据零样本结果

当前真实数据目录实际名称为 `experiments/data/tangora/`。数据来自 Tangaroa 速度场，使用：

```text
velocity_magnitude = sqrt(u^2 + v^2 + w^2)
```

构造标量场并提取增广轮廓树。完整原始网格很大，因此当前使用 `20×12×8` 裁剪/降采样版本进行可控实验：

```text
树节点：1,920
分支：13
临界节点：14
Jaccard 和 GraphGPS kept：349
压缩率：约 81.8%
临界节点保留率：14/14
```

当前已有的是**原始 GraphGPS 零样本**，还没有在 Tangaroa 上完成 SAT+GPS pairrisk 的正式公平评估。已有 GraphGPS `gnn_weight=0.05` 结果：

| 方法 | kept | NNAE-C | NNAE-H | PCE-A | PCE-V |
|---|---:|---:|---:|---:|---:|
| Jaccard-single | 349 | 0.0005199 | **0.0089634** | **0.0001389** | 0.0136139 |
| Jaccard-multi | 349 | **0.0005167** | 0.0090094 | 0.0001548 | 0.0135041 |
| 原始 GraphGPS-multi 零样本 | 349 | 0.0005331 | 0.0090655 | 0.0001507 | **0.0133458** |

解释：原始 GraphGPS 零样本只在 PCE-V 上略好，Chamfer、Hausdorff 和 Jaccard 结构误差略差。`gnn_weight=1.0`、`eps=0.1` 的探索性运行只保留 38 个节点，不能作为公平比较。

论文当前可以写：

> Tangaroa 实验验证了跨域零样本推理流程和拓扑节点保留约束，但当前还不足以证明 SAT+GPS 已在真实数据上超过 Jaccard。需要在 Tangaroa 的多个裁剪/分辨率上重新校准 kept 并评估 `gw=0.20`。

## 10. 当前结果的解释、边界和不能过度声称的内容

### 10.1 可以声称的内容

1. SAT+GPS 确实已经接入：每层包含 Edge-GINE 局部分支和 MultiheadAttention 全局分支；
2. SAT 改造体现在 `Q=K=h_local,V=h`；
3. 当前节点基础特征为 4 维 ID PE，模型实际输入为 28 维；
4. λ 被作为有向 Edge-GINE 的连续边属性；
5. 在 20 synthetic cases、相同约 50% kept 下，`gw=0.20,lw=0.25` 的 SAT+GPS pairrisk 五项平均指标均低于 Jaccard；
6. 调大 `gnn_weight` 后 SAT+GPS 对删点排序的影响明显增强；
7. 中等权重 `0.20–0.35` 比 `0.75–1.0` 更稳定。

### 10.2 目前不能声称的内容

1. 不能把 `SAT+GPS + Jaccard + λ` 写成纯 SAT+GPS-only；
2. 不能说 SAT embedding 单独已经超过 Jaccard；`gw=1.0` 当前明显退化；
3. 不能说 Tangaroa 已经超过 Jaccard；目前只有原始 GraphGPS 零样本结果；
4. 不能说结果具有统计显著性；当前主要是单 checkpoint、单训练种子、单一约 50% 压缩率；
5. 不能把全量 20-case 平均称为独立测试集，因为其中包含训练和验证 case；
6. 不能直接把 kept≈100 的旧 ISM 表与当前 50% kept 表放在同一严格横向排名中；
7. 不能继续沿用 `main.tex` 中“11 维输入、两层随机 GCN、无训练 He 初始化”的描述来解释当前 SAT+GPS 结果。

### 10.3 为什么 `gw=1.0` 反而变差

当前 encoder 的训练目标是 NT-Xent，而非直接预测删除节点后的几何误差。它保证相邻分支节点和扰动副本在表示空间中相似，但没有保证 cosine 距离与 Chamfer、Hausdorff、面积或体积误差单调对应。此外，cosine、Jaccard 和 λ residual 的数值范围也不同，未经校准时直接让 cosine 主导会造成距离尺度失配。因此 `gw=0.20` 是当前的工程平衡点，而不是理论上固定的最优常数。

## 11. `main.tex` 必须修改的地方

外部 AI 在改论文前，建议先按下表处理 `main.tex`。本轮只新增实验文档，没有自动改写 `main.tex`，以保留你的论文原稿。

| `main.tex` 旧表述 | 当前实际情况 | 论文建议 |
|---|---|---|
| 实验硬件 RTX 2080 Ti | 当前 SAT 训练记录为 RTX 3080 Ti | 改成真实运行设备；若保留旧实验硬件，按实验分别注明 |
| 默认 `alpha=0.5` | 当前主要评估为 `alpha=0.40` | 主结果写 `alpha=0.40`；旧 alpha 敏感性单独作为历史实验 |
| GNN 节点输入 11 维 | 当前基础特征 4 维，最终输入 28 维 | 改写为“4-D ID PE + 16-D RWSE + 8-D LapPE = 28-D” |
| 标量/坐标/度数/SegmentationId 11 维 | 已从当前 GraphGPS/SAT+GPS 管线去掉 | 只能作为早期版本或消融，不可用于解释当前 checkpoint |
| 两层 GCN、确定性 He 随机初始化、无需训练 | 当前为 3 层 GraphGPS/SATGraphGPS，NT-Xent 训练 | 重写 GNN 章节和训练设置 |
| 无边特征 GCN/GINE | SAT+GPS 使用 5-D λ/标量差/轴向边特征 | 增加 Edge-GINE 和 λ 定义 |
| 普通 GNN embedding 为随机网络 | 当前 SAT+GPS embedding 来自 `gnn_satlite_lambda.pt` | 明确 checkpoint、训练 pair、NT-Xent |
| Jaccard 组合 `max(J,pos,bbox)` | 当前简化器的 `jaccard` 只使用活性边 Jaccard；`satpair` 额外使用 λ residual | 旧 pos/bbox 公式若保留，必须标为未用于当前主结果的理论/历史变体 |
| 几何权重直接按活性边规模 `s_l` 归一化 | 当前代码使用 `|s_ref-s_cand|/max(s_ref,s_cand)` 后归一化 | 按当前实现修正实验方法公式，或明确区分理论式和实现式 |
| `d_l` 只写 Jaccard 或普通 GNN | 当前主结果为 `D_pair=gw D_SATGPS+(1-gw)(D_J+lw R_lambda)` | 在方法和实验章节加入 SAT pairrisk 公式 |
| GNN 优势可直接迁移真实数据 | Tangaroa 原始 GraphGPS 尚未全面超过 Jaccard | 把 Tangaroa 写成零样本诊断和未来验证 |
| 20-case 平均等同独立测试 | 20-case 包含训练/验证/测试 case | 分开报告全量、验证和 `case01/17/20` 测试子集 |

## 12. 论文推荐叙事结构

外部 AI 可以按以下顺序组织论文实验部分：

### 12.1 实验问题

验证三个问题：

1. 结构节点硬保留是否能保证拓扑事件不被删除？
2. 活性子图度量是否能在相同压缩率下保持等值面几何和属性？
3. SAT+GPS 结构表示和 λ 连续边信息是否进一步改善 Jaccard？

### 12.2 方法组

建议至少列出：

```text
Jaccard-single
Jaccard-multi
原始 GraphGPS-multi
SAT+lambda-multi
SAT+GPS pairrisk (gw=0.05)
SAT+GPS pairrisk (gw=0.20，主配置)
SAT+GPS pairrisk (gw=0.35，几何敏感性)
```

ISM、GCN-random、GCN-trained 可以放在 kept≈100 的历史/补充实验中，不能与当前 50% kept 表无条件混排。

### 12.3 主结果

主表使用第 6.1 节的 20-case 平均结果，强调：

- 五项平均指标均下降；
- `gw=0.20` 的综合平衡最好；
- `gw=0.35` 的 Chamfer/Hausdorff 更好，但属性指标改善变小；
- 结构节点保留率始终 100%。

### 12.4 消融

推荐按以下顺序做/写：

```text
A. Jaccard
B. Jaccard + λ residual
C. SAT+GPS cosine + Jaccard
D. SAT+GPS cosine + Jaccard + λ residual
E. gw=0.05/0.10/0.20/0.35/0.50/0.75/1.00
```

当前已有 D 和权重扫描；A、B、C 应在论文最终版补齐为完全同协议的显式消融，以便把收益分解到 Jaccard、λ 和 SAT+GPS 三个来源。

## 13. 下一步实验计划

### 13.1 近期必须完成

1. 把 `gnn_weight=0.20` 在 Tangaroa `20×12×8` 上按相同 kept 重新评估；
2. 在 Tangaroa 的 `100×60×40` 或多个空间裁剪上做留出验证；
3. 补齐 A–E 消融表，特别是 `Jaccard+λ` 和纯 `SAT+GPS cosine`；
4. 在 25%、50%、75% 压缩率上复跑 `gw=0.20`；
5. 使用 3–5 个随机训练种子报告均值和标准差；
6. 记录运行时间、活性子图数量、峰值内存和 embedding 缓存大小。

### 13.2 中期模型优化

如果目标是严格的纯 SAT+GPS-only 超过 Jaccard，当前最重要的改动不是继续盲目增大 `gnn_weight`，而是训练几何风险头：

```text
输入：参考节点/候选节点的 SAT+GPS 表示、共同活跃边、lambda 残差、树位置
输出：删除候选节点后的局部 Chamfer/Hausdorff/NNAE-J/PCE-A/PCE-V 风险
```

建议损失：

```text
0.30 × Chamfer ranking loss
+ 0.25 × Hausdorff ranking loss
+ 0.15 × NNAE-J ranking loss
+ 0.15 × PCE-A ranking loss
+ 0.15 × PCE-V ranking loss
+ 0.05 × NT-Xent
```

训练完成后再测试：

```text
D_final = D_pairwise_geometry_risk
```

而不是继续用 Jaccard 作为 80% 或 95% 的最终决策依据。

### 13.3 特征迁移优化

当前 `sin(id/17)` 等 flat node-ID 编码依赖网格线性编号，跨分辨率和跨裁剪的几何含义不稳定。后续可替换为归一化空间坐标 `(x,y,z)` 的 Fourier 编码，并重新训练 SAT+GPS checkpoint。该改动应作为新的 checkpoint 和新的实验目录保存，不能覆盖当前模型。

## 14. 可复现实验命令

当前 Python 环境：

```text
/root/miniconda3/envs/graphgps/bin/python
```

### 14.1 只加载已有 SAT checkpoint 并导出 embedding

```bash
/root/miniconda3/envs/graphgps/bin/python \
  experiments/scripts/train_sat_graphgps.py \
  --skip-train
```

注意：训练脚本默认会使用/写入 `gnn_satlite_lambda.pt`；复现实验前应确认是否使用 `--skip-train`，避免覆盖已有 checkpoint。

### 14.2 运行单个 SAT pairrisk case

```bash
/root/miniconda3/envs/graphgps/bin/python \
  experiments/scripts/python_simplifier.py \
  --tree experiments/data/dataset_txt/results/case17_12x7x5/tree_output.txt \
  --vti experiments/data/dataset_txt/datas/case17_12x7x5.vti \
  --metric satpair \
  --embeddings experiments/data/embeddings_sat_lambda/case17_gnn_satlambda.csv \
  --eps 0.056826 \
  --alpha 0.40 \
  --gnn-weight 0.20 \
  --lambda-weight 0.25 \
  --multi-branch 1 \
  --output experiments/results/reproduce_case17_gw02
```

`eps` 必须针对目标 kept 重新校准；上面的数值只对应当前 case17 的约 50% 目标。

### 14.3 重新运行权重扫描

```bash
/root/miniconda3/envs/graphgps/bin/python \
  experiments/scripts/eval_sat_gnn_weight_sweep.py \
  --weights 0.05,0.10,0.20,0.35,0.50,0.75,1.0 \
  --lambda-weight 0.25 \
  --iterations 18 \
  --output experiments/results/sat_gnn_weight_reproduce
```

## 15. 最终结论模板（可直接交给外部 AI 改写）

> 本文在 20 个合成增广轮廓树上评估了活性子图 Jaccard、原始 GraphGPS、SAT+lambda 和 SAT+GPS pair-aligned risk。当前 SAT+GPS 编码器保留 GraphGPS 的局部消息传递与全局注意力框架，并以 Edge-GINE 提取的局部结构表示控制 SAT 风格的 query/key，使用原始节点状态作为 value；节点输入由 4 维 ID 位置编码、16 维 RWSE 和 8 维 LapPE 组成，共 28 维，边输入额外包含 λ、标量差和网格轴向信息。所有方法在每个 case 内对齐到约 50% kept，并无条件保留全部拓扑临界节点。
>
> 在 `gnn_weight=0.20`、`lambda_weight=0.25` 的配置下，SAT+GPS pairrisk 相对 Jaccard-multi 的 NNAE-C、NNAE-H、NNAE-J、PCE-A 和 PCE-V 平均误差分别下降 4.840%、4.283%、1.790%、4.188% 和 0.520%。调大 SAT cosine 权重能够提高其对删点排序的影响，但权重超过 0.50 后拓扑和属性指标开始退化，`gnn_weight=1.0` 的纯 cosine 版本目前明显不稳定。
>
> 因此，当前实验支持“SAT+GPS 结构表示与 λ 连续边信息结合后，可以在合成数据的相同压缩率下改善 Jaccard”的结论；但该结果仍属于 SAT+GPS 与 Jaccard/λ 的混合方法，不应解释为纯 SAT+GPS-only 已经全面取代 Jaccard。Tangaroa 当前仅完成原始 GraphGPS 零样本诊断，真实数据上的 SAT+GPS 泛化仍需进一步验证。

