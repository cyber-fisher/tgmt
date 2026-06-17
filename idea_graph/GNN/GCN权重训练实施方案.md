# GCN 权重训练实施方案

本文档说明如何基于当前活性子图数据构造正负样本，对 GCN 编码器进行训练。

当前 `GNN/vectorize_active_subgraphs.py` 已经实现了：

```text
VTU/VTI 读取
活性子图提取
节点特征构造
2 层 GCN 编码
embedding 输出
distGNN 计算
```

但当前 GCN 权重是由固定随机种子生成的，没有通过数据训练。训练目标是让 GCN 学到更有意义的结构相似性：

```text
相似活性子图 -> embedding 更近
差异活性子图 -> embedding 更远
```

## 1. 总体目标

已新增训练脚本：

```text
GNN/train_gnn_encoder.py
```

训练完成后保存模型：

```text
GNN/out/gnn_encoder.pt
```

之后向量化脚本支持加载训练好的模型：

```bash
python3 GNN/vectorize_active_subgraphs.py \
  ... \
  --model GNN/out/gnn_encoder.pt
```

整体流程：

```text
nodes.vtu / arcs.vtu / data.vti
        |
        v
提取所有 merge tree 节点的活性子图
        |
        v
对每个子图做两次随机增强
        |
        v
同一子图的两个增强版本作为正样本
batch 内其他子图作为负样本
        |
        v
InfoNCE / NT-Xent 对比学习训练 GCN
        |
        v
保存训练好的 encoder
```

## 2. 正样本构造

对同一个活性子图 `G_n` 做两次轻微扰动：

```text
G_n^1 = augment(G_n)
G_n^2 = augment(G_n)
```

这两个图来自同一个 merge tree 节点，因此作为正样本对：

```text
(G_n^1, G_n^2)
```

推荐增强方式：

| 增强方式 | 推荐参数 | 说明 |
|---|---:|---|
| edge dropout | `0.10` | 随机删除少量活性边 |
| feature mask | `0.10` | 随机遮挡部分节点特征 |
| scalar noise | `0.01` | 对 scalar 相关特征加轻微噪声 |
| coord noise | `0.01` | 对中心化坐标特征加轻微噪声 |

这些增强只作用在训练张量上，不修改原始 VTI/VTU 数据。

## 3. 负样本构造

第一版采用 batch 内负样本。

假设一个 batch 取 `N` 个活性子图：

```text
G1, G2, ..., GN
```

每个图增强两次：

```text
G1_a, G1_b
G2_a, G2_b
...
GN_a, GN_b
```

其中：

```text
G1_a 和 G1_b 是正样本
G1_a 和 batch 中其他图都是负样本
```

这种方式不需要人工标签，适合当前数据规模。

后续可以加入更严格的负样本过滤，例如：

- 不同拓扑链的节点优先作为负样本。
- scalar 差距大的节点优先作为负样本。
- 活性边数或 bbox 差异明显的节点优先作为负样本。

第一版先不做复杂 negative mining，先把训练管线跑通。

## 4. 损失函数

推荐使用 InfoNCE / NT-Xent loss。

对一个 batch，得到 `2N` 个 embedding：

```text
Z = [z1_a, z1_b, z2_a, z2_b, ..., zN_a, zN_b]
```

计算 cosine similarity 矩阵：

```text
S[i, j] = cosine(Z[i], Z[j]) / temperature
```

每个样本的正样本是同一个原始子图的另一个增强版本：

```text
0 <-> 1
2 <-> 3
4 <-> 5
...
```

损失形式：

```text
loss_i = -log(
  exp(S[i, pos_i])
  /
  sum_{j != i} exp(S[i, j])
)
```

推荐参数：

```text
temperature = 0.2
```

## 5. 模型结构

第一版保持当前 GCN 结构不变：

```text
节点特征，11 维
  -> GCN layer 1
  -> ReLU
  -> GCN layer 2
  -> ReLU
  -> mean pooling + max pooling
  -> 拼接 log(|V|), log(|E|)
  -> L2 normalize
```

推荐：

```text
hidden = 32
```

当前 tangaroa 数据只有几百个活性子图，`hidden=32` 比 `hidden=64` 更稳，不容易过拟合。

## 6. 需要修改的代码

### 6.1 让 GCN 支持训练

`GcnGraphEncoder` 已支持 `trainable` 参数：

```python
class GcnGraphEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=64, seed=17, trainable=False):
        ...
        for param in self.parameters():
            param.requires_grad_(trainable)
```

随机 baseline 使用：

```python
encoder = GcnGraphEncoder(hidden_dim=32, seed=17, trainable=False)
```

训练时使用：

```python
encoder = GcnGraphEncoder(hidden_dim=32, seed=17, trainable=True)
```

### 6.2 训练脚本

已新增：

```text
GNN/train_gnn_encoder.py
```

主要功能：

```text
1. 读取 nodes.vtu / arcs.vtu / data.vti
2. 提取并缓存所有非空活性子图
3. 构造图张量
4. 每个 epoch 随机分 batch
5. 每个子图生成两个增强版本
6. 计算 InfoNCE loss
7. 反向传播更新 GCN 权重
8. 保存模型 checkpoint
9. 输出 loss 日志
```

### 6.3 向量化脚本支持加载模型

`vectorize_active_subgraphs.py` 已支持参数：

```bash
--model GNN/out/gnn_encoder.pt
```

如果提供 `--model`，脚本会加载 checkpoint 中的 GCN 权重：

```python
checkpoint = torch.load(args.model, map_location="cpu")
encoder.load_state_dict(checkpoint["model_state_dict"])
```

如果没有提供 `--model`，继续使用当前确定性随机 baseline。

## 7. 图张量和增强

当前特征构造逻辑已通过 `ActiveSubgraphVectorizer.build_graph_tensors(...)` 复用：

```python
ActiveSubgraphVectorizer.build_graph_tensors(subgraph) -> GraphTensors
```

返回：

```python
GraphTensors:
    x: torch.Tensor             # [num_nodes, 11]
    edge_index: list[(u, v)]
    graph_stats: torch.Tensor   # [log(|V|), log(|E|)]
```

增强在 `GraphTensors` 上执行。

### 7.1 edge dropout

```python
keep = torch.rand(num_edges) > edge_dropout
edge_index = edge_index[keep]
```

### 7.2 feature mask

```python
mask = torch.rand_like(x) < feature_mask
x[mask] = 0
```

### 7.3 scalar noise

scalar 相关特征是：

```text
x[:, 0:3]
```

加噪声：

```python
x[:, 0:3] += torch.randn_like(x[:, 0:3]) * scalar_noise
```

### 7.4 coord noise

坐标相关特征是：

```text
x[:, 3:6]
```

加噪声：

```python
x[:, 3:6] += torch.randn_like(x[:, 3:6]) * coord_noise
```

## 8. 推荐训练命令

第一版训练命令：

```bash
python3 GNN/train_gnn_encoder.py \
  -n /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_nodes.vtu \
  -a /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_arcs.vtu \
  -s /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_2.vti \
  -o GNN/out/gnn_encoder.pt \
  --hidden 32 \
  --epochs 200 \
  --batch-size 64 \
  --lr 1e-3 \
  --temperature 0.2 \
  --edge-dropout 0.1 \
  --feature-mask 0.1 \
  --scalar-noise 0.01 \
  --coord-noise 0.01
```

训练后重新生成 embedding：

```bash
python3 GNN/vectorize_active_subgraphs.py \
  -n /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_nodes.vtu \
  -a /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_arcs.vtu \
  -s /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/tangaroa_port_2.vti \
  -o GNN/out/tangaroa_gnn_trained \
  --model GNN/out/gnn_encoder.pt
```

## 9. 训练输出

建议训练脚本输出：

```text
GNN/out/gnn_encoder.pt
GNN/out/gnn_encoder_loss.csv
```

`gnn_encoder.pt` 保存：

```python
{
    "model_state_dict": encoder.state_dict(),
    "hidden_dim": hidden_dim,
    "seed": seed,
    "feature_dim": 11,
    "embedding_dim": 2 * hidden_dim + 2,
    "train_args": vars(args),
}
```

`gnn_encoder_loss.csv` 保存：

```csv
epoch,loss,pos_sim,neg_sim
```

其中：

| 字段 | 含义 |
|---|---|
| `loss` | 对比学习损失 |
| `pos_sim` | 正样本平均 cosine similarity |
| `neg_sim` | 负样本平均 cosine similarity |

训练正常时应该看到：

```text
loss 下降
pos_sim 上升
neg_sim 低于 pos_sim
```

## 10. 训练后如何重新选择阈值

训练后 `distGNN` 的分布会变化，不能继续默认使用旧阈值。

重新统计距离分布：

```bash
python3 - <<'PY'
import csv
import numpy as np

rows = list(csv.DictReader(open("GNN/out/tangaroa_gnn_trained_chain_distances.csv")))
d = np.array([float(r["distGNN"]) for r in rows])

print("min", d.min())
print("p50", np.quantile(d, 0.50))
print("p75", np.quantile(d, 0.75))
print("p90", np.quantile(d, 0.90))
print("p95", np.quantile(d, 0.95))
print("max", d.max())
PY
```

建议初始阈值选在：

```text
p90 到 p95 之间
```

然后再根据 ParaView 中 `keep=1` 的位置是否合理微调。

## 11. 训练效果验收

第一版训练完成后，至少检查：

1. `loss` 能稳定下降。
2. `pos_sim` 明显高于 `neg_sim`。
3. `vectorize_active_subgraphs.py --model` 可以正常输出 CSV。
4. 输出的 `embeddings.csv` 没有 NaN / Inf。
5. 输出的 `chain_distances.csv` 没有 NaN / Inf。
6. `distGNN` 分布比随机 baseline 更有区分度。
7. ParaView 中 `keep=1` 的节点落在可解释的局部变化区域。

如果这些条件满足，就说明 GCN 权重训练流程可用。

## 12. 后续增强

第一版跑通后，可以继续扩展：

1. 多数据集训练，提高泛化能力。
2. 负样本过滤，减少相似节点被误当作负样本。
3. hard negative mining，重点训练难区分样本。
4. 尝试 GraphSAGE / GAT 替代 GCN。
5. 加入混合距离：

```text
distFinal =
  alpha * distGNN
+ beta  * distSize
+ gamma * distBBox
```

这样可以同时利用 GNN 表达和可解释几何统计量。
