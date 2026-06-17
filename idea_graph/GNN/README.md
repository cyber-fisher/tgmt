# GNN 活性子图向量化实现

这个目录实现了 `活性子图向量化方案.md` 中的方案 B：对每个 merge tree 节点提取局部活性子图，并用 2 层 GCN 编码成固定长度向量。

实现入口：

```bash
python3 GNN/vectorize_active_subgraphs.py \
  -n /path/to/*_nodes.vtu \
  -a /path/to/*_arcs.vtu \
  -s /path/to/data.vti \
  -o GNN/out/tangaroa_gnn
```

`dataset_txt` 输入也支持。这个格式包含：

```text
dataset_txt/
  datas/<case>.vti
  results/<case>/tree_output.txt
```

单个 case 向量化：

```bash
python3 GNN/vectorize_active_subgraphs.py \
  --txt-root /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/dataset_txt \
  --case case17_12x7x5 \
  -o GNN/out_txt/case17_gnn \
  --hidden 32
```

输出：

- `<prefix>_embeddings.csv`：每个 merge tree 节点的 GNN 向量。
- `<prefix>_chain_distances.csv`：沿拓扑链的贪心参考节点和候选节点之间的 GNN cosine distance。

主要参数：

- `--scalar v`：VTI 中的标量场名称，默认 `v`。
- `--hidden 64`：GCN hidden dimension，最终向量维度为 `2 * hidden + 2`。
- `--seed 17`：确定性随机权重种子。
- `--epsilon 0.15`：生成 chain distance 表时的保留阈值。
- `--model GNN/out/gnn_encoder.pt`：加载训练好的 GCN 权重；不传则使用确定性随机 baseline。

训练 GCN 权重：

```bash
python3 GNN/train_gnn_encoder.py \
  -n /path/to/*_nodes.vtu \
  -a /path/to/*_arcs.vtu \
  -s /path/to/data.vti \
  -o GNN/out/gnn_encoder.pt \
  --hidden 32 \
  --epochs 200 \
  --batch-size 32
```

使用 `dataset_txt` 的所有 case 训练：

```bash
python3 GNN/train_gnn_encoder.py \
  --txt-root /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/dataset_txt \
  -o GNN/out_txt/gnn_encoder_dataset_txt.pt \
  --hidden 32 \
  --epochs 200 \
  --batch-size 32
```

如果只训练一个 case，加 `--case case17_12x7x5`。

训练脚本使用同一活性子图的两次随机增强作为正样本，batch 内其他子图作为负样本，通过 InfoNCE / NT-Xent loss 训练 2 层 GCN。

ParaView 导出：

```bash
python3 GNN/export_paraview_results.py \
  -n /path/to/*_nodes.vtu \
  -a /path/to/*_arcs.vtu \
  -s /path/to/data.vti \
  -d GNN/out/tangaroa_gnn_chain_distances.csv \
  -o GNN/out/tangaroa_gnn_paraview
```

用于等值面的文件是 `<prefix>_volume_for_contour.vti`。在 ParaView 中应选中这个 `.vti` 数据源再使用 Contour。

`dataset_txt` 的 ParaView 导出示例：

```bash
python3 GNN/export_paraview_results.py \
  --txt-root /home/czs/TTKAndParaView/ttk-dev/standalone/MergeTree/dataset_gyc/gyc/dataset_txt \
  --case case17_12x7x5 \
  -d GNN/out_txt/case17_gnn_chain_distances.csv \
  -o GNN/out_txt/case17_paraview
```
