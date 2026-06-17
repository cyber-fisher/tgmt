#!/usr/bin/env python3
"""Train the active-subgraph GCN encoder with self-supervised contrastive loss."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from vectorize_active_subgraphs import (
    ActiveSubgraphExtractor,
    ActiveSubgraphVectorizer,
    GcnGraphEncoder,
    GraphTensors,
    GridEdge,
    create_data_from_args,
)


@dataclass
class GraphSample:
    node_id: int
    graph: GraphTensors


def clone_graph(graph: GraphTensors) -> GraphTensors:
    return GraphTensors(
        features=graph.features.clone(),
        edge_index=list(graph.edge_index),
        graph_stats=graph.graph_stats.clone(),
    )


def augment_graph(
    graph: GraphTensors,
    edge_dropout: float,
    feature_mask: float,
    scalar_noise: float,
    coord_noise: float,
) -> GraphTensors:
    augmented = clone_graph(graph)
    x = augmented.features

    if edge_dropout > 0.0 and augmented.edge_index:
        kept_edges: List[GridEdge] = []
        for edge in augmented.edge_index:
            if torch.rand(()) >= edge_dropout:
                kept_edges.append(edge)
        augmented.edge_index = kept_edges
        augmented.graph_stats = torch.tensor(
            [math.log1p(x.shape[0]), math.log1p(len(kept_edges))],
            dtype=torch.float32,
        )

    if feature_mask > 0.0 and x.numel() > 0:
        mask = torch.rand_like(x) < feature_mask
        x[mask] = 0.0

    if scalar_noise > 0.0 and x.shape[0] > 0:
        x[:, 0:3] += torch.randn_like(x[:, 0:3]) * scalar_noise

    if coord_noise > 0.0 and x.shape[0] > 0:
        x[:, 3:6] += torch.randn_like(x[:, 3:6]) * coord_noise

    return augmented


def encode_graph(encoder: GcnGraphEncoder, graph: GraphTensors) -> torch.Tensor:
    return encoder(graph.features, graph.edge_index, graph.features.shape[0], graph.graph_stats)


def info_nce_loss(embeddings: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, float, float]:
    if embeddings.shape[0] < 4 or embeddings.shape[0] % 2 != 0:
        raise ValueError("InfoNCE expects at least two positive pairs")

    logits = embeddings @ embeddings.T / temperature
    batch_views = logits.shape[0]
    device = logits.device

    self_mask = torch.eye(batch_views, dtype=torch.bool, device=device)
    logits = logits.masked_fill(self_mask, -1.0e9)

    targets = torch.arange(batch_views, device=device)
    targets = torch.where(targets % 2 == 0, targets + 1, targets - 1)

    loss = torch.nn.functional.cross_entropy(logits, targets)

    with torch.no_grad():
        similarity = embeddings @ embeddings.T
        pos_sim = similarity[torch.arange(batch_views, device=device), targets].mean().item()
        pos_mask = torch.zeros_like(self_mask)
        pos_mask[torch.arange(batch_views, device=device), targets] = True
        neg_mask = ~(self_mask | pos_mask)
        neg_sim = similarity[neg_mask].mean().item() if torch.any(neg_mask) else 0.0

    return loss, pos_sim, neg_sim


def load_training_samples(
    args: argparse.Namespace,
    hidden_dim: int,
    seed: int,
    max_vertices: int,
) -> List[GraphSample]:
    if args.txt_root is not None and args.case is None:
        all_samples: List[GraphSample] = []
        for case_name in discover_txt_cases(args.txt_root):
            case_args = argparse.Namespace(**vars(args))
            case_args.case = case_name
            print(f"loading txt case: {case_name}")
            all_samples.extend(
                load_training_samples(
                    case_args,
                    hidden_dim,
                    seed,
                    max_vertices,
                )
            )
        print(f"total training samples from txt root: {len(all_samples)}")
        return all_samples

    data = create_data_from_args(args)
    extractor = ActiveSubgraphExtractor(data)
    vectorizer = ActiveSubgraphVectorizer(
        data,
        extractor,
        GcnGraphEncoder(hidden_dim=hidden_dim, seed=seed, trainable=False),
    )

    samples: List[GraphSample] = []
    empty_count = 0
    skipped_large_count = 0
    for node_id in sorted(data.nodes):
        subgraph = extractor.extract(node_id, data.incident_seg_ids.get(node_id, set()))
        if not subgraph.active_vertices:
            empty_count += 1
            continue
        if max_vertices > 0 and len(subgraph.active_vertices) > max_vertices:
            skipped_large_count += 1
            continue
        samples.append(GraphSample(node_id=node_id, graph=vectorizer.build_graph_tensors(subgraph)))

    print(f"training samples: {len(samples)}")
    print(f"empty subgraphs skipped: {empty_count}")
    if max_vertices > 0:
        print(f"large subgraphs skipped: {skipped_large_count} (max vertices {max_vertices})")
    return samples


def discover_txt_cases(txt_root: Path) -> List[str]:
    datas_dir = txt_root / "datas"
    if not datas_dir.is_dir():
        raise ValueError(f"datas directory not found under {txt_root}")
    cases = sorted(path.stem for path in datas_dir.glob("*.vti"))
    cases = [
        case
        for case in cases
        if (txt_root / "results" / case / "tree_output.txt").is_file()
    ]
    if not cases:
        raise ValueError(f"no dataset_txt cases found under {txt_root}")
    return cases


def loss_path_for_model(output: Path) -> Path:
    suffix = output.suffix
    if suffix:
        return output.with_name(output.name[: -len(suffix)] + "_loss.csv")
    return output.with_name(output.name + "_loss.csv")


def serializable_args(args: argparse.Namespace) -> dict:
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def train(args: argparse.Namespace) -> int:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(args.torch_threads, 4)))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    samples = load_training_samples(
        args,
        args.hidden,
        args.seed,
        args.max_vertices,
    )
    if len(samples) < 2:
        raise ValueError("need at least two non-empty active subgraphs for contrastive training")

    encoder = GcnGraphEncoder(hidden_dim=args.hidden, seed=args.seed, trainable=True)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_hidden = int(checkpoint.get("hidden_dim", args.hidden))
        if checkpoint_hidden != args.hidden:
            raise ValueError(f"resume hidden_dim={checkpoint_hidden} does not match --hidden {args.hidden}")
        encoder.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    loss_path = loss_path_for_model(args.output)
    append_loss = args.resume is not None and loss_path.exists()
    with loss_path.open("a" if append_loss else "w", newline="") as fp:
        writer = csv.writer(fp)
        if not append_loss:
            writer.writerow(["epoch", "loss", "pos_sim", "neg_sim"])

        for epoch in range(start_epoch, args.epochs + 1):
            random.shuffle(samples)
            epoch_losses: List[float] = []
            epoch_pos: List[float] = []
            epoch_neg: List[float] = []

            for batch in make_batches(samples, args.batch_size, args.max_batch_vertices):
                if len(batch) < 2:
                    continue

                views: List[torch.Tensor] = []
                for sample in batch:
                    graph_a = augment_graph(
                        sample.graph,
                        args.edge_dropout,
                        args.feature_mask,
                        args.scalar_noise,
                        args.coord_noise,
                    )
                    graph_b = augment_graph(
                        sample.graph,
                        args.edge_dropout,
                        args.feature_mask,
                        args.scalar_noise,
                        args.coord_noise,
                    )
                    views.append(encode_graph(encoder, graph_a))
                    views.append(encode_graph(encoder, graph_b))

                embeddings = torch.stack(views, dim=0)
                loss, pos_sim, neg_sim = info_nce_loss(embeddings, args.temperature)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
                optimizer.step()

                epoch_losses.append(float(loss.item()))
                epoch_pos.append(pos_sim)
                epoch_neg.append(neg_sim)

            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            mean_pos = sum(epoch_pos) / max(1, len(epoch_pos))
            mean_neg = sum(epoch_neg) / max(1, len(epoch_neg))
            writer.writerow([epoch, f"{mean_loss:.10g}", f"{mean_pos:.10g}", f"{mean_neg:.10g}"])
            fp.flush()

            if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
                print(
                    f"epoch {epoch:04d} "
                    f"loss={mean_loss:.6f} "
                    f"pos_sim={mean_pos:.6f} "
                    f"neg_sim={mean_neg:.6f}"
                )
            if epoch % args.checkpoint_every == 0:
                save_checkpoint(args.output, encoder, optimizer, args, epoch)

    save_checkpoint(args.output, encoder, optimizer, args, args.epochs)

    print(f"wrote model: {args.output}")
    print(f"wrote loss log: {loss_path}")
    return 0


def graph_vertex_count(sample: GraphSample) -> int:
    return int(sample.graph.features.shape[0])


def make_batches(samples: Sequence[GraphSample], batch_size: int, max_batch_vertices: int) -> List[List[GraphSample]]:
    if max_batch_vertices <= 0:
        return [list(samples[start : start + batch_size]) for start in range(0, len(samples), batch_size)]

    batches: List[List[GraphSample]] = []
    current: List[GraphSample] = []
    current_vertices = 0
    for sample in samples:
        sample_vertices = graph_vertex_count(sample)
        would_exceed_count = len(current) >= batch_size
        would_exceed_vertices = current and current_vertices + sample_vertices > max_batch_vertices
        if would_exceed_count or would_exceed_vertices:
            batches.append(current)
            current = []
            current_vertices = 0
        current.append(sample)
        current_vertices += sample_vertices
    if current:
        batches.append(current)
    return batches


def save_checkpoint(
    output: Path,
    encoder: GcnGraphEncoder,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
) -> None:
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hidden_dim": args.hidden,
            "seed": args.seed,
            "feature_dim": 11,
            "embedding_dim": args.hidden * 2 + 2,
            "epoch": epoch,
            "train_args": serializable_args(args),
        },
        output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train active-subgraph GCN encoder with InfoNCE.")
    parser.add_argument("-n", "--nodes", type=Path, help="merge tree nodes VTU path")
    parser.add_argument("-a", "--arcs", type=Path, help="merge tree arcs VTU path")
    parser.add_argument("-s", "--segmentation", type=Path, help="scalar/segmentation VTI path")
    parser.add_argument("--txt-root", type=Path, help="dataset_txt root containing datas/ and results/")
    parser.add_argument("--case", help="case name under dataset_txt; omit in training to use all cases")
    parser.add_argument("--tree-txt", type=Path, help="direct tree_output.txt path")
    parser.add_argument("--image", type=Path, help="direct VTI image path for --tree-txt")
    parser.add_argument("--tree-type", default=0, type=int, choices=[0, 1], help="txt tree type: 0=JT, 1=ST")
    parser.add_argument("-o", "--output", required=True, type=Path, help="output checkpoint path")
    parser.add_argument("--scalar", default="v", help="scalar field name in VTI, default: v")
    parser.add_argument("--hidden", default=32, type=int, help="GCN hidden dimension, default: 32")
    parser.add_argument("--seed", default=17, type=int, help="random seed, default: 17")
    parser.add_argument("--epochs", default=200, type=int, help="training epochs, default: 200")
    parser.add_argument("--batch-size", default=32, type=int, help="number of source graphs per batch")
    parser.add_argument("--lr", default=1.0e-3, type=float, help="learning rate, default: 1e-3")
    parser.add_argument("--weight-decay", default=1.0e-4, type=float, help="AdamW weight decay")
    parser.add_argument("--temperature", default=0.2, type=float, help="InfoNCE temperature")
    parser.add_argument("--edge-dropout", default=0.1, type=float, help="edge dropout probability")
    parser.add_argument("--feature-mask", default=0.1, type=float, help="feature mask probability")
    parser.add_argument("--scalar-noise", default=0.01, type=float, help="noise std for scalar features")
    parser.add_argument("--coord-noise", default=0.01, type=float, help="noise std for coordinate features")
    parser.add_argument("--grad-clip", default=5.0, type=float, help="gradient clipping norm")
    parser.add_argument("--log-every", default=10, type=int, help="epoch logging interval")
    parser.add_argument("--checkpoint-every", default=10, type=int, help="checkpoint interval in epochs")
    parser.add_argument("--resume", type=Path, help="resume training from checkpoint")
    parser.add_argument("--torch-threads", default=1, type=int, help="torch CPU thread count; 1 is most stable")
    parser.add_argument(
        "--max-batch-vertices",
        default=120000,
        type=int,
        help="split batches so total source vertices per batch stays below this; 0 disables",
    )
    parser.add_argument(
        "--max-vertices",
        default=0,
        type=int,
        help="skip subgraphs larger than this; 0 keeps all subgraphs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(train(parse_args()))
