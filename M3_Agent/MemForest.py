from __future__ import annotations

import argparse
import gc
import math
import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans

from mmagent import *
import sys
import mmagent.videograph as vg
sys.modules["videograph"] = vg  

PROMPT_TEMPLATE = """
You are an intelligent memory summarization assistant. Please extract the core information from the following two memory entries and their timestamps, and generate a concise and coherent summary paragraph. If the memories contain relative time expressions (e.g., “last June,” “two months ago”), convert them into specific dates based on the corresponding timestamps (for example, if a memory mentions “last June” and the timestamp is February 2023, the actual time should be June 2022). The summary length should be kept within the combined length of the two original memories as much as possible. Output only the final summarized result, without any explanations, analysis, reasoning steps, or additional content.
Memory 1: {}, Timestamp 1: {}
Memory 2: {}, Timestamp 2: {}
"""

sys_modules_videograph = vg

TEXT_NODE_TYPES = {"episodic", "semantic"}
DEFAULT_CLUSTER_RATIO = 0.05
DEFAULT_KEEP_RATIO = 0.7
DEFAULT_SEMANTIC_TIME_ALPHA = 0.8
DEFAULT_TIME_WINDOW_SIZE = 5
DEFAULT_API_RETRIES = 3
DEFAULT_FAILURE_WAIT_SECONDS = 15.0
DEFAULT_API_TIMEOUT_SECONDS = 60.0
DEFAULT_CLUSTER_WORKERS = 60
DEFAULT_API_FAILURE_LOG_PATH = Path(__file__).resolve().parent / "memforest_api_failures_robot50.txt"


def _ensure_directory(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _collect_graph_files(src_root: str, dst_root: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if os.path.isfile(src_root):
        dst_path = os.path.join(dst_root, os.path.basename(src_root))
        pairs.append((src_root, dst_path))
    elif os.path.isdir(src_root):
        for root, _, files in os.walk(src_root):
            for filename in files:
                if not filename.endswith(".pkl"):
                    continue
                src_path = os.path.join(root, filename)
                rel_path = os.path.relpath(src_path, src_root)
                dst_path = os.path.join(dst_root, rel_path)
                pairs.append((src_path, dst_path))
    else:
        raise FileNotFoundError(f"Cannot find memory graph path: {src_root}")
    pairs.sort(key=lambda item: item[0])
    return pairs


def _node_embedding(node) -> Optional[np.ndarray]:
    embeddings = getattr(node, "embeddings", None)
    if not embeddings:
        return None
    emb_array = np.asarray(embeddings, dtype=np.float32)
    if emb_array.ndim == 1:
        return emb_array
    return emb_array.mean(axis=0)


def _node_timestamp_value(node, fallback: float = 0.0) -> float:
    metadata = getattr(node, "metadata", None)
    if not isinstance(metadata, dict):
        return fallback
    ts = metadata.get("timestamp", fallback)
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return float(ts)
    except Exception:
        return fallback


def _node_timestamp_text(node) -> str:
    metadata = getattr(node, "metadata", None)
    if not isinstance(metadata, dict):
        return ""
    ts = metadata.get("timestamp", "")
    return str(ts)


def _node_text(node) -> str:
    metadata = getattr(node, "metadata", None)
    if not isinstance(metadata, dict):
        return ""
    contents = metadata.get("contents", [])
    if isinstance(contents, list):
        parts = [str(x) for x in contents if x is not None]
        return "\n".join(parts).strip()
    return str(contents).strip()


def _ordered_text_node_ids(graph) -> List[int]:
    ordered_ids: List[int] = []
    seen: Set[int] = set()
    for node_id in getattr(graph, "text_nodes", []):
        node = graph.nodes.get(node_id)
        if not node or node.type not in TEXT_NODE_TYPES:
            continue
        ordered_ids.append(node_id)
        seen.add(node_id)
    for node_id, node in graph.nodes.items():
        if node.type in TEXT_NODE_TYPES and node_id not in seen:
            ordered_ids.append(node_id)
    return ordered_ids


def _is_equivalence_text_node(node) -> bool:
    if not node or getattr(node, "type", None) not in TEXT_NODE_TYPES:
        return False
    metadata = getattr(node, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    contents = metadata.get("contents", [])
    if not isinstance(contents, list):
        contents = [contents]
    for item in contents:
        text = str(item).strip().lower()
        if text.startswith("equivalence"):
            return True
    return False


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _best_merge_pair_via_mst(vectors: np.ndarray) -> Optional[Tuple[int, int]]:
    n = vectors.shape[0]
    if n < 2:
        return None
    edges: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(vectors[i], vectors[j])
            edges.append((sim, i, j))
    edges.sort(key=lambda x: x[0], reverse=True)
    dsu = DisjointSet(n)
    mst: List[Tuple[float, int, int]] = []
    for sim, i, j in edges:
        if dsu.union(i, j):
            mst.append((sim, i, j))
            if len(mst) == n - 1:
                break
    if not mst:
        return None

    degree = [0] * n
    for _, u, v in mst:
        degree[u] += 1
        degree[v] += 1

    degree_sums = [degree[u] + degree[v] for _, u, v in mst]
    max_degree_sum = max(degree_sums) if degree_sums else 1
    if max_degree_sum <= 0:
        max_degree_sum = 1

    def edge_score(item: Tuple[float, int, int]) -> float:
        sim, u, v = item
        degree_norm = (degree[u] + degree[v]) / max_degree_sum
        return 0.99 * sim - 0.01 * degree_norm

    _, i, j = max(mst, key=edge_score)
    return i, j


def _target_keep_count(size: int, keep_ratio: float) -> int:
    if size <= 1:
        return size
    return max(1, min(size, int(math.ceil(size * keep_ratio))))


def _build_joint_clusters(
    graph,
    text_node_ids: Sequence[int],
    cluster_ratio: float,
    semantic_time_alpha: float,
    time_window_size: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    node_ids = [node_id for node_id in text_node_ids if node_id in graph.nodes]
    if not node_ids:
        return {}

    valid_ids: List[int] = []
    valid_vectors: List[np.ndarray] = []
    invalid_ids: List[int] = []
    for node_id in node_ids:
        emb = _node_embedding(graph.nodes[node_id])
        if emb is None:
            invalid_ids.append(node_id)
            continue
        valid_ids.append(node_id)
        valid_vectors.append(np.asarray(emb, dtype=np.float32))

    result: Dict[int, List[int]] = {}
    next_cluster_id = 0

    if valid_ids:
        vectors = np.stack(valid_vectors, axis=0)
        n = len(valid_ids)
        k = max(1, min(n, int(math.floor(cluster_ratio * n))))
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(vectors)

        semantic_clusters: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels.tolist()):
            semantic_clusters[label].append(idx)

        center_index_by_cluster: Dict[int, int] = {}
        for cid, indices in semantic_clusters.items():
            cluster_vectors = vectors[indices]
            centroid = cluster_vectors.mean(axis=0)
            dist2 = np.sum((cluster_vectors - centroid) ** 2, axis=1)
            center_local = int(np.argmin(dist2))
            center_index_by_cluster[cid] = indices[center_local]

        center_indices = set(center_index_by_cluster.values())
        assigned: Dict[int, List[int]] = {cid: [center_index_by_cluster[cid]] for cid in semantic_clusters.keys()}

        ordered_indices = sorted(
            range(n),
            key=lambda i: (_node_timestamp_value(graph.nodes[valid_ids[i]], fallback=float(i)), i),
        )
        time_pos = {idx: pos for pos, idx in enumerate(ordered_indices)}
        center_items = sorted(center_index_by_cluster.items())

        for idx in ordered_indices:
            if idx in center_indices:
                continue
            original_cluster = int(labels[idx])
            best_score = None
            best_clusters: List[int] = []
            for cid, center_idx in center_items:
                semantic_flag = _cosine_similarity(vectors[idx], vectors[center_idx])
                delta = abs(time_pos[idx] - time_pos[center_idx])
                time_flag = 1.0 if delta <= time_window_size else 0.0
                score = semantic_time_alpha * semantic_flag + (1.0 - semantic_time_alpha) * time_flag
                if best_score is None or score > best_score + 1e-12:
                    best_score = score
                    best_clusters = [cid]
                elif abs(score - best_score) <= 1e-12:
                    best_clusters.append(cid)

            if len(best_clusters) == 1:
                target_cluster = best_clusters[0]
            else:
                target_cluster = original_cluster if original_cluster in best_clusters else best_clusters[0]
            assigned[target_cluster].append(idx)

        for cid in sorted(semantic_clusters.keys()):
            cluster_indices = assigned.get(cid, [])
            if not cluster_indices:
                continue
            result[next_cluster_id] = [valid_ids[idx] for idx in cluster_indices]
            next_cluster_id += 1

    for node_id in invalid_ids:
        result[next_cluster_id] = [node_id]
        next_cluster_id += 1
    return result


def _summarize_two_memories(
    llm_client: OpenAI,
    llm_model: str,
    older_text: str,
    older_ts: str,
    newer_text: str,
    newer_ts: str,
) -> str:
    prompt = PROMPT_TEMPLATE.format(older_text, older_ts, newer_text, newer_ts)
    resp = llm_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip() or newer_text


def _embed_text(embed_client: OpenAI, embedding_model: str, text: str) -> np.ndarray:
    resp = embed_client.embeddings.create(model=embedding_model, input=[text])
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def _merge_pair_with_retry(
    graph,
    older_id: int,
    newer_id: int,
    llm_client: OpenAI,
    embed_client: OpenAI,
    cfg: dict,
    retries: int,
    wait_seconds: float,
) -> Tuple[str, np.ndarray, bool]:
    older_node = graph.nodes[older_id]
    newer_node = graph.nodes[newer_id]
    older_text = _node_text(older_node)
    newer_text = _node_text(newer_node)
    older_ts = _node_timestamp_text(older_node)
    newer_ts = _node_timestamp_text(newer_node)

    for attempt in range(max(1, retries)):
        try:
            merged_text = _summarize_two_memories(
                llm_client=llm_client,
                llm_model=cfg["MODEL"],
                older_text=older_text,
                older_ts=older_ts,
                newer_text=newer_text,
                newer_ts=newer_ts,
            )
            merged_vec = _embed_text(embed_client, cfg["EMBEDDING_MODEL"], merged_text)
            return merged_text, merged_vec, False
        except Exception:
            if attempt < max(1, retries) - 1:
                time.sleep(wait_seconds)
            continue

    fallback_vec = _node_embedding(newer_node)
    if fallback_vec is None:
        fallback_vec = np.zeros(1, dtype=np.float32)
    return newer_text, np.asarray(fallback_vec, dtype=np.float32), True


def _merge_graph_nodes(
    graph,
    older_id: int,
    newer_id: int,
    merged_text: str,
    merged_vector: np.ndarray,
    lock: threading.Lock,
) -> int:
    with lock:
        if older_id not in graph.nodes or newer_id not in graph.nodes:
            return newer_id

        merge_ids = {older_id, newer_id}
        aggregated_edges: Dict[int, float] = defaultdict(float)
        for (src, dst), weight in list(graph.edges.items()):
            if src in merge_ids and dst not in merge_ids:
                aggregated_edges[dst] += float(weight)

        for edge in list(graph.edges.keys()):
            if edge[0] in merge_ids or edge[1] in merge_ids:
                graph.edges.pop(edge, None)

        newer_node = graph.nodes[newer_id]
        new_metadata = dict(newer_node.metadata) if isinstance(newer_node.metadata, dict) else {}
        new_metadata["contents"] = [merged_text]
        newer_node.metadata = new_metadata
        newer_node.embeddings = [np.asarray(merged_vector, dtype=np.float32).tolist()]

        for neighbor_id, weight in aggregated_edges.items():
            if neighbor_id not in graph.nodes:
                continue
            graph.edges[(newer_id, neighbor_id)] = weight
            graph.edges[(neighbor_id, newer_id)] = weight

        if older_id in graph.nodes:
            del graph.nodes[older_id]
        if hasattr(graph, "text_nodes"):
            graph.text_nodes = [node_id for node_id in graph.text_nodes if node_id != older_id]
        if hasattr(graph, "text_nodes_by_clip"):
            for clip_id in list(graph.text_nodes_by_clip.keys()):
                filtered = [node_id for node_id in graph.text_nodes_by_clip[clip_id] if node_id != older_id]
                if filtered:
                    graph.text_nodes_by_clip[clip_id] = filtered
                else:
                    del graph.text_nodes_by_clip[clip_id]
        if hasattr(graph, "event_sequence_by_clip"):
            for clip_id in list(graph.event_sequence_by_clip.keys()):
                filtered = [node_id for node_id in graph.event_sequence_by_clip[clip_id] if node_id != older_id]
                if filtered:
                    graph.event_sequence_by_clip[clip_id] = filtered
                else:
                    del graph.event_sequence_by_clip[clip_id]
        return newer_id


def _compress_cluster_in_graph(
    graph,
    cluster_id: int,
    cluster_node_ids: Sequence[int],
    llm_client: OpenAI,
    embed_client: OpenAI,
    cfg: dict,
    keep_ratio: float,
    api_retries: int,
    failure_wait_seconds: float,
    lock: threading.Lock,
) -> Dict[str, int]:
    items = [node_id for node_id in cluster_node_ids if node_id in graph.nodes]
    items.sort(key=lambda nid: (_node_timestamp_value(graph.nodes[nid], fallback=float(nid)), nid))
    before = len(items)
    if before <= 1:
        return {"cluster_id": cluster_id, "before": before, "merged": 0, "after": before, "api_failures": 0}

    target_keep = _target_keep_count(before, keep_ratio)
    merged_count = 0
    api_failures = 0

    while len(items) > target_keep:
        vectors: List[np.ndarray] = []
        emb_dim: Optional[int] = None
        raw_vectors: List[Optional[np.ndarray]] = []
        for node_id in items:
            emb = _node_embedding(graph.nodes.get(node_id))
            raw_vectors.append(emb)
            if emb is not None and emb_dim is None:
                emb_dim = emb.shape[0]
        if emb_dim is None:
            break
        for emb in raw_vectors:
            if emb is None:
                vectors.append(np.zeros(emb_dim, dtype=np.float32))
            else:
                vec = np.asarray(emb, dtype=np.float32)
                if vec.shape[0] > emb_dim:
                    vec = vec[:emb_dim]
                elif vec.shape[0] < emb_dim:
                    padded = np.zeros(emb_dim, dtype=np.float32)
                    padded[: vec.shape[0]] = vec
                    vec = padded
                vectors.append(vec)
        pair = _best_merge_pair_via_mst(np.stack(vectors, axis=0))
        if pair is None:
            break
        i, j = pair
        id_a = items[i]
        id_b = items[j]
        ts_a = _node_timestamp_value(graph.nodes[id_a], fallback=float(id_a))
        ts_b = _node_timestamp_value(graph.nodes[id_b], fallback=float(id_b))
        older_id, newer_id = (id_a, id_b) if ts_a <= ts_b else (id_b, id_a)

        merged_text, merged_vec, failed = _merge_pair_with_retry(
            graph=graph,
            older_id=older_id,
            newer_id=newer_id,
            llm_client=llm_client,
            embed_client=embed_client,
            cfg=cfg,
            retries=api_retries,
            wait_seconds=failure_wait_seconds,
        )
        if failed:
            api_failures += 1

        kept_id = _merge_graph_nodes(
            graph=graph,
            older_id=older_id,
            newer_id=newer_id,
            merged_text=merged_text,
            merged_vector=merged_vec,
            lock=lock,
        )

        for idx in sorted((i, j), reverse=True):
            items.pop(idx)
        items.append(kept_id)
        items.sort(key=lambda nid: (_node_timestamp_value(graph.nodes[nid], fallback=float(nid)), nid))
        merged_count += 1

    return {"cluster_id": cluster_id, "before": before, "merged": merged_count, "after": len(items), "api_failures": api_failures}


def _load_runtime_config(config_path: Path) -> dict:
    import json

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL", "EMBEDDING_MODEL"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"Missing required keys in {config_path}: {missing}")
    return cfg


def _append_api_failure_log(log_path: Path, video_name: str, failure_count: int) -> None:
    if failure_count <= 0:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}] video={video_name} api_failures={failure_count}\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


def compress_graph(
    graph,
    cfg: dict,
    cluster_ratio: float,
    semantic_time_alpha: float,
    time_window_size: int,
    keep_ratio: float,
    cluster_workers: int,
    api_retries: int,
    failure_wait_seconds: float,
    api_timeout_seconds: float,
) -> Tuple[object, Dict[str, int]]:
    text_node_ids = _ordered_text_node_ids(graph)
    if not text_node_ids:
        return graph, {
            "text_total_before": 0,
            "text_total_after": 0,
            "total_merged": 0,
            "total_removed": 0,
            "api_failures": 0,
            "clusters": 0,
        }

    merge_candidate_ids = [
        node_id
        for node_id in text_node_ids
        if not _is_equivalence_text_node(graph.nodes.get(node_id))
    ]
    if not merge_candidate_ids:
        if hasattr(graph, "refresh_equivalences"):
            graph.refresh_equivalences()
        text_after = len(_ordered_text_node_ids(graph))
        return graph, {
            "text_total_before": len(text_node_ids),
            "text_total_after": text_after,
            "total_merged": 0,
            "total_removed": len(text_node_ids) - text_after,
            "api_failures": 0,
            "clusters": 0,
        }

    clusters = _build_joint_clusters(
        graph=graph,
        text_node_ids=merge_candidate_ids,
        cluster_ratio=cluster_ratio,
        semantic_time_alpha=semantic_time_alpha,
        time_window_size=time_window_size,
        seed=42,
    )

    llm_client = OpenAI(
        api_key=cfg["OPENAI_API_KEY"],
        base_url=cfg["OPENAI_BASE_URL"],
        timeout=api_timeout_seconds,
    )
    embed_client = OpenAI(
        api_key=cfg["OPENAI_API_KEY"],
        base_url=cfg["OPENAI_BASE_URL"],
        timeout=api_timeout_seconds,
    )

    total_merged = 0
    total_api_failures = 0
    lock = threading.Lock()
    workers = max(1, min(cluster_workers, len(clusters)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(
                _compress_cluster_in_graph,
                graph,
                cluster_id,
                cluster_nodes,
                llm_client,
                embed_client,
                cfg,
                keep_ratio,
                api_retries,
                failure_wait_seconds,
                lock,
            ): cluster_id
            for cluster_id, cluster_nodes in clusters.items()
        }
        for fut in as_completed(future_map):
            stats = fut.result()
            total_merged += stats["merged"]
            total_api_failures += stats["api_failures"]

    if hasattr(graph, "refresh_equivalences"):
        graph.refresh_equivalences()

    text_after = len(_ordered_text_node_ids(graph))
    summary = {
        "text_total_before": len(text_node_ids),
        "text_total_after": text_after,
        "total_merged": total_merged,
        "total_removed": len(text_node_ids) - text_after,
        "api_failures": total_api_failures,
        "clusters": len(clusters),
    }
    return graph, summary


def process_graph_file(
    input_path: str,
    output_path: str,
    cfg: dict,
    cluster_ratio: float,
    semantic_time_alpha: float,
    time_window_size: int,
    keep_ratio: float,
    cluster_workers: int,
    api_retries: int,
    failure_wait_seconds: float,
    api_timeout_seconds: float,
) -> Dict[str, int]:
    with open(input_path, "rb") as f:
        graph = pickle.load(f)
    if hasattr(graph, "refresh_equivalences"):
        graph.refresh_equivalences()
    graph, summary = compress_graph(
        graph=graph,
        cfg=cfg,
        cluster_ratio=cluster_ratio,
        semantic_time_alpha=semantic_time_alpha,
        time_window_size=time_window_size,
        keep_ratio=keep_ratio,
        cluster_workers=cluster_workers,
        api_retries=api_retries,
        failure_wait_seconds=failure_wait_seconds,
        api_timeout_seconds=api_timeout_seconds,
    )
    _ensure_directory(os.path.dirname(output_path))
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_path", type=str, default="data/memory_graphs/robot")
    parser.add_argument("--compressed_mem_path", type=str, default="data/memory_graphs/robot_keep70_new")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(__file__).resolve().parent / "config.example.json",
        help="Path to config.example.json",
    )
    parser.add_argument("--cluster-ratio", type=float, default=DEFAULT_CLUSTER_RATIO)
    parser.add_argument("--semantic-time-alpha", type=float, default=DEFAULT_SEMANTIC_TIME_ALPHA)
    parser.add_argument("--time-window-size", type=int, default=DEFAULT_TIME_WINDOW_SIZE)
    parser.add_argument("--keep-ratio", type=float, default=DEFAULT_KEEP_RATIO)
    parser.add_argument("--cluster-workers", type=int, default=DEFAULT_CLUSTER_WORKERS)
    parser.add_argument("--api-retries", type=int, default=DEFAULT_API_RETRIES)
    parser.add_argument("--failure-wait-seconds", type=float, default=DEFAULT_FAILURE_WAIT_SECONDS)
    parser.add_argument("--api-timeout-seconds", type=float, default=DEFAULT_API_TIMEOUT_SECONDS)
    parser.add_argument(
        "--api-failure-log",
        type=Path,
        default=DEFAULT_API_FAILURE_LOG_PATH,
        help="Append-only txt log for per-video API failures",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not (0.0 < args.cluster_ratio <= 1.0):
        raise SystemExit("cluster-ratio must be in (0, 1]")
    if not (0.0 <= args.semantic_time_alpha <= 1.0):
        raise SystemExit("semantic-time-alpha must be in [0, 1]")
    if args.time_window_size < 0:
        raise SystemExit("time-window-size must be >= 0")
    if not (0.0 < args.keep_ratio <= 1.0):
        raise SystemExit("keep-ratio must be in (0, 1]")
    if args.cluster_workers <= 0:
        raise SystemExit("cluster-workers must be > 0")
    if args.api_retries <= 0:
        raise SystemExit("api-retries must be > 0")
    if args.failure_wait_seconds < 0:
        raise SystemExit("failure-wait-seconds must be >= 0")
    if args.api_timeout_seconds <= 0:
        raise SystemExit("api-timeout-seconds must be > 0")

    cfg = _load_runtime_config(args.config_path.expanduser().resolve())
    api_failure_log_path = args.api_failure_log.expanduser().resolve()
    file_pairs = _collect_graph_files(args.mem_path, args.compressed_mem_path)
    if not file_pairs:
        print("No memory graph files found to compress.")
        return

    mem_is_dir = os.path.isdir(args.mem_path)
    compressed_is_dir = os.path.isdir(args.compressed_mem_path)
    total_before = 0
    total_after = 0
    total_merged = 0
    total_failures = 0

    for src_path, dst_path in file_pairs:
        summary = process_graph_file(
            input_path=src_path,
            output_path=dst_path,
            cfg=cfg,
            cluster_ratio=args.cluster_ratio,
            semantic_time_alpha=args.semantic_time_alpha,
            time_window_size=args.time_window_size,
            keep_ratio=args.keep_ratio,
            cluster_workers=args.cluster_workers,
            api_retries=args.api_retries,
            failure_wait_seconds=args.failure_wait_seconds,
            api_timeout_seconds=args.api_timeout_seconds,
        )
        total_before += summary["text_total_before"]
        total_after += summary["text_total_after"]
        total_merged += summary["total_merged"]
        total_failures += summary["api_failures"]

        src_label = os.path.relpath(src_path, args.mem_path) if mem_is_dir else os.path.basename(src_path)
        dst_label = (
            os.path.relpath(dst_path, args.compressed_mem_path)
            if compressed_is_dir
            else os.path.basename(dst_path)
        )
        video_name = os.path.splitext(os.path.basename(src_path))[0]
        _append_api_failure_log(api_failure_log_path, video_name, summary["api_failures"])
        print(
            f"Compressed {src_label} -> {dst_label} | "
            f"before={summary['text_total_before']}, after={summary['text_total_after']}, "
            f"merged={summary['total_merged']}, api_failures={summary['api_failures']}, "
            f"clusters={summary['clusters']}"
        )

    print(
        f"Done. total_before={total_before}, total_after={total_after}, "
        f"total_removed={total_before - total_after}, total_merged={total_merged}, "
        f"total_api_failures={total_failures}"
    )
    gc.collect()


if __name__ == "__main__":
    main()
