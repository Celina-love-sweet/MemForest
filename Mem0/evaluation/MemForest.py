

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sqlite3
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

PROMPT_TEMPLATE = """
You are an intelligent memory summarization assistant. Please extract the core information from the following two memory entries and their timestamps, and generate a concise and coherent summary paragraph. If the memories contain relative time expressions (e.g., “last June,” “two months ago”), convert them into specific dates based on the corresponding timestamps (for example, if a memory mentions “last June” and the timestamp is February 2023, the actual time should be June 2022). The summary length should be kept within the combined length of the two original memories as much as possible. Output only the final summarized result, without any explanations, analysis, reasoning steps, or additional content.
Memory 1: {}, Timestamp 1: {}
Memory 2: {}, Timestamp 2: {}
"""

DEFAULT_MEM_STORE_PATH = Path("/MemForest/Mem0/evaluation/locomo")
DEFAULT_CONFIG_PATH = Path("/MemForest/Mem0/evaluation/config.example.json")

@dataclass
class MemoryNode:
    point_id: object
    payload: dict
    vector: np.ndarray
    vector_name: str | None
    created_at: datetime
    changed: bool = False

    @property
    def text(self) -> str:
        value = self.payload.get("data", "")
        return value if isinstance(value, str) else str(value)

    @property
    def timestamp_text(self) -> str:
        value = self.payload.get("timestamp", "")
        return value if isinstance(value, str) else str(value)


@dataclass
class ClusterResult:
    cluster_id: int
    before: int
    merged: int
    final_nodes: List[MemoryNode]
    api_failures: int


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, default="mem0", help="Qdrant collection name")
    parser.add_argument("--start-idx", type=int, default=0, help="Start sample idx (inclusive)")
    parser.add_argument("--end-idx", type=int, default=9, help="End sample idx (inclusive)")
    parser.add_argument("--cluster-ratio", type=float, default=0.05, help="KMeans cluster ratio per sample")
    parser.add_argument(
        "--semantic-time-alpha",
        type=float,
        default=0.8,
        help="Importance balance alpha. score=alpha*semantic+(1-alpha)*time",
    )
    parser.add_argument(
        "--time-window-size",
        type=int,
        default=5,
        help="Time window size on both sides of each center node",
    )
    parser.add_argument("--keep-ratio", type=float, default=0.7, help="Keep ratio per final cluster")
    parser.add_argument("--cluster-workers", type=int, default=45, help="Concurrent cluster workers per sample")
    parser.add_argument("--api-retries", type=int, default=3, help="Retries before abandoning a merge API call")
    parser.add_argument(
        "--failure-wait-seconds",
        type=float,
        default=15.0,
        help="Wait between merge API retries after failure",
    )
    parser.add_argument(
        "--api-timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout (seconds) for each LLM/embedding API request",
    )
    parser.add_argument("--batch", type=int, default=256, help="Delete/upsert batch size")
    parser.add_argument("--mem-store-path", type=Path, default=DEFAULT_MEM_STORE_PATH, help="Path to mem_store")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.example.json")
    parser.add_argument("--dry-run", action="store_true", help="Only print stats, do not write changes")
    parser.add_argument("--no-compact", action="store_true", help="Skip sqlite VACUUM at end")
    return parser.parse_args()


def load_runtime_config(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    required = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL", "EMBEDDING_MODEL"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise SystemExit(f"Missing required keys in {config_path}: {missing}")
    return cfg


def extract_user_id(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("user_id", "userId", "uid"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("user_id", "userId", "uid"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def parse_idx_from_user_id(user_id: str) -> int | None:
    m = re.search(r"_(\d+)$", user_id.strip())
    return int(m.group(1)) if m else None


def to_dense_vector(vector_obj: object) -> tuple[np.ndarray | None, str | None]:
    if isinstance(vector_obj, list):
        return np.asarray(vector_obj, dtype=np.float32), None
    if isinstance(vector_obj, dict) and vector_obj:
        first_key = next(iter(vector_obj.keys()))
        first_value = vector_obj[first_key]
        if isinstance(first_value, list):
            return np.asarray(first_value, dtype=np.float32), str(first_key)
    return None, None


def parse_memory_time(payload: dict) -> datetime:
    raw_ts = payload.get("timestamp")
    if isinstance(raw_ts, str) and raw_ts.strip():
        ts = re.sub(r"\b(am|pm)\b", lambda m: m.group(1).upper(), raw_ts.strip(), flags=re.IGNORECASE)
        for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
            try:
                return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    raw_created = payload.get("created_at")
    if isinstance(raw_created, str):
        try:
            dt = datetime.fromisoformat(raw_created.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)


def load_nodes_by_idx(client: QdrantClient, collection: str, batch: int = 512) -> Dict[int, List[MemoryNode]]:
    groups: Dict[int, List[MemoryNode]] = defaultdict(list)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )
        if not points:
            break
        for p in points:
            uid = extract_user_id(p.payload)
            if not uid:
                continue
            idx = parse_idx_from_user_id(uid)
            if idx is None:
                continue
            vec, vector_name = to_dense_vector(p.vector)
            if vec is None:
                continue
            payload = dict(p.payload) if isinstance(p.payload, dict) else {}
            groups[idx].append(
                MemoryNode(
                    point_id=p.id,
                    payload=payload,
                    vector=vec,
                    vector_name=vector_name,
                    created_at=parse_memory_time(payload),
                )
            )
        if offset is None:
            break
    return groups


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def kmeans_labels(vectors: np.ndarray, k: int, seed: int = 42, max_iter: int = 30) -> np.ndarray:
    n, d = vectors.shape
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int32)

    rng = np.random.default_rng(seed)
    centroids = np.empty((k, d), dtype=np.float32)

    first = int(rng.integers(0, n))
    centroids[0] = vectors[first]

    closest_dist2 = np.sum((vectors - centroids[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(np.sum(closest_dist2))
        if total <= 0.0:
            idx = int(rng.integers(0, n))
        else:
            probs = closest_dist2 / total
            idx = int(rng.choice(n, p=probs))
        centroids[c] = vectors[idx]
        dist2 = np.sum((vectors - centroids[c]) ** 2, axis=1)
        closest_dist2 = np.minimum(closest_dist2, dist2)

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dist_mat = np.sum((vectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist_mat, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centroids[c] = vectors[mask].mean(axis=0)
            else:
                farthest = int(np.argmax(np.min(dist_mat, axis=1)))
                centroids[c] = vectors[farthest]

    return labels


def summarize_memory(llm_client: OpenAI, llm_model: str, older: MemoryNode, newer: MemoryNode) -> str:
    prompt = PROMPT_TEMPLATE.format(older.text, older.timestamp_text, newer.text, newer.timestamp_text)
    resp = llm_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip() or newer.text


def embed_text(embed_client: OpenAI, embedding_model: str, text: str) -> np.ndarray:
    resp = embed_client.embeddings.create(model=embedding_model, input=[text])
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def merge_pair_with_retry(
    older: MemoryNode,
    newer: MemoryNode,
    llm_client: OpenAI,
    embed_client: OpenAI,
    cfg: dict,
    retries: int,
    wait_seconds: float,
) -> tuple[MemoryNode, bool]:
    for attempt in range(max(1, retries)):
        try:
            merged_text = summarize_memory(llm_client, cfg["MODEL"], older, newer)
            merged_vec = embed_text(embed_client, cfg["EMBEDDING_MODEL"], merged_text)
            payload = dict(newer.payload)
            payload["data"] = merged_text
            merged = MemoryNode(
                point_id=newer.point_id,
                payload=payload,
                vector=merged_vec,
                vector_name=newer.vector_name,
                created_at=newer.created_at,
                changed=True,
            )
            return merged, False
        except Exception:
            if attempt < max(1, retries) - 1:
                time.sleep(wait_seconds)
            continue
    return MemoryNode(
        point_id=newer.point_id,
        payload=dict(newer.payload),
        vector=newer.vector,
        vector_name=newer.vector_name,
        created_at=newer.created_at,
        changed=newer.changed,
    ), True


def target_keep_count(size: int, keep_ratio: float) -> int:
    if size <= 1:
        return size
    return max(1, min(size, int(math.ceil(size * keep_ratio))))


def best_merge_pair_via_mst(nodes: List[MemoryNode]) -> tuple[int, int] | None:
    n = len(nodes)
    if n < 2:
        return None
    edges: List[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(nodes[i].vector, nodes[j].vector)
            edges.append((sim, i, j))
    edges.sort(key=lambda x: x[0], reverse=True)
    dsu = DisjointSet(n)
    mst: List[tuple[float, int, int]] = []
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

    def edge_score(item: tuple[float, int, int]) -> float:
        sim, u, v = item
        degree_norm = (degree[u] + degree[v]) / max_degree_sum
        return 0.90 * sim - 0.10 * degree_norm

    _, i, j = max(mst, key=edge_score)
    return i, j


def build_joint_clusters(
    sample_nodes: List[MemoryNode],
    cluster_ratio: float,
    semantic_time_alpha: float,
    time_window_size: int,
    seed: int,
) -> Dict[int, List[MemoryNode]]:
    n = len(sample_nodes)
    if n == 0:
        return {}

    vectors = np.stack([node.vector for node in sample_nodes], axis=0)
    k = max(1, min(n, int(math.floor(cluster_ratio * n))))
    labels = kmeans_labels(vectors, k=k, seed=seed)

    semantic_clusters: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        semantic_clusters[label].append(idx)

    cluster_ids = sorted(semantic_clusters.keys())
    center_index_by_cluster: Dict[int, int] = {}
    for cid in cluster_ids:
        indices = semantic_clusters[cid]
        cluster_vectors = vectors[indices]
        centroid = cluster_vectors.mean(axis=0)
        dist2 = np.sum((cluster_vectors - centroid) ** 2, axis=1)
        center_local = int(np.argmin(dist2))
        center_index_by_cluster[cid] = indices[center_local]

    center_indices = set(center_index_by_cluster.values())
    assigned: Dict[int, List[int]] = {cid: [center_index_by_cluster[cid]] for cid in cluster_ids}

    ordered_indices = sorted(range(n), key=lambda i: (sample_nodes[i].created_at, i))
    time_pos = {idx: pos for pos, idx in enumerate(ordered_indices)}
    center_items = sorted(center_index_by_cluster.items())

    for idx in ordered_indices:
        if idx in center_indices:
            continue
        original_cluster = int(labels[idx])
        best_score = None
        best_clusters: List[int] = []
        for cid, center_idx in center_items:
            semantic_flag = cosine_similarity(vectors[idx], vectors[center_idx])
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

    result: Dict[int, List[MemoryNode]] = {}
    for cid in cluster_ids:
        indices = assigned.get(cid, [])
        if not indices:
            continue
        result[cid] = [sample_nodes[i] for i in indices]
    return result


def compress_cluster(
    cluster_id: int,
    nodes: List[MemoryNode],
    cfg: dict,
    keep_ratio: float,
    api_retries: int,
    failure_wait_seconds: float,
    api_timeout_seconds: float,
) -> ClusterResult:
    items = sorted(nodes, key=lambda n: n.created_at)
    before = len(items)
    target_keep = target_keep_count(before, keep_ratio)
    merged = 0
    api_failures = 0
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
    while len(items) > target_keep:
        pair = best_merge_pair_via_mst(items)
        if pair is None:
            break
        i, j = pair
        first = items[i]
        second = items[j]
        older, newer = (first, second) if first.created_at <= second.created_at else (second, first)
        merged_node, failed = merge_pair_with_retry(
            older=older,
            newer=newer,
            llm_client=llm_client,
            embed_client=embed_client,
            cfg=cfg,
            retries=api_retries,
            wait_seconds=failure_wait_seconds,
        )
        if failed:
            api_failures += 1
        for idx in sorted((i, j), reverse=True):
            items.pop(idx)
        items.append(merged_node)
        items.sort(key=lambda n: n.created_at)
        merged += 1
    return ClusterResult(
        cluster_id=cluster_id,
        before=before,
        merged=merged,
        final_nodes=items,
        api_failures=api_failures,
    )


def batched(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def delete_points(client: QdrantClient, collection: str, ids: Sequence, batch: int) -> None:
    for chunk in batched(list(ids), batch):
        client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=list(chunk)),
        )


def upsert_points(client: QdrantClient, collection: str, nodes: Sequence[MemoryNode], batch: int) -> None:
    for chunk in batched(list(nodes), batch):
        points = []
        for node in chunk:
            vec = node.vector.tolist()
            vector_data: list[float] | dict[str, list[float]]
            if node.vector_name:
                vector_data = {node.vector_name: vec}
            else:
                vector_data = vec
            points.append(models.PointStruct(id=node.point_id, vector=vector_data, payload=node.payload))
        client.upsert(collection_name=collection, points=points)


def sqlite_file_paths(qdrant_path: Path, collection: str) -> List[Path]:
    db = qdrant_path / "collection" / collection / "storage.sqlite"
    return [db, Path(str(db) + "-wal"), Path(str(db) + "-shm")]


def total_size_bytes(paths: Sequence[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.exists())


def format_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def main() -> None:
    args = parse_args()
    if args.start_idx > args.end_idx:
        raise SystemExit("start-idx must be <= end-idx")
    if not (0.0 < args.cluster_ratio <= 1.0):
        raise SystemExit("cluster-ratio must be in (0, 1]")
    if not (0.0 <= args.semantic_time_alpha <= 1.0):
        raise SystemExit("semantic-time-alpha must be in [0, 1]")
    if args.time_window_size < 0:
        raise SystemExit("time-window-size must be >= 0")
    if not (0.0 <= args.keep_ratio <= 1.0):
        raise SystemExit("keep-ratio must be in [0, 1]")
    if args.cluster_workers <= 0:
        raise SystemExit("cluster-workers must be > 0")
    if args.api_retries <= 0:
        raise SystemExit("api-retries must be > 0")
    if args.failure_wait_seconds < 0:
        raise SystemExit("failure-wait-seconds must be >= 0")
    if args.api_timeout_seconds <= 0:
        raise SystemExit("api-timeout-seconds must be > 0")

    mem_store_path = args.mem_store_path.expanduser().resolve()
    config_path = args.config_path.expanduser().resolve()
    qdrant_path = (mem_store_path / "qdrant").resolve()
    if not (qdrant_path / "collection").exists():
        raise SystemExit(f"Qdrant path not found or invalid: {qdrant_path}")
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    cfg = load_runtime_config(config_path)

    client = QdrantClient(path=str(qdrant_path))
    collections = {c.name for c in client.get_collections().collections}
    if args.collection not in collections:
        raise SystemExit(
            f"Collection '{args.collection}' not found. Available: {', '.join(sorted(collections)) or '<none>'}"
        )

    sqlite_paths = sqlite_file_paths(qdrant_path, args.collection)
    size_before = total_size_bytes(sqlite_paths)
    print(f"Memory store size (before compression): {format_mb(size_before)}")

    groups = load_nodes_by_idx(client, args.collection, batch=args.batch)
    if not groups:
        raise SystemExit("No memory nodes with idx suffix found in payload user_id.")

    for idx in range(args.start_idx, args.end_idx + 1):
        sample_nodes = groups.get(idx, [])
        if not sample_nodes:
            print(f"Sample {idx}: nodes=0, clusters=0, expected_after=0\n")
            continue

        clusters = build_joint_clusters(
            sample_nodes=sample_nodes,
            cluster_ratio=args.cluster_ratio,
            semantic_time_alpha=args.semantic_time_alpha,
            time_window_size=args.time_window_size,
            seed=42 + idx,
        )
        expected_after = sum(target_keep_count(len(nodes), args.keep_ratio) for nodes in clusters.values())
        print(f"Sample {idx}: nodes={len(sample_nodes)}, clusters={len(clusters)}, expected_after={expected_after}")

        cluster_workers = max(1, min(args.cluster_workers, len(clusters)))
        cluster_results: List[ClusterResult] = []
        with ThreadPoolExecutor(max_workers=cluster_workers) as ex:
            future_map = {
                ex.submit(
                    compress_cluster,
                    cid,
                    cluster_nodes,
                    cfg,
                    args.keep_ratio,
                    args.api_retries,
                    args.failure_wait_seconds,
                    args.api_timeout_seconds,
                ): cid
                for cid, cluster_nodes in clusters.items()
            }
            for fut in as_completed(future_map):
                cid = future_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    raise RuntimeError(f"Sample {idx}, cluster {cid} failed: {e}") from e
                cluster_results.append(res)

        cluster_results.sort(key=lambda res: res.cluster_id)
        for res in cluster_results:
            print(
                f"Sample {idx}, Cluster {res.cluster_id}: before={res.before}, merged={res.merged}, "
                f"after={len(res.final_nodes)}, api_failures={res.api_failures}"
            )

        final_nodes = [node for result in cluster_results for node in result.final_nodes]
        final_ids = {n.point_id for n in final_nodes}
        original_ids = {n.point_id for n in sample_nodes}
        delete_ids = sorted(original_ids - final_ids, key=str)
        update_nodes = [n for n in final_nodes if n.changed]

        print(f"Sample {idx}: actual_after={len(final_nodes)}\n")

        if args.dry_run:
            continue

        if delete_ids:
            delete_points(client, args.collection, delete_ids, batch=args.batch)
        if update_nodes:
            upsert_points(client, args.collection, update_nodes, batch=args.batch)

    if args.dry_run or args.no_compact:
        if hasattr(client, "close"):
            client.close()
        size_after_no_compact = total_size_bytes(sqlite_paths)
        print(f"Memory store size (after compression): {format_mb(size_after_no_compact)}")
        return

    if hasattr(client, "close"):
        client.close()
    del client
    gc.collect()

    db_file = sqlite_paths[0]
    if db_file.exists():
        conn = sqlite3.connect(str(db_file), timeout=120)
        try:
            conn.execute("PRAGMA busy_timeout = 120000;")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.execute("VACUUM;")
            conn.commit()
        finally:
            conn.close()

    size_after = total_size_bytes(sqlite_paths)
    print(f"Memory store size (after compression): {format_mb(size_after)}")


if __name__ == "__main__":
    main()
