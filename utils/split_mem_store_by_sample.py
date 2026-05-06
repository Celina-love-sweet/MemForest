
import argparse
import copy
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models


USER_SUFFIX_RE = re.compile(r"_(\d+)$")
DEFAULT_SRC_STORE = Path("/MemForest/Mem0/evaluation/longmemeval")
DEFAULT_DST_ROOT = Path("/MemForest/Mem0/evaluation/longmemeval_sub")


def normalize_store_root(path: Path) -> Path:
    """
    Accept:
      - <mem_store_root>                 (contains qdrant/)
      - <mem_store_root>/qdrant
    Return:
      - <mem_store_root>
    """
    p = path.resolve()
    if (p / "qdrant" / "collection").exists():
        return p
    if p.name == "qdrant" and (p / "collection").exists():
        return p.parent
    raise ValueError("Invalid source store path. Expect mem_store root or its qdrant subdir.")


def parse_idx_from_user_id(user_id: str) -> Optional[int]:
    if not isinstance(user_id, str):
        return None
    m = USER_SUFFIX_RE.search(user_id.strip())
    if not m:
        return None
    return int(m.group(1))


def sample_group_ranges(start_idx: int, end_idx: int, group_size: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    s = start_idx
    while s <= end_idx:
        e = min(s + group_size - 1, end_idx)
        ranges.append((s, e))
        s = e + 1
    return ranges


def idx_to_group(idx: int, start_idx: int, end_idx: int, group_size: int) -> Optional[Tuple[int, int]]:
    if idx < start_idx or idx > end_idx:
        return None
    relative = idx - start_idx
    group_no = relative // group_size
    s = start_idx + group_no * group_size
    e = min(s + group_size - 1, end_idx)
    return (s, e)


class GroupWriter:
    def __init__(
        self,
        store_root: Path,
        collection_name: str,
        vectors_config,
        sparse_vectors_config,
        on_disk_payload,
        overwrite: bool,
        upsert_batch: int,
    ):
        self.store_root = store_root
        self.collection_name = collection_name
        self.upsert_batch = upsert_batch
        self.buffer: List[models.PointStruct] = []
        self.written = 0

        self.meta_dir = self.store_root / "meta"
        self.qdrant_dir = self.store_root / "qdrant"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_dir.mkdir(parents=True, exist_ok=True)

        self.client = QdrantClient(path=str(self.qdrant_dir))

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            if overwrite:
                self.client.delete_collection(collection_name=self.collection_name)
            else:
                raise RuntimeError(
                    "Collection already exists in destination: "
                    f"{self.qdrant_dir / 'collection' / self.collection_name}"
                )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=copy.deepcopy(vectors_config),
            sparse_vectors_config=copy.deepcopy(sparse_vectors_config),
            on_disk_payload=on_disk_payload,
        )

    def add_point(self, point_id, vector, payload):
        p = models.PointStruct(id=point_id, vector=vector, payload=payload if isinstance(payload, dict) else {})
        self.buffer.append(p)
        if len(self.buffer) >= self.upsert_batch:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        self.client.upsert(collection_name=self.collection_name, points=self.buffer, wait=True)
        self.written += len(self.buffer)
        self.buffer = []

    def close(self):
        self.flush()
        self.client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split local Qdrant memory store by sample index ranges.")
    parser.add_argument(
        "--src-store",
        type=Path,
        default=DEFAULT_SRC_STORE,
        help="Source mem_store root or source qdrant path.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=DEFAULT_DST_ROOT,
        help="Destination parent directory for sub-stores.",
    )
    parser.add_argument("--collection", type=str, default="mem0", help="Collection name in Qdrant.")
    parser.add_argument("--start-idx", type=int, default=0, help="Start sample index (inclusive).")
    parser.add_argument("--end-idx", type=int, default=499, help="End sample index (inclusive).")
    parser.add_argument("--group-size", type=int, default=10, help="Samples per sub-store.")
    parser.add_argument("--scroll-batch", type=int, default=1024, help="Source scroll batch size.")
    parser.add_argument("--upsert-batch", type=int, default=512, help="Destination upsert batch size per sub-store.")
    parser.add_argument(
        "--compact",
        action="store_true",
        default=True,
        help="Run sqlite checkpoint+VACUUM for each output sub-store (default: enabled).",
    )
    parser.add_argument(
        "--no-compact",
        action="store_false",
        dest="compact",
        help="Disable sqlite compaction for output sub-stores.",
    )
    parser.add_argument(
        "--src-name",
        type=str,
        default=None,
        help="Base name for destination store folders. Default: source store folder name.",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite existing collection in destination sub-stores.",
    )
    return parser.parse_args()


def compact_store_sqlite(store_root: Path, collection_name: str) -> None:
    db_file = store_root / "qdrant" / "collection" / collection_name / "storage.sqlite"
    if not db_file.exists():
        return
    conn = sqlite3.connect(str(db_file), timeout=120)
    try:
        conn.execute("PRAGMA busy_timeout = 120000;")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.execute("VACUUM;")
        conn.commit()
    finally:
        conn.close()
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(db_file) + suffix)
        if sidecar.exists():
            sidecar.unlink()


def main():
    args = parse_args()
    if args.group_size <= 0:
        raise ValueError("--group-size must be > 0")
    if args.start_idx > args.end_idx:
        raise ValueError("--start-idx must be <= --end-idx")

    src_store_root = normalize_store_root(args.src_store)
    src_qdrant_root = src_store_root / "qdrant"
    if not (src_qdrant_root / "collection").exists():
        raise FileNotFoundError(f"Source qdrant collection folder not found: {src_qdrant_root / 'collection'}")

    dst_root = args.dst_root.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    src_name = args.src_name or src_store_root.name

    group_ranges = sample_group_ranges(args.start_idx, args.end_idx, args.group_size)
    print(f"Source store: {src_store_root}")
    print(f"Destination root: {dst_root}")
    print(f"Sample range: {args.start_idx}-{args.end_idx}, group_size={args.group_size}, groups={len(group_ranges)}")

    src_client = QdrantClient(path=str(src_qdrant_root))
    try:
        existing = [c.name for c in src_client.get_collections().collections]
        if args.collection not in existing:
            raise RuntimeError(
                f"Collection '{args.collection}' not found in source. Available: {', '.join(existing) or '<none>'}"
            )

        src_info = src_client.get_collection(args.collection)
        vectors_config = src_info.config.params.vectors
        sparse_vectors_config = src_info.config.params.sparse_vectors
        on_disk_payload = src_info.config.params.on_disk_payload

        writers: Dict[Tuple[int, int], GroupWriter] = {}
        for s, e in group_ranges:
            out_store = dst_root / f"{src_name}({s}-{e})"
            writers[(s, e)] = GroupWriter(
                store_root=out_store,
                collection_name=args.collection,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                on_disk_payload=on_disk_payload,
                overwrite=args.overwrite,
                upsert_batch=args.upsert_batch,
            )

        total_points = 0
        copied_points = 0
        skipped_no_user_id = 0
        skipped_out_of_range = 0

        # Fast path for local mode:
        # avoid repeated src_client.scroll() because each local scroll sorts ids again.
        local_backend = getattr(src_client, "_client", None)
        get_collection = getattr(local_backend, "_get_collection", None)
        if callable(get_collection):
            local_col = get_collection(args.collection)
            sorted_ids = sorted(local_col.ids.items(), key=lambda x: local_col._universal_id(x[0]))
            for point_id, internal_idx in sorted_ids:
                if bool(local_col.deleted[internal_idx]):
                    continue
                total_points += 1
                payload = local_col._get_payload(internal_idx, True)
                if not isinstance(payload, dict):
                    payload = {}
                user_id = payload.get("user_id")
                idx = parse_idx_from_user_id(user_id) if isinstance(user_id, str) else None
                if idx is None:
                    skipped_no_user_id += 1
                    continue

                group = idx_to_group(idx, args.start_idx, args.end_idx, args.group_size)
                if group is None:
                    skipped_out_of_range += 1
                    continue

                vector = local_col._get_vectors(internal_idx, True)
                writers[group].add_point(point_id=point_id, vector=vector, payload=payload)
                copied_points += 1

                if total_points % 20000 == 0:
                    print(
                        f"Scanned={total_points}, copied={copied_points}, "
                        f"no_user_id={skipped_no_user_id}, out_of_range={skipped_out_of_range}",
                        flush=True,
                    )
        else:
            # Fallback for non-local backends.
            offset = None
            while True:
                points, offset = src_client.scroll(
                    collection_name=args.collection,
                    limit=args.scroll_batch,
                    with_payload=True,
                    with_vectors=True,
                    offset=offset,
                )
                if not points:
                    break

                for p in points:
                    total_points += 1
                    payload = p.payload if isinstance(p.payload, dict) else {}
                    user_id = payload.get("user_id")
                    idx = parse_idx_from_user_id(user_id) if isinstance(user_id, str) else None
                    if idx is None:
                        skipped_no_user_id += 1
                        continue

                    group = idx_to_group(idx, args.start_idx, args.end_idx, args.group_size)
                    if group is None:
                        skipped_out_of_range += 1
                        continue

                    writers[group].add_point(point_id=p.id, vector=p.vector, payload=payload)
                    copied_points += 1

                if total_points % 20000 == 0:
                    print(
                        f"Scanned={total_points}, copied={copied_points}, "
                        f"no_user_id={skipped_no_user_id}, out_of_range={skipped_out_of_range}",
                        flush=True,
                    )

        for writer in writers.values():
            writer.close()
        if args.compact:
            for s, e in group_ranges:
                out_store = dst_root / f"{src_name}({s}-{e})"
                compact_store_sqlite(out_store, args.collection)

        print("Done.")
        print(
            f"Scanned={total_points}, copied={copied_points}, "
            f"no_user_id={skipped_no_user_id}, out_of_range={skipped_out_of_range}"
        )
        print("Per-group copied points:")
        for s, e in group_ranges:
            n = writers[(s, e)].written
            print(f"  {s:03d}-{e:03d}: {n}")
    finally:
        src_client.close()


if __name__ == "__main__":
    main()
