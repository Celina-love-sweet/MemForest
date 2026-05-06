import argparse
import re
import sqlite3
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("Merge sub mem_stores back to one big mem_store (SQLite mode).")
    p.add_argument(
        "--subs-root",
        type=Path,
        default=Path("/MemForst/Mem0/evaluation/longmemeval_sub"),
        help="Directory that contains many sub stores, e.g. mem_store_xxx(0-9), mem_store_xxx(10-19)...",
    )
    p.add_argument(
        "--dst-store",
        type=Path,
        default=Path("/MemForest/Mem0/evaluation/longmemeval_new"),
        help="Destination merged mem_store directory.",
    )
    p.add_argument("--batch", type=int, default=5000, help="Insert batch size.")
    p.add_argument(
        "--vacuum",
        action="store_true",
        default=True,
        help="Run VACUUM on destination DB at the end (default: enabled).",
    )
    p.add_argument(
        "--no-vacuum",
        action="store_false",
        dest="vacuum",
        help="Disable VACUUM at the end.",
    )
    return p.parse_args()


RANGE_RE = re.compile(r"\((\d+)-(\d+)\)$")


def extract_range_key(path: Path):
    m = RANGE_RE.search(path.name)
    if not m:
        return (10**9, 10**9, path.name)
    return (int(m.group(1)), int(m.group(2)), path.name)


def find_substores(root: Path):
    subs = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        db = p / "qdrant" / "collection" / "mem0" / "storage.sqlite"
        meta = p / "qdrant" / "meta.json"
        if db.exists() and meta.exists():
            subs.append(p)
    subs.sort(key=extract_range_key)
    return subs


def read_sqlite_settings(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("PRAGMA page_size")
    page_size = int(cur.fetchone()[0])
    cur.execute("PRAGMA auto_vacuum")
    auto_vacuum = int(cur.fetchone()[0])
    cur.execute("PRAGMA encoding")
    encoding = str(cur.fetchone()[0])
    conn.close()
    return {"page_size": page_size, "auto_vacuum": auto_vacuum, "encoding": encoding}


def prepare_destination(dst_store: Path, src_meta_json: Path, sqlite_settings):
    meta_dir = dst_store / "meta"
    qdrant_dir = dst_store / "qdrant"
    col_dir = qdrant_dir / "collection" / "mem0"
    meta_dir.mkdir(parents=True, exist_ok=True)
    col_dir.mkdir(parents=True, exist_ok=True)

    # Copy qdrant meta.json from first substore.
    (qdrant_dir / "meta.json").write_text(src_meta_json.read_text(encoding="utf-8"), encoding="utf-8")

    db_path = col_dir / "storage.sqlite"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # Align destination layout with source as much as possible.
    cur.execute(f"PRAGMA page_size={int(sqlite_settings['page_size'])}")
    cur.execute(f"PRAGMA auto_vacuum={int(sqlite_settings['auto_vacuum'])}")
    cur.execute("PRAGMA journal_mode=MEMORY")
    cur.execute("PRAGMA synchronous=OFF")
    # Keep default UTF-8 behavior; encoding pragma is effective only before table creation.
    if str(sqlite_settings.get("encoding", "")).upper() == "UTF-16":
        cur.execute("PRAGMA encoding='UTF-16'")
    cur.execute("CREATE TABLE points (id TEXT PRIMARY KEY, point BLOB)")
    conn.commit()
    return conn, cur, db_path


def main():
    args = parse_args()
    subs_root = args.subs_root.resolve()
    dst_store = args.dst_store.resolve()

    if not subs_root.exists():
        raise FileNotFoundError(f"subs root not found: {subs_root}")

    substores = find_substores(subs_root)
    if not substores:
        raise RuntimeError(f"No valid sub stores found under: {subs_root}")

    print(f"subs_root={subs_root}")
    print(f"dst_store={dst_store}")
    print(f"substores={len(substores)}")

    first_meta = substores[0] / "qdrant" / "meta.json"
    first_db = substores[0] / "qdrant" / "collection" / "mem0" / "storage.sqlite"
    sqlite_settings = read_sqlite_settings(first_db)
    dst_conn, dst_cur, dst_db_path = prepare_destination(dst_store, first_meta, sqlite_settings)

    total_scanned = 0
    total_inserted = 0
    total_duplicated = 0
    batch_rows = []

    def flush():
        nonlocal total_inserted, total_duplicated, batch_rows
        if not batch_rows:
            return
        before = dst_conn.total_changes
        dst_cur.executemany("INSERT OR IGNORE INTO points (id, point) VALUES (?, ?)", batch_rows)
        delta = dst_conn.total_changes - before
        total_inserted += delta
        total_duplicated += len(batch_rows) - delta
        batch_rows = []

    for i, sub in enumerate(substores, start=1):
        src_db = sub / "qdrant" / "collection" / "mem0" / "storage.sqlite"
        src_conn = sqlite3.connect(str(src_db))
        src_cur = src_conn.cursor()
        src_cur.execute("SELECT id, point FROM points")

        scanned_this = 0
        for rid, blob in src_cur:
            total_scanned += 1
            scanned_this += 1
            batch_rows.append((rid, blob))
            if len(batch_rows) >= args.batch:
                flush()

        src_conn.close()
        flush()
        dst_conn.commit()
        print(
            f"[{i}/{len(substores)}] {sub.name}: scanned={scanned_this}, "
            f"total_scanned={total_scanned}, inserted={total_inserted}, duplicated={total_duplicated}",
            flush=True,
        )

    dst_conn.commit()
    if args.vacuum:
        print("Running VACUUM on destination...")
        dst_cur.execute("VACUUM")
        dst_conn.commit()
    dst_conn.close()

    # Cleanup any sidecar files that may remain after SQLite operations.
    for suffix in ("-journal", "-wal", "-shm"):
        sidecar = Path(str(dst_db_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()

    print("Done.")
    print(f"destination_db={dst_db_path}")
    print(f"total_scanned={total_scanned}")
    print(f"total_inserted={total_inserted}")
    print(f"total_duplicated={total_duplicated}")


if __name__ == "__main__":
    main()
