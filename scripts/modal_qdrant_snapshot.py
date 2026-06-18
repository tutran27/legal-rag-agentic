from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "legal-rag-qdrant-snapshot"
VOLUME_NAME = "legal-rag-ingest-data"
REMOTE_DIR = "/qdrant_snapshots"


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    volumes={"/data": volume},
    timeout=30 * 60,
)
def list_snapshots() -> list[dict]:
    snapshots = []
    root = Path("/data") / REMOTE_DIR.strip("/")
    if not root.exists():
        return snapshots
    for path in sorted(root.glob("*.snapshot")):
        snapshots.append(
            {
                "name": path.name,
                "remote_path": f"{REMOTE_DIR}/{path.name}",
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            }
        )
    return snapshots


@app.function(
    volumes={"/data": volume},
    timeout=30 * 60,
)
def remove_snapshot(snapshot_name: str) -> str:
    if "/" in snapshot_name or "\\" in snapshot_name:
        raise ValueError("snapshot_name chỉ được là tên file.")
    path = Path("/data") / REMOTE_DIR.strip("/") / snapshot_name
    path.unlink(missing_ok=True)
    volume.commit()
    return f"Đã xóa {REMOTE_DIR}/{snapshot_name}"


@app.local_entrypoint()
def main(
    action: str = "upload",
    snapshot: str = "",
    remote_name: str = "",
):
    if action == "upload":
        if not snapshot:
            raise ValueError("Cần truyền --snapshot path/to/file.snapshot")
        local_path = Path(snapshot)
        if not local_path.exists():
            raise FileNotFoundError(local_path)
        target_name = remote_name or local_path.name
        if not target_name.endswith(".snapshot"):
            target_name = f"{target_name}.snapshot"
        remote_path = f"{REMOTE_DIR}/{target_name}"
        with volume.batch_upload(force=True) as upload:
            upload.put_file(str(local_path), remote_path)
        print(f"Đã upload {local_path} lên {VOLUME_NAME}:{remote_path}")
        return

    if action == "list":
        for item in list_snapshots.remote():
            print(
                f"{item['remote_path']} - {item['size_mb']} MB"
            )
        return

    if action == "remove":
        if not remote_name:
            raise ValueError("Cần truyền --remote-name snapshot_name")
        print(remove_snapshot.remote(remote_name))
        return

    raise ValueError("action phải là upload, list hoặc remove.")
