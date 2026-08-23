import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reward_v4_manifest_freezes_snapshot_runtime():
    manifest = json.loads(
        (ROOT / "data/environment-v4.json").read_text(encoding="utf-8")
    )
    assert "snapshot_store.py" in manifest["runtime_files_sha256"]
    runtime = manifest["runtime_files_sha256"]
    for name in ("pack_api.py", "snapshot_store.py"):
        path = ROOT / "environments/ShopSimulator/shop_env/shop_env" / name
        # Repository manifests freeze Linux LF bytes; Windows checkouts may use CRLF.
        canonical = path.read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(canonical).hexdigest() == runtime[name]
