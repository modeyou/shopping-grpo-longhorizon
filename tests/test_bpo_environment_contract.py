import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def normalized_sha256(path):
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_reward_v4_manifest_binds_bpo_snapshot_runtime():
    manifest = json.loads((ROOT / "data/environment-v4.json").read_text(encoding="utf-8"))
    runtime = manifest["runtime_files_sha256"]
    for name in ("pack_api.py", "snapshot_store.py"):
        path = ROOT / "environments/ShopSimulator/shop_env/shop_env" / name
        assert path.is_file()
        assert runtime[name] == normalized_sha256(path)
