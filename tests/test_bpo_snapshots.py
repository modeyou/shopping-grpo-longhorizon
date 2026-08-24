from types import SimpleNamespace

from shopping_grpo.training.bpo.session import ClonedBranchSession
from shopping_grpo.training.grpo.adapter.shopper import ControlledShopper


class FakeClient:
    def complete(self, messages, tools):
        return {"content": '{"answer":"蓝色","used_facts":["蓝色"]}'}


def test_controlled_shopper_clone_has_independent_history():
    shopper = ControlledShopper(
        FakeClient(), initial_request="买一个", allowed_facts=["蓝色"], max_questions=2
    )
    shopper.answer("什么颜色？")
    cloned = shopper.clone()
    cloned.history[0]["answer"] = "changed"
    assert shopper.history[0]["answer"] == "蓝色"
    assert cloned.call_count == shopper.call_count


def test_snapshot_store_clones_server_and_browser_state():
    import sys
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parents[1]
        / "environments/ShopSimulator/shop_env/shop_env"
    )
    sys.path.insert(0, str(module_path))
    try:
        from snapshot_store import SnapshotStore
    finally:
        sys.path.remove(str(module_path))
    server = SimpleNamespace(user_sessions={"slot-0-abc": {"done": False, "cart": [1]}})
    source = SimpleNamespace(
        session="slot-0-abc",
        server=server,
        browser=SimpleNamespace(
            session_id="slot-0-abc",
            current_url="http://shop/slot-0-abc/item",
            page_source="slot-0-abc page",
        ),
        idx=3,
        history=["search"],
    )
    target = SimpleNamespace(
        server=server,
        browser=SimpleNamespace(),
        get_available_actions=lambda: None,
    )
    store = SnapshotStore()
    snapshot_id = store.create(source, 0)
    result = store.clone_into(snapshot_id, target, 1)
    assert result["env_idx"] == 1
    assert target.session == "slot-1-abc"
    assert target.browser.current_url == "http://shop/slot-1-abc/item"
    target.history.append("click")
    assert source.history == ["search"]
    assert store.drop(snapshot_id) is True
    assert store.drop(snapshot_id) is False
