from types import SimpleNamespace

from shopping_grpo.training.bpo.session import ClonedBranchSession
from shopping_grpo.training.grpo.adapter.shopper import ControlledShopper


class FakeClient:
    def complete(self, messages, tools):
        return {"content": '{"answer":"蓝色","used_facts":["蓝色"]}'}


class _DerivedClickableGraph:
    def __deepcopy__(self, memo):
        raise RecursionError("derived clickable graph must not be copied")


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
        text_to_clickable=_DerivedClickableGraph(),
    )
    store = SnapshotStore()
    snapshot_id = store.create(source, 0)

    def target():
        value = SimpleNamespace(server=server, browser=SimpleNamespace())

        def rebuild():
            value.text_to_clickable = {"rebuilt": value.browser.current_url}

        value.get_available_actions = rebuild
        return value

    targets = [target() for _ in range(3)]
    results = [
        store.clone_into(snapshot_id, target, index)
        for index, target in enumerate(targets, start=1)
    ]
    assert [result["env_idx"] for result in results] == [1, 2, 3]
    assert [target.session for target in targets] == [
        "slot-1-abc", "slot-2-abc", "slot-3-abc"
    ]
    assert targets[0].browser.current_url == "http://shop/slot-1-abc/item"
    assert targets[0].text_to_clickable == {
        "rebuilt": "http://shop/slot-1-abc/item"
    }
    targets[0].history.append("click")
    assert source.history == ["search"]
    assert targets[1].history == ["search"]
    targets[0].server.user_sessions[targets[0].session]["cart"].append(2)
    assert targets[1].server.user_sessions[targets[1].session]["cart"] == [1]
    assert store.drop(snapshot_id) is True
    assert store.drop(snapshot_id) is False
