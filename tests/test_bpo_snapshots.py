from types import SimpleNamespace

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.training.grpo.adapter.shopper import ControlledShopper


def test_environment_client_snapshot_clone_drop_lifecycle():
    calls = []

    def transport(_endpoint, payload, _timeout):
        calls.append(payload)
        if payload["action"] == "snapshot":
            return {"result": {"snapshot_id": "opaque"}}
        if payload["action"] == "clone":
            return {"result": {"env_idx": 7, "done": False}}
        if payload["action"] == "drop_snapshot":
            return {"result": {"dropped": True}}
        raise AssertionError(payload)

    env = ShopAgentEnv(transport=transport, multiturn=True)
    env.env_idx = 2
    snapshot_id = env.snapshot()
    clone = env.clone(snapshot_id)
    assert clone.env_idx == 7
    assert clone.multiturn is True
    assert env.drop_snapshot(snapshot_id)["dropped"] is True
    assert calls == [
        {"action": "snapshot", "env_idx": 2},
        {"action": "clone", "snapshot_id": "opaque"},
        {"action": "drop_snapshot", "snapshot_id": "opaque"},
    ]


def test_server_snapshot_is_deep_copied_and_clone_has_independent_state():
    from environments.ShopSimulator.shop_env.shop_env.snapshot_store import SnapshotStore

    server = SimpleNamespace(user_sessions={"slot-0-42": {"done": False, "cart": ["a"]}})
    source = SimpleNamespace(
        session="slot-0-42",
        server=server,
        browser=SimpleNamespace(
            session_id="slot-0-42",
            current_url="http://shop/slot-0-42/item",
            page_source="slot-0-42 page",
        ),
        history=["first"],
        prev_obs=["obs"],
        get_available_actions=lambda: None,
    )
    target = SimpleNamespace(
        session=None,
        server=server,
        browser=SimpleNamespace(session_id=None, current_url="", page_source=""),
        get_available_actions=lambda: None,
    )
    store = SnapshotStore()
    snapshot_id = store.create(source, 0)
    source.history.append("after")
    server.user_sessions["slot-0-42"]["cart"].append("b")
    result = store.clone_into(snapshot_id, target, 3)
    assert result["env_idx"] == 3
    assert target.session == "slot-3-42"
    assert target.history == ["first"]
    assert server.user_sessions[target.session]["cart"] == ["a"]
    assert "slot-3-42" in target.browser.current_url
    assert store.drop(snapshot_id) is True
    assert store.drop(snapshot_id) is False


def test_shopper_clone_keeps_history_but_is_trajectory_local():
    shopper = ControlledShopper(
        object(), initial_request="request", allowed_facts=["fact"], max_questions=2
    )
    shopper.history = [{"question": "q1", "answer": "a1"}]
    shopper.call_count = 1
    clone = shopper.clone()
    clone.history[0]["answer"] = "changed"
    clone.history.append({"question": "q2", "answer": "a2"})
    assert shopper.history == [{"question": "q1", "answer": "a1"}]
    assert shopper.call_count == clone.call_count == 1
