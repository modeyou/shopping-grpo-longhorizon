import asyncio
from types import SimpleNamespace

from shopping_grpo.training.bpo.session import ClonedBranchSession
from shopping_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    current_shopper,
)
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


def test_reused_slot_task_session_clears_terminal_residue_before_snapshot():
    import sys
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parents[1]
        / "environments/ShopSimulator/shop_env/shop_env"
    )
    sys.path.insert(0, str(module_path))
    try:
        from snapshot_store import SnapshotStore, reset_terminal_session_state
    finally:
        sys.path.remove(str(module_path))

    session_id = "slot-4-10284"
    server = SimpleNamespace(
        user_sessions={
            session_id: {
                "goal": {"instruction_text": "goal"},
                "done": True,
                "reward": 1.0,
                "reward_detail": {"reward_type": "gold_purchase"},
                "verbose_info": {"stale": True},
                "cart": [],
            }
        }
    )
    source = SimpleNamespace(
        session=session_id,
        server=server,
        browser=SimpleNamespace(
            session_id=session_id,
            current_url=f"http://shop/{session_id}",
            page_source=f"{session_id} page",
        ),
        idx=10284,
        history=[],
        get_available_actions=lambda: None,
    )

    terminal_store = SnapshotStore()
    terminal_snapshot = terminal_store.create(source, 4)
    terminal_target = SimpleNamespace(
        server=server,
        browser=SimpleNamespace(),
        get_available_actions=lambda: None,
    )
    terminal_result = terminal_store.clone_into(
        terminal_snapshot, terminal_target, 6
    )
    assert terminal_result["done"] is True

    reset_terminal_session_state(source)
    fresh = server.user_sessions[session_id]
    assert fresh["done"] is False
    assert "reward" not in fresh
    assert "reward_detail" not in fresh
    assert "verbose_info" not in fresh

    store = SnapshotStore()
    snapshot_id = store.create(source, 4)
    target = SimpleNamespace(
        server=server,
        browser=SimpleNamespace(),
        get_available_actions=lambda: None,
    )
    result = store.clone_into(snapshot_id, target, 5)
    assert result["done"] is False
    assert server.user_sessions[target.session]["done"] is False


def test_cloned_branch_session_binds_and_releases_coroutine_local_state():
    clone = SimpleNamespace(release_count=0)

    def release():
        clone.release_count += 1

    clone.release = release
    source = SimpleNamespace(clone=lambda snapshot_id: clone)
    state = SimpleNamespace(name="state")
    shopper = SimpleNamespace(name="shopper")

    async def exercise():
        previous = (
            current_environment.get(),
            current_runtime_state.get(),
            current_shopper.get(),
        )
        session = ClonedBranchSession(source, "snapshot-1", state, shopper)
        await session.start()
        assert current_environment.get() is clone
        assert current_runtime_state.get() is state
        assert current_shopper.get() is shopper
        await session.close()
        assert (
            current_environment.get(),
            current_runtime_state.get(),
            current_shopper.get(),
        ) == previous

    asyncio.run(exercise())
    assert clone.release_count == 1


def test_cloned_branch_session_rejects_terminal_runtime_state_before_clone():
    source = SimpleNamespace(clone=lambda snapshot_id: (_ for _ in ()).throw(
        AssertionError("terminal state must be rejected before cloning")
    ))

    async def exercise():
        session = ClonedBranchSession(
            source,
            "snapshot-terminal",
            {"done": True, "terminate": True},
            SimpleNamespace(),
        )
        try:
            await session.start()
        except RuntimeError as exc:
            assert "terminal runtime state" in str(exc)
        else:
            raise AssertionError("terminal runtime state was accepted")

    asyncio.run(exercise())


def test_cloned_branch_session_releases_terminal_restored_environment():
    clone = SimpleNamespace(done=True, release_count=0)

    def release():
        clone.release_count += 1

    clone.release = release
    source = SimpleNamespace(clone=lambda snapshot_id: clone)

    async def exercise():
        session = ClonedBranchSession(
            source,
            "snapshot-terminal-env",
            {"done": False, "terminate": False},
            SimpleNamespace(),
        )
        try:
            await session.start()
        except RuntimeError as exc:
            assert "terminal environment" in str(exc)
        else:
            raise AssertionError("terminal restored environment was accepted")

    asyncio.run(exercise())
    assert clone.release_count == 1
