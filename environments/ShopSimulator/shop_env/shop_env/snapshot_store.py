"""Server-side snapshots for BPO."""

from copy import deepcopy
import threading
from uuid import uuid4

_ENV_FIELDS = (
    "idx", "instruction_text", "instruction_simple", "goal_options", "prev_obs",
    "prev_actions", "history", "user_persona", "reason_key",
)


def _rewrite_session(value, old_session, new_session):
    if isinstance(value, str):
        return value.replace(str(old_session), str(new_session))
    return value


class SnapshotStore:
    """Keep simulator state server-side and expose only random snapshot ids."""

    def __init__(self):
        self._lock = threading.RLock()
        self._snapshots = {}

    def clear(self):
        with self._lock:
            self._snapshots.clear()

    def create(self, env, env_idx):
        session = env.session
        if session is None or session not in env.server.user_sessions:
            raise RuntimeError("cannot snapshot an environment without an active session")
        snapshot_id = uuid4().hex
        payload = {
            "session": deepcopy(session),
            "server_session": deepcopy(env.server.user_sessions[session]),
            "browser": {
                "current_url": deepcopy(env.browser.current_url),
                "page_source": deepcopy(env.browser.page_source),
            },
            "environment": {
                name: deepcopy(getattr(env, name))
                for name in _ENV_FIELDS if hasattr(env, name)
            },
        }
        with self._lock:
            self._snapshots[snapshot_id] = payload
        return snapshot_id

    def clone_into(self, snapshot_id, target_env, target_env_idx):
        with self._lock:
            if snapshot_id not in self._snapshots:
                raise KeyError("unknown or expired snapshot id")
            payload = deepcopy(self._snapshots[snapshot_id])
        old_session = payload["session"]
        new_session = f"slot-{int(target_env_idx)}-{str(old_session).split('-')[-1]}"
        target_env.server.user_sessions[new_session] = payload["server_session"]
        target_env.session = new_session
        for name, value in payload["environment"].items():
            setattr(target_env, name, value)
        target_env.browser.session_id = new_session
        target_env.browser.current_url = _rewrite_session(
            payload["browser"]["current_url"], old_session, new_session
        )
        target_env.browser.page_source = _rewrite_session(
            payload["browser"]["page_source"], old_session, new_session
        )
        # BeautifulSoup Tags contain cyclic parent links. text_to_clickable is
        # derived from page_source, so rebuild it instead of snapshotting it.
        target_env.text_to_clickable = None
        target_env.get_available_actions()
        return {
            "env_idx": int(target_env_idx),
            "snapshot_id": snapshot_id,
            "done": bool(target_env.server.user_sessions[new_session].get("done", False)),
        }

    def drop(self, snapshot_id):
        with self._lock:
            return self._snapshots.pop(snapshot_id, None) is not None
