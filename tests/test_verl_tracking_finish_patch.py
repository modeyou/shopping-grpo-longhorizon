import logging

import scripts.apply_verl_tracking_finish_patch as installer
from scripts.check_grpo_runtime import validate_tracking_finish_patch
from shopping_grpo.training.grpo.tracking_patch import (
    ORIGINAL_CLOSE,
    PATCH_MARKER,
    patch_source,
)

SOURCE = '''import logging
logger = logging.getLogger(__name__)

class Tracking:
    def __init__(self):
        self.logger = {}

''' + ORIGINAL_CLOSE


class Backend:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    def finish(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error


def tracking_class():
    namespace = {}
    exec(patch_source(SOURCE), namespace)
    return namespace["Tracking"]


def test_tracking_finish_is_explicit_and_idempotent():
    patched = patch_source(SOURCE)
    assert patched.count(PATCH_MARKER) == 1
    assert patch_source(patched) == patched
    tracking = tracking_class()()
    swanlab = Backend()
    tracking.logger["swanlab"] = swanlab
    tracking.finish()
    tracking.finish()
    tracking.__del__()
    assert swanlab.calls == 1


def test_tracking_finish_failure_is_nonfatal(caplog):
    tracking = tracking_class()()
    tracking.logger["swanlab"] = Backend(RuntimeError("cannot join current thread"))
    with caplog.at_level(logging.WARNING):
        tracking.finish()
    assert "cannot join current thread" in caplog.text


def test_tracking_patch_installer_apply_check_restore(tmp_path, monkeypatch):
    target = tmp_path / "tracking.py"
    target.write_text(SOURCE, encoding="utf-8", newline="\n")
    monkeypatch.setattr(installer, "EXPECTED_ORIGINAL_SHA256", installer.sha256(target))
    installer.apply(target)
    installer.verify(target)
    installer.apply(target)
    installer.restore(target)
    assert target.read_text(encoding="utf-8") == SOURCE


def test_runtime_preflight_accepts_patched_tracking(tmp_path, monkeypatch):
    package = tmp_path / "verl"
    target = package / "utils" / "tracking.py"
    target.parent.mkdir(parents=True)
    target.write_text(SOURCE, encoding="utf-8", newline="\n")
    verl_source = package / "__init__.py"
    verl_source.write_text("", encoding="utf-8")
    monkeypatch.setattr(installer, "EXPECTED_ORIGINAL_SHA256", installer.sha256(target))
    installer.apply(target)
    validate_tracking_finish_patch(verl_source)
