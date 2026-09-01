import logging

import scripts.apply_verl_tracking_finish_patch as installer
from scripts.check_grpo_runtime import validate_tracking_finish_patch
from shopping_grpo.training.grpo.tracking_patch import (
    FINISH,
    LEGACY_PATCH_MARKERS,
    ORIGINAL_CLOSE,
    ORIGINAL_LOG,
    PATCH_MARKER,
    patch_source,
)

SOURCE = '''import logging
logger = logging.getLogger(__name__)

class Tracking:
    def __init__(self):
        self.logger = {}

''' + ORIGINAL_LOG + "\n" + ORIGINAL_CLOSE


class Backend:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error
        self.logged = []

    def log(self, *, data, step):
        self.logged.append((data, step))

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


def test_tracking_projects_only_swanlab_into_five_dashboard_sections():
    tracking = tracking_class()()
    console = Backend()
    swanlab = Backend()
    tracking.logger.update({"console": console, "swanlab": swanlab})
    raw = {
        "val-shopping/summary/strict_success_rate": 0.64,
        "bpo_batch/root_groups": 1,
        "bpo_return/sibling_range_mean": 0.5,
        "actor/grad_norm": 1.2,
        "bpo_cost/shopper_api_calls_total": 18,
        "reward/shaped_mean": 0.7,
    }

    tracking.log(raw, step=10)

    assert console.logged == [(raw, 10)]
    dashboard, step = swanlab.logged[0]
    assert step == 10
    assert dashboard == {
        "validation/gold_purchase_success": 0.64,
        "sampling/accepted_root_groups": 1,
        "credit/sibling_return_range_mean": 0.5,
        "optimization/grad_norm": 1.2,
        "runtime/shopper_api_calls_total": 18,
    }


def test_tracking_patch_installer_apply_check_restore(tmp_path, monkeypatch):
    target = tmp_path / "tracking.py"
    target.write_text(SOURCE, encoding="utf-8", newline="\n")
    monkeypatch.setattr(installer, "EXPECTED_ORIGINAL_SHA256", installer.sha256(target))
    installer.apply(target)
    installer.verify(target)
    installer.apply(target)
    installer.restore(target)
    assert target.read_text(encoding="utf-8") == SOURCE


def test_tracking_patch_installer_upgrades_legacy_patch_from_backup(
    tmp_path, monkeypatch
):
    target = tmp_path / "tracking.py"
    target.write_text(SOURCE, encoding="utf-8", newline="\n")
    backup = target.with_name(target.name + installer.BACKUP_SUFFIX)
    backup.write_text(SOURCE, encoding="utf-8", newline="\n")
    legacy_source = SOURCE.replace(
        "        self.logger = {}\n",
        "        self.logger = {}\n"
        f"        # {LEGACY_PATCH_MARKERS[0]}\n"
        "        self._shopping_finished = False\n",
        1,
    ).replace(ORIGINAL_CLOSE, FINISH, 1)
    target.write_text(legacy_source, encoding="utf-8", newline="\n")
    monkeypatch.setattr(installer, "EXPECTED_ORIGINAL_SHA256", installer.sha256(backup))

    installer.apply(target)

    installer.verify(target)
    assert PATCH_MARKER in target.read_text(encoding="utf-8")


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
