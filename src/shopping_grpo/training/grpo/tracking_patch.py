"""Deterministic veRL 0.8 tracking and SwanLab-dashboard transform."""

PATCH_MARKER = "SHOPPING_GRPO_TRACKING_DASHBOARD_PATCH_V2"
LEGACY_PATCH_MARKERS = ("SHOPPING_GRPO_TRACKING_FINISH_PATCH_V1",)

ORIGINAL_LOG = '''    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)
'''

DASHBOARD_LOG = '''    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                backend_data = data
                if default_backend == "swanlab":
                    from shopping_grpo.training.grpo.dynamic_sampling import (
                        swanlab_dashboard_metrics,
                    )

                    backend_data = swanlab_dashboard_metrics(data)
                if backend_data:
                    logger_instance.log(data=backend_data, step=step)
'''

ORIGINAL_CLOSE = '''    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()
        if "clearml" in self.logger:
            self.logger["clearml"].finish()
        if "trackio" in self.logger:
            self.logger["trackio"].finish()
        if "file" in self.logger:
            self.logger["file"].finish()
'''

FINISH = '''    def finish(self):
        """Finish tracking once, from the trainer thread when possible."""
        if getattr(self, "_shopping_finished", False):
            return
        self._shopping_finished = True
        backends = getattr(self, "logger", {})
        try:
            if "wandb" in backends:
                backends["wandb"].finish(exit_code=0)
            if "swanlab" in backends:
                backends["swanlab"].finish()
            if "vemlp_wandb" in backends:
                backends["vemlp_wandb"].finish(exit_code=0)
            if "tensorboard" in backends:
                backends["tensorboard"].finish()
            if "clearml" in backends:
                backends["clearml"].finish()
            if "trackio" in backends:
                backends["trackio"].finish()
            if "file" in backends:
                backends["file"].finish()
        except Exception as exc:
            logger.warning("experiment tracker finish failed: %s", exc)

    def __del__(self):
        self.finish()
'''


def patch_source(source: str) -> str:
    """Add an idempotent finish contract and SwanLab-only metric projection."""
    if PATCH_MARKER in source:
        return source
    anchor = "        self.logger = {}\n"
    if (
        any(marker in source for marker in LEGACY_PATCH_MARKERS)
        or source.count(anchor) != 1
        or source.count(ORIGINAL_LOG) != 1
        or source.count(ORIGINAL_CLOSE) != 1
    ):
        raise ValueError("pinned veRL Tracking source anchors mismatch")
    source = source.replace(
        anchor,
        anchor + f"        # {PATCH_MARKER}\n        self._shopping_finished = False\n",
        1,
    )
    source = source.replace(ORIGINAL_LOG, DASHBOARD_LOG, 1)
    return source.replace(ORIGINAL_CLOSE, FINISH, 1)
